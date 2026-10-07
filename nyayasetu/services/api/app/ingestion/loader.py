"""Legal document ingestion loader.

Orchestrates the full pipeline for a single Act:

    raw text
        → ActSectionParser   (deterministic, from section_parser.py)
        → chunk_text         (deterministic chunker)
        → LLMClient.embed    (batched embedding via LM Studio)
        → SQLAlchemy models  (transactional DB write)

Design decisions
----------------
* The loader is stateless; inject dependencies via __init__.
* All DB writes happen in a single transaction; rollback on any failure.
* Duplicate detection uses a SHA-256 checksum of the raw text to avoid
  re-processing the exact same source file.
* Embedding is batched to respect LM Studio's per-request limits.
* ``sqlalchemy.dialects.postgresql.insert`` is used for the tsvector update
  so that the GIN index is always consistent without relying on a DB trigger.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import ChunkerConfig, TextChunk, chunk_text
from app.ingestion.exceptions import (
    DocumentAlreadyExistsError,
    EmbeddingError,
    IngestionError,
)
from app.ingestion.section_parser import ActSectionParser, ParsedSection, ParserConfig
from app.llm.llm_client import LLMClient
from app.models.legal import LegalDocument, LegalSection, SectionChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoaderConfig:
    """Runtime configuration for the document loader."""

    embedding_batch_size: int = 32
    chunker: ChunkerConfig = ChunkerConfig()


@dataclass(frozen=True)
class LoadResult:
    document_id: int
    sections_loaded: int
    chunks_loaded: int
    skipped_repealed: int


class LegalDocumentLoader:
    """Parse, chunk, embed, and persist a single Act into the DB.

    Parameters
    ----------
    session:
        An open ``AsyncSession`` (injected; caller manages lifecycle).
    llm_client:
        Any object that implements the ``LLMClient`` protocol.
    config:
        Loader-level tunables (batch size, chunker config).
    parser_config:
        Optional parser tunables forwarded to ``ActSectionParser``.
    """

    def __init__(
        self,
        session: AsyncSession,
        llm_client: LLMClient,
        config: LoaderConfig | None = None,
        parser_config: ParserConfig | None = None,
    ) -> None:
        self._session = session
        self._llm = llm_client
        self._config = config or LoaderConfig()
        self._parser = ActSectionParser(parser_config)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def load(
        self,
        *,
        act_name: str,
        full_title: str | None,
        act_number: str | None,
        jurisdiction: str,
        version: int,
        effective_from: date,
        repealed_on: date | None,
        source_url: str | None,
        raw_text: str,
        skip_repealed_sections: bool = True,
    ) -> LoadResult:
        """Full ingestion pipeline for one Act.

        Raises
        ------
        DocumentAlreadyExistsError
            If (act_name, jurisdiction, version) already exists with the same checksum.
        IngestionError
            For any other ingestion failure (wraps the root cause).
        """
        checksum = _sha256(raw_text)
        logger.info(
            "Starting ingestion: act=%r jur=%r ver=%d checksum=%.12s…",
            act_name,
            jurisdiction,
            version,
            checksum,
        )

        try:
            await self._assert_not_duplicate(act_name, jurisdiction, version, checksum)

            doc = await self._upsert_document(
                act_name=act_name,
                full_title=full_title,
                act_number=act_number,
                jurisdiction=jurisdiction,
                version=version,
                effective_from=effective_from,
                repealed_on=repealed_on,
                source_url=source_url,
                checksum=checksum,
            )

            sections = self._parser.parse(act_name, raw_text)
            logger.info("Parser extracted %d sections from %r.", len(sections), act_name)

            sections_loaded, chunks_loaded, skipped = await self._load_sections(
                doc=doc,
                sections=sections,
                effective_from=effective_from,
                skip_repealed=skip_repealed_sections,
            )

            await self._session.commit()
            logger.info(
                "Committed: doc_id=%d sections=%d chunks=%d skipped_repealed=%d",
                doc.id,
                sections_loaded,
                chunks_loaded,
                skipped,
            )
            return LoadResult(
                document_id=doc.id,
                sections_loaded=sections_loaded,
                chunks_loaded=chunks_loaded,
                skipped_repealed=skipped,
            )

        except (DocumentAlreadyExistsError, EmbeddingError):
            await self._session.rollback()
            raise
        except Exception as exc:
            await self._session.rollback()
            logger.exception("Ingestion failed for %r: %s", act_name, exc)
            raise IngestionError(f"Failed to ingest {act_name!r}") from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _assert_not_duplicate(
        self,
        act_name: str,
        jurisdiction: str,
        version: int,
        checksum: str,
    ) -> None:
        stmt = select(LegalDocument.id, LegalDocument.checksum).where(
            LegalDocument.act_name == act_name,
            LegalDocument.jurisdiction == jurisdiction,
            LegalDocument.version == version,
        )
        row = (await self._session.execute(stmt)).one_or_none()
        if row is not None:
            existing_id, existing_checksum = row
            if existing_checksum == checksum:
                logger.info(
                    "Exact duplicate detected (doc_id=%d) — skipping.", existing_id
                )
                raise DocumentAlreadyExistsError(act_name, jurisdiction, version)
            # Different checksum → same version but different content is a
            # data error; caller must bump the version number.
            raise DocumentAlreadyExistsError(act_name, jurisdiction, version)

    async def _upsert_document(
        self,
        *,
        act_name: str,
        full_title: str | None,
        act_number: str | None,
        jurisdiction: str,
        version: int,
        effective_from: date,
        repealed_on: date | None,
        source_url: str | None,
        checksum: str,
    ) -> LegalDocument:
        doc = LegalDocument(
            act_name=act_name,
            full_title=full_title,
            act_number=act_number,
            jurisdiction=jurisdiction,
            version=version,
            effective_from=effective_from,
            repealed_on=repealed_on,
            source_url=source_url,
            checksum=checksum,
        )
        self._session.add(doc)
        await self._session.flush()  # populate doc.id without committing
        return doc

    async def _load_sections(
        self,
        *,
        doc: LegalDocument,
        sections: list[ParsedSection],
        effective_from: date,
        skip_repealed: bool,
    ) -> tuple[int, int, int]:
        """Persist all sections + their chunks. Returns (sections, chunks, skipped)."""
        sections_loaded = 0
        chunks_loaded = 0
        skipped = 0

        # Collect all chunk texts across all sections for batched embedding.
        # We build the ORM objects first (without embeddings), then fill them in.
        section_orm_objects: list[LegalSection] = []
        chunk_orm_objects: list[list[SectionChunk]] = []

        for parsed in sections:
            if skip_repealed and parsed.repealed:
                skipped += 1
                logger.debug("Skipping repealed section %s.%s", doc.act_name, parsed.number)
                continue

            sec = LegalSection(
                document_id=doc.id,
                section_number=parsed.number,
                heading=parsed.heading or None,
                content=parsed.text,
                is_repealed=parsed.repealed,
                effective_from=effective_from,
                repealed_on=doc.repealed_on,
            )
            self._session.add(sec)
            await self._session.flush()  # populate sec.id

            text_chunks = chunk_text(parsed.text, self._config.chunker)
            if not text_chunks:
                logger.warning("Section %s produced no chunks — skipping.", parsed.number)
                continue

            chunk_objs = [
                SectionChunk(
                    section_id=sec.id,
                    chunk_index=c.index,
                    text=c.text,
                    embedding=None,  # filled after batched embed call
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
                for c in text_chunks
            ]
            section_orm_objects.append(sec)
            chunk_orm_objects.append(chunk_objs)
            sections_loaded += 1

        # Batched embedding pass.
        flat_chunks = [c for chunk_list in chunk_orm_objects for c in chunk_list]
        flat_texts = [c.text for c in flat_chunks]

        embeddings = await self._embed_batched(flat_texts)

        # Assign embeddings back and set tsvector.
        for chunk_obj, embedding in zip(flat_chunks, embeddings, strict=True):
            chunk_obj.embedding = embedding
            self._session.add(chunk_obj)
            chunks_loaded += 1

        # Bulk-update tsvectors via a single SQL call after flush.
        await self._session.flush()
        await self._refresh_tsvectors(doc.id)

        return sections_loaded, chunks_loaded, skipped

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Call LLMClient.embed in batches; raise EmbeddingError on failure."""
        if not texts:
            return []
        batch_size = self._config.embedding_batch_size
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug("Embedding batch %d/%d (%d texts)", i // batch_size + 1,
                         -(-len(texts) // batch_size), len(batch))
            try:
                vectors = self._llm.embed(batch)
            except Exception as exc:
                raise EmbeddingError(
                    f"embed() failed on batch starting at index {i}"
                ) from exc
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Expected {len(batch)} embeddings, got {len(vectors)}"
                )
            results.extend(vectors)
        return results

    async def _refresh_tsvectors(self, document_id: int) -> None:
        """Update tsv column for all chunks belonging to this document.

        Uses ``to_tsvector`` so the GIN index is populated immediately without
        requiring a separate trigger.
        """
        await self._session.execute(
            text(
                """
                UPDATE section_chunks sc
                   SET tsv = to_tsvector('english', sc.text)
                  FROM legal_sections ls
                 WHERE sc.section_id = ls.id
                   AND ls.document_id = :doc_id
                """
            ),
            {"doc_id": document_id},
        )
        logger.debug("tsvectors refreshed for document_id=%d", document_id)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
