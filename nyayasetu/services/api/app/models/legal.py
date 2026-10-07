"""SQLAlchemy 2 async ORM models for the legal knowledge base.

Tables
------
legal_documents   – one row per Act / statute (with versioning)
legal_sections    – parsed sections of an Act (effective_from / repealed_on)
section_chunks    – text chunks ready for vector + full-text retrieval

Indexes
-------
* section_chunks.embedding  → pgvector HNSW (cosine) — fast ANN search
* section_chunks.tsv        → PostgreSQL GIN          — full-text search
"""
from __future__ import annotations

import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Date,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ── Embedding dimension produced by the configured embedding model ────────────
# nomic-embed-text-v1.5  → 768 dims
# text-embedding-ada-002 → 1536 dims
# Change here + re-run Alembic if you switch models.
EMBEDDING_DIM: int = 768


class Base(DeclarativeBase):
    """Common declarative base shared by all ORM models."""


# ─────────────────────────────────────────────────────────────────────────────
# legal_documents
# ─────────────────────────────────────────────────────────────────────────────
class LegalDocument(Base):
    """One record per Act/statute per version.

    A new ``version`` row is inserted when an amendment is ingested; the old
    row remains so that historical queries still resolve correctly.

    Columns
    -------
    act_name        – canonical short name, e.g. "Indian Contract Act"
    full_title      – official long title
    act_number      – e.g. "9 of 1872"
    jurisdiction    – ISO 3166-1 alpha-2 or state code, e.g. "IN", "IN-MH"
    version         – monotonically increasing integer (1 = original)
    effective_from  – date on which this version came into force
    repealed_on     – date repealed/superseded, NULL if still in force
    source_url      – canonical source (India Code, etc.)
    checksum        – SHA-256 of raw source text; used to detect duplicates
    """

    __tablename__ = "legal_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    act_name: Mapped[str] = mapped_column(String(255), nullable=False)
    full_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    act_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    jurisdiction: Mapped[str] = mapped_column(String(10), nullable=False)
    version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    effective_from: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    repealed_on: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # relationships
    sections: Mapped[list[LegalSection]] = relationship(
        "LegalSection", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Prevent re-ingesting the same (act, jurisdiction, version) combination.
        UniqueConstraint(
            "act_name", "jurisdiction", "version", name="uq_document_act_jur_ver"
        ),
        Index("ix_legal_documents_act_name", "act_name"),
        Index("ix_legal_documents_jurisdiction", "jurisdiction"),
        Index("ix_legal_documents_effective_from", "effective_from"),
    )

    def is_in_force(self, on: datetime.date | None = None) -> bool:
        """Return True if the document is in force on *on* (defaults to today)."""
        check_date = on or datetime.date.today()
        if check_date < self.effective_from:
            return False
        return self.repealed_on is None or check_date < self.repealed_on

    def __repr__(self) -> str:
        return (
            f"LegalDocument(id={self.id}, act={self.act_name!r}, "
            f"jurisdiction={self.jurisdiction!r}, version={self.version})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# legal_sections
# ─────────────────────────────────────────────────────────────────────────────
class LegalSection(Base):
    """One row per section of an Act in a specific document version.

    Columns
    -------
    section_number  – e.g. "10", "10A", "10B"
    heading         – short heading extracted by the parser
    content         – full section text
    is_repealed     – True if the parser identified this section as repealed/omitted
    effective_from  – inherited from parent document unless overridden by amendment
    repealed_on     – section-level repeal date (may differ from document-level)
    """

    __tablename__ = "legal_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False
    )
    section_number: Mapped[str] = mapped_column(String(20), nullable=False)
    heading: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_repealed: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    effective_from: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    repealed_on: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)

    # relationships
    document: Mapped[LegalDocument] = relationship(
        "LegalDocument", back_populates="sections"
    )
    chunks: Mapped[list[SectionChunk]] = relationship(
        "SectionChunk", back_populates="section", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_legal_sections_document_id", "document_id"),
        Index("ix_legal_sections_section_number", "section_number"),
        Index(
            "ix_legal_sections_effective_from_repealed_on",
            "effective_from",
            "repealed_on",
        ),
    )

    def is_in_force(self, on: datetime.date | None = None) -> bool:
        check_date = on or datetime.date.today()
        if check_date < self.effective_from:
            return False
        return self.repealed_on is None or check_date < self.repealed_on

    def __repr__(self) -> str:
        return (
            f"LegalSection(id={self.id}, doc={self.document_id}, "
            f"s={self.section_number!r}, repealed={self.is_repealed})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# section_chunks
# ─────────────────────────────────────────────────────────────────────────────
class SectionChunk(Base):
    """Sub-section text chunk with a dense embedding and a tsvector for BM25-style search.

    Columns
    -------
    chunk_index  – 0-based position within the section (for ordering)
    text         – raw chunk text (may overlap with adjacent chunks)
    embedding    – pgvector Vector(EMBEDDING_DIM); NULL until embedded
    tsv          – TSVECTOR computed from ``text``; populated by a DB trigger
                   defined in the Alembic migration (not a Computed() column so
                   that the chunker can set it explicitly during bulk inserts for
                   performance, but the trigger keeps it consistent otherwise)
    char_start   – character offset in section.content (for provenance)
    char_end     – character offset in section.content (for provenance)

    Indexes
    -------
    HNSW  on embedding with cosine distance – for vector ANN search
    GIN   on tsv                             – for full-text search
    """

    __tablename__ = "section_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("legal_sections.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    tsv: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    char_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # relationship
    section: Mapped[LegalSection] = relationship(
        "LegalSection", back_populates="chunks"
    )

    __table_args__ = (
        # ── pgvector HNSW index (cosine distance) ──────────────────────────
        # m=16, ef_construction=64 are safe defaults; tune after benchmarking.
        Index(
            "ix_section_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": "16", "ef_construction": "64"},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # ── GIN index for tsvector full-text search ─────────────────────────
        Index(
            "ix_section_chunks_tsv_gin",
            "tsv",
            postgresql_using="gin",
        ),
        Index("ix_section_chunks_section_id", "section_id"),
    )

    def __repr__(self) -> str:
        return (
            f"SectionChunk(id={self.id}, section={self.section_id}, "
            f"idx={self.chunk_index}, chars={self.char_start}-{self.char_end})"
        )
