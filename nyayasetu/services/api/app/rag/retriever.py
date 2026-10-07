"""Hybrid retrieval engine for NyayaSetu combining pgvector ANN and Postgres BM25.

Architecture
------------
1. Vector Search: Dense semantic similarity via pgvector (HNSW cosine distance).
2. BM25 Search: Sparse keyword match via PostgreSQL tsvector & ts_rank_cd.
3. Reciprocal Rank Fusion (RRF): Combines ranks from vector and BM25 search.
4. Reranker Interface: Optional post-retrieval reranking (e.g. cross-encoder).
5. Metadata Filtering: Jurisdiction exact/array match, effective_from/repealed_on date bounds.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from sqlalchemy import Date, String, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.legal import LegalDocument, LegalSection, SectionChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Dataclass representing a retrieved legal chunk with metadata and scores."""

    chunk_id: int
    section_id: int
    document_id: int
    act_name: str
    section_number: str
    heading: str | None
    content: str  # Full section content
    chunk_text: str  # Specific chunk text
    jurisdiction: str
    effective_from: datetime.date
    repealed_on: datetime.date | None
    is_repealed: bool
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    source_type: str = "hybrid"  # "hybrid", "vector", "bm25"


@dataclass
class RetrievalFilter:
    """Filter parameters for legal retrieval."""

    jurisdiction: str | list[str] | None = None
    as_of_date: datetime.date | None = None
    act_names: list[str] | None = None
    include_repealed: bool = False


class BaseReranker(Protocol):
    """Protocol for reranking search results."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        ...


class NoOpReranker:
    """Default pass-through reranker that maintains current score order."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker:
    """Pluggable cross-encoder reranker interface.
    
    If model pipeline is available, rescores (query, chunk_text) pairs.
    Otherwise falls back gracefully to standard scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        
        # If model is loaded or sentence_transformers available, rescore
        if self._model is not None:
            try:
                pairs = [[query, r.chunk_text] for r in results]
                scores = self._model.predict(pairs)
                scored_results = []
                for res, score in zip(results, scores):
                    # Replace score with reranker score
                    scored_results.append(
                        SearchResult(
                            chunk_id=res.chunk_id,
                            section_id=res.section_id,
                            document_id=res.document_id,
                            act_name=res.act_name,
                            section_number=res.section_number,
                            heading=res.heading,
                            content=res.content,
                            chunk_text=res.chunk_text,
                            jurisdiction=res.jurisdiction,
                            effective_from=res.effective_from,
                            repealed_on=res.repealed_on,
                            is_repealed=res.is_repealed,
                            score=float(score),
                            vector_rank=res.vector_rank,
                            bm25_rank=res.bm25_rank,
                            source_type=res.source_type,
                        )
                    )
                scored_results.sort(key=lambda r: r.score, reverse=True)
                return scored_results[:top_k]
            except Exception as exc:
                logger.warning("CrossEncoder reranking failed, using default scores: %s", exc)
        
        return results[:top_k]


class Embedder(Protocol):
    """Protocol for embedding queries."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HybridRetriever:
    """Hybrid Legal Retriever combining vector ANN search and BM25 text search with RRF."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.reranker = reranker or NoOpReranker()

    def _build_filter_conditions(
        self, filter_: RetrievalFilter | None
    ) -> list[Any]:
        """Build SQLAlchemy filter expressions for document & section metadata."""
        if not filter_:
            return []

        conditions: list[Any] = []

        # 1. Jurisdiction filter
        if filter_.jurisdiction:
            if isinstance(filter_.jurisdiction, list):
                conditions.append(LegalDocument.jurisdiction.in_(filter_.jurisdiction))
            else:
                conditions.append(LegalDocument.jurisdiction == filter_.jurisdiction)

        # 2. Act names filter
        if filter_.act_names:
            conditions.append(LegalDocument.act_name.in_(filter_.act_names))

        # 3. As-of date filter (effective_from <= as_of_date < repealed_on)
        if filter_.as_of_date:
            dt = filter_.as_of_date
            conditions.append(LegalDocument.effective_from <= dt)
            conditions.append(
                or_(LegalDocument.repealed_on.is_(None), LegalDocument.repealed_on > dt)
            )
            conditions.append(LegalSection.effective_from <= dt)
            conditions.append(
                or_(LegalSection.repealed_on.is_(None), LegalSection.repealed_on > dt)
            )

        # 4. Repealed status filter
        if not filter_.include_repealed:
            conditions.append(LegalSection.is_repealed.is_(False))

        return conditions

    async def _vector_search(
        self,
        query_vector: list[float],
        filter_conditions: list[Any],
        limit: int = 30,
    ) -> list[tuple[SectionChunk, LegalSection, LegalDocument, float]]:
        """Dense vector search using pgvector cosine distance."""
        distance_expr = SectionChunk.embedding.cosine_distance(query_vector)

        stmt = (
            select(SectionChunk, LegalSection, LegalDocument, distance_expr.label("distance"))
            .join(LegalSection, SectionChunk.section_id == LegalSection.id)
            .join(LegalDocument, LegalSection.document_id == LegalDocument.id)
            .where(SectionChunk.embedding.is_not(None))
        )

        if filter_conditions:
            stmt = stmt.where(and_(*filter_conditions))

        stmt = stmt.order_by(distance_expr).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.all())

    async def _bm25_search(
        self,
        query_text: str,
        filter_conditions: list[Any],
        limit: int = 30,
    ) -> list[tuple[SectionChunk, LegalSection, LegalDocument, float]]:
        """Sparse BM25 search using PostgreSQL tsvector and ts_rank_cd."""
        ts_query = func.plainto_tsquery("english", query_text)
        rank_expr = func.ts_rank_cd(SectionChunk.tsv, ts_query)

        stmt = (
            select(SectionChunk, LegalSection, LegalDocument, rank_expr.label("rank"))
            .join(LegalSection, SectionChunk.section_id == LegalSection.id)
            .join(LegalDocument, LegalSection.document_id == LegalDocument.id)
            .where(SectionChunk.tsv.op("@@")(ts_query))
        )

        if filter_conditions:
            stmt = stmt.where(and_(*filter_conditions))

        stmt = stmt.order_by(rank_expr.desc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.all())

    @staticmethod
    def calculate_rrf(
        vector_results: list[tuple[SectionChunk, LegalSection, LegalDocument, float]],
        bm25_results: list[tuple[SectionChunk, LegalSection, LegalDocument, float]],
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Compute Reciprocal Rank Fusion (RRF) scores over vector and BM25 search results."""
        scores: dict[int, float] = {}
        vector_ranks: dict[int, int] = {}
        bm25_ranks: dict[int, int] = {}
        chunk_map: dict[int, tuple[SectionChunk, LegalSection, LegalDocument]] = {}

        # Process vector search ranks (1-indexed)
        for rank, (chunk, sec, doc, _dist) in enumerate(vector_results, start=1):
            cid = chunk.id
            chunk_map[cid] = (chunk, sec, doc)
            vector_ranks[cid] = rank
            scores[cid] = scores.get(cid, 0.0) + (vector_weight / (rrf_k + rank))

        # Process BM25 search ranks (1-indexed)
        for rank, (chunk, sec, doc, _rank) in enumerate(bm25_results, start=1):
            cid = chunk.id
            chunk_map[cid] = (chunk, sec, doc)
            bm25_ranks[cid] = rank
            scores[cid] = scores.get(cid, 0.0) + (bm25_weight / (rrf_k + rank))

        # Build list of SearchResult objects
        search_results: list[SearchResult] = []
        for cid, score in scores.items():
            chunk, sec, doc = chunk_map[cid]
            vrank = vector_ranks.get(cid)
            brank = bm25_ranks.get(cid)

            if vrank is not None and brank is not None:
                source_type = "hybrid"
            elif vrank is not None:
                source_type = "vector"
            else:
                source_type = "bm25"

            search_results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    section_id=sec.id,
                    document_id=doc.id,
                    act_name=doc.act_name,
                    section_number=sec.section_number,
                    heading=sec.heading,
                    content=sec.content,
                    chunk_text=chunk.text,
                    jurisdiction=doc.jurisdiction,
                    effective_from=sec.effective_from,
                    repealed_on=sec.repealed_on,
                    is_repealed=sec.is_repealed,
                    score=score,
                    vector_rank=vrank,
                    bm25_rank=brank,
                    source_type=source_type,
                )
            )

        # Sort combined candidates by RRF score descending
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    async def search(
        self,
        query: str,
        filter_: RetrievalFilter | None = None,
        top_k: int = 10,
        vector_top_k: int = 30,
        bm25_top_k: int = 30,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[SearchResult]:
        """Perform hybrid retrieval: vector search + BM25 text search -> RRF fusion -> Rerank."""
        if not query.strip():
            return []

        filter_conditions = self._build_filter_conditions(filter_)

        # 1. Generate query embedding
        embeddings = self.embedder.embed([query])
        if not embeddings or not embeddings[0]:
            logger.warning("Embedder returned empty vector for query: %r", query)
            vector_results: list[tuple[SectionChunk, LegalSection, LegalDocument, float]] = []
        else:
            query_vector = embeddings[0]
            vector_results = await self._vector_search(
                query_vector, filter_conditions, limit=vector_top_k
            )

        # 2. BM25 text search
        bm25_results = await self._bm25_search(
            query, filter_conditions, limit=bm25_top_k
        )

        # 3. Reciprocal Rank Fusion
        fused_results = self.calculate_rrf(
            vector_results=vector_results,
            bm25_results=bm25_results,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # 4. Rerank and return top K
        final_results = await self.reranker.rerank(
            query=query, results=fused_results, top_k=top_k
        )

        logger.info(
            "Hybrid search complete: query=%r results=%d (vector=%d, bm25=%d)",
            query,
            len(final_results),
            len(vector_results),
            len(bm25_results),
        )

        return final_results
