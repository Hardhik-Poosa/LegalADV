"""Unit tests for HybridRetriever, RetrievalFilter, Reranker, and RecallEvaluator."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.legal import LegalDocument, LegalSection, SectionChunk
from app.rag.retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    NoOpReranker,
    RetrievalFilter,
    SearchResult,
)
from evaluation.evaluate_recall_at_k import EvalQuery, RecallEvaluator


# ── Helpers / Mock Data ───────────────────────────────────────────────────────
def make_mock_doc(doc_id: int = 1, act: str = "Indian Contract Act", jur: str = "IN") -> LegalDocument:
    doc = LegalDocument(
        id=doc_id,
        act_name=act,
        full_title=f"The {act}, 1872",
        act_number="9 of 1872",
        jurisdiction=jur,
        version=1,
        effective_from=datetime.date(1872, 9, 1),
        repealed_on=None,
    )
    return doc


def make_mock_sec(sec_id: int = 10, doc_id: int = 1, sec_num: str = "10") -> LegalSection:
    sec = LegalSection(
        id=sec_id,
        document_id=doc_id,
        section_number=sec_num,
        heading="What agreements are contracts",
        content="All agreements are contracts if they are made by the free consent of parties...",
        is_repealed=False,
        effective_from=datetime.date(1872, 9, 1),
        repealed_on=None,
    )
    return sec


def make_mock_chunk(chunk_id: int = 100, sec_id: int = 10, text: str = "All agreements are contracts...") -> SectionChunk:
    chunk = SectionChunk(
        id=chunk_id,
        section_id=sec_id,
        chunk_index=0,
        text=text,
        embedding=[0.1] * 768,
        tsv=None,
    )
    return chunk


# ── Test RRF Math ─────────────────────────────────────────────────────────────
class TestRRFScoreCalculation:
    def test_rrf_single_source_vector(self):
        doc = make_mock_doc(1)
        sec = make_mock_sec(10, 1)
        chunk = make_mock_chunk(100, 10)

        vector_results = [(chunk, sec, doc, 0.1)]
        bm25_results = []

        results = HybridRetriever.calculate_rrf(vector_results, bm25_results, rrf_k=60)
        assert len(results) == 1
        res = results[0]
        assert res.chunk_id == 100
        assert res.source_type == "vector"
        assert res.vector_rank == 1
        assert res.bm25_rank is None
        # RRF score: 1 / (60 + 1) = 1 / 61
        assert pytest.approx(res.score, 1e-5) == 1.0 / 61.0

    def test_rrf_single_source_bm25(self):
        doc = make_mock_doc(1)
        sec = make_mock_sec(10, 1)
        chunk = make_mock_chunk(100, 10)

        vector_results = []
        bm25_results = [(chunk, sec, doc, 0.5)]

        results = HybridRetriever.calculate_rrf(vector_results, bm25_results, rrf_k=60)
        assert len(results) == 1
        res = results[0]
        assert res.chunk_id == 100
        assert res.source_type == "bm25"
        assert res.vector_rank is None
        assert res.bm25_rank == 1
        assert pytest.approx(res.score, 1e-5) == 1.0 / 61.0

    def test_rrf_hybrid_fusion_higher_score(self):
        doc = make_mock_doc(1)
        sec1 = make_mock_sec(10, 1, "10")
        sec2 = make_mock_sec(11, 1, "11")
        chunk1 = make_mock_chunk(100, 10, "Chunk 1")
        chunk2 = make_mock_chunk(101, 11, "Chunk 2")

        # chunk1 is rank 1 in vector, rank 2 in bm25
        # chunk2 is rank 2 in vector, rank 1 in bm25
        vector_results = [(chunk1, sec1, doc, 0.1), (chunk2, sec2, doc, 0.2)]
        bm25_results = [(chunk2, sec2, doc, 0.9), (chunk1, sec1, doc, 0.8)]

        results = HybridRetriever.calculate_rrf(
            vector_results, bm25_results, rrf_k=60, vector_weight=1.0, bm25_weight=1.0
        )
        assert len(results) == 2
        # Both appear in vector and bm25 -> hybrid
        for r in results:
            assert r.source_type == "hybrid"
            # score = 1/(60+1) + 1/(60+2)
            assert pytest.approx(r.score, 1e-5) == (1.0 / 61.0) + (1.0 / 62.0)

    def test_rrf_weighted_scores(self):
        doc = make_mock_doc(1)
        sec = make_mock_sec(10, 1)
        chunk = make_mock_chunk(100, 10)

        vector_results = [(chunk, sec, doc, 0.1)]
        bm25_results = []

        results = HybridRetriever.calculate_rrf(
            vector_results, bm25_results, rrf_k=60, vector_weight=2.0
        )
        assert len(results) == 1
        assert pytest.approx(results[0].score, 1e-5) == 2.0 / 61.0


# ── Test Filter Conditions Construction ─────────────────────────────────────
class TestFilterConstruction:
    def test_empty_filter(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        retriever = HybridRetriever(mock_session, mock_embedder)

        conds = retriever._build_filter_conditions(None)
        assert len(conds) == 0

    def test_jurisdiction_string(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        retriever = HybridRetriever(mock_session, mock_embedder)

        filt = RetrievalFilter(jurisdiction="IN-MH")
        conds = retriever._build_filter_conditions(filt)
        assert len(conds) == 2  # jurisdiction equality + is_repealed==False

    def test_jurisdiction_list(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        retriever = HybridRetriever(mock_session, mock_embedder)

        filt = RetrievalFilter(jurisdiction=["IN", "IN-MH"])
        conds = retriever._build_filter_conditions(filt)
        assert len(conds) == 2

    def test_as_of_date_filter(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        retriever = HybridRetriever(mock_session, mock_embedder)

        dt = datetime.date(2025, 1, 1)
        filt = RetrievalFilter(as_of_date=dt)
        conds = retriever._build_filter_conditions(filt)
        # Should include document effective_from, document repealed_on, section effective_from, section repealed_on, section is_repealed
        assert len(conds) == 5


# ── Test HybridRetriever Search ──────────────────────────────────────────────
@pytest.mark.asyncio
class TestHybridRetrieverSearch:
    async def test_empty_query(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        retriever = HybridRetriever(mock_session, mock_embedder)

        results = await retriever.search("")
        assert results == []

    async def test_embedder_returns_empty(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = []  # Empty embedding list

        # Mock DB execute for BM25
        doc = make_mock_doc(1)
        sec = make_mock_sec(10, 1)
        chunk = make_mock_chunk(100, 10)

        bm25_db_row = (chunk, sec, doc, 0.5)
        mock_execute_result = MagicMock()
        mock_execute_result.all.return_value = [bm25_db_row]
        mock_session.execute.return_value = mock_execute_result

        retriever = HybridRetriever(mock_session, mock_embedder)
        results = await retriever.search("contract definition")

        assert len(results) == 1
        assert results[0].chunk_id == 100
        assert results[0].source_type == "bm25"

    async def test_full_hybrid_search(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 768]

        doc = make_mock_doc(1)
        sec = make_mock_sec(10, 1)
        chunk = make_mock_chunk(100, 10)

        db_row = (chunk, sec, doc, 0.2)
        mock_execute_result = MagicMock()
        mock_execute_result.all.return_value = [db_row]
        mock_session.execute.return_value = mock_execute_result

        retriever = HybridRetriever(mock_session, mock_embedder)
        results = await retriever.search("contract consent")

        assert len(results) == 1
        res = results[0]
        assert res.chunk_id == 100
        assert res.act_name == "Indian Contract Act"
        assert res.section_number == "10"
        assert res.source_type == "hybrid"


# ── Test Rerankers ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestReranker:
    async def test_noop_reranker_limits_top_k(self):
        reranker = NoOpReranker()
        results = [
            SearchResult(
                chunk_id=i,
                section_id=i,
                document_id=1,
                act_name="Act",
                section_number=str(i),
                heading=None,
                content="text",
                chunk_text="text",
                jurisdiction="IN",
                effective_from=datetime.date(1872, 1, 1),
                repealed_on=None,
                is_repealed=False,
                score=1.0 - (i * 0.1),
            )
            for i in range(10)
        ]
        reranked = await reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3
        assert reranked[0].chunk_id == 0

    async def test_cross_encoder_fallback(self):
        reranker = CrossEncoderReranker()
        # Model is None, should fall back gracefully
        results = [
            SearchResult(
                chunk_id=1,
                section_id=1,
                document_id=1,
                act_name="Act",
                section_number="1",
                heading=None,
                content="text",
                chunk_text="text",
                jurisdiction="IN",
                effective_from=datetime.date(1872, 1, 1),
                repealed_on=None,
                is_repealed=False,
                score=0.9,
            )
        ]
        reranked = await reranker.rerank("query", results, top_k=5)
        assert len(reranked) == 1
        assert reranked[0].score == 0.9


# ── Test Evaluation Script ────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestRecallEvaluator:
    async def test_evaluator_metrics_hit(self):
        evaluator = RecallEvaluator(k_values=[1, 3, 5])
        mock_retriever = AsyncMock()

        # Retriever returns search result matching target query
        hit_result = SearchResult(
            chunk_id=1,
            section_id=10,
            document_id=1,
            act_name="Indian Contract Act",
            section_number="10",
            heading=None,
            content="text",
            chunk_text="text",
            jurisdiction="IN",
            effective_from=datetime.date(1872, 1, 1),
            repealed_on=None,
            is_repealed=False,
            score=0.95,
        )
        mock_retriever.search.return_value = [hit_result]

        queries = [
            EvalQuery(
                query_id="q1",
                query_text="Contract elements",
                target_act="Indian Contract Act",
                target_section="10",
            )
        ]

        metrics = await evaluator.evaluate_dataset(mock_retriever, queries)
        assert metrics.total_queries == 1
        assert metrics.recall_at_k[1] == 1.0
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.mrr == 1.0

    async def test_evaluator_metrics_miss(self):
        evaluator = RecallEvaluator(k_values=[1, 3, 5])
        mock_retriever = AsyncMock()

        # Retriever returns section 99 instead of expected 10
        miss_result = SearchResult(
            chunk_id=1,
            section_id=99,
            document_id=1,
            act_name="Indian Contract Act",
            section_number="99",
            heading=None,
            content="text",
            chunk_text="text",
            jurisdiction="IN",
            effective_from=datetime.date(1872, 1, 1),
            repealed_on=None,
            is_repealed=False,
            score=0.95,
        )
        mock_retriever.search.return_value = [miss_result]

        queries = [
            EvalQuery(
                query_id="q1",
                query_text="Contract elements",
                target_act="Indian Contract Act",
                target_section="10",
            )
        ]

        metrics = await evaluator.evaluate_dataset(mock_retriever, queries)
        assert metrics.total_queries == 1
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.mrr == 0.0
