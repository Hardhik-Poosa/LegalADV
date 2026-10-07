"""Unit tests for SituationExtractor and AnswerPipeline."""
from __future__ import annotations

import datetime
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.llm_client import ChatMessage, LLMResponseError
from app.rag.citation_verifier import CitationVerificationError
from app.rag.retriever import SearchResult
from app.services.answer_pipeline import AnswerPipeline, LegalAnswerResponse
from app.services.extractor import ExtractedSituation, SituationExtractor


# ── Helpers / Mock Data ───────────────────────────────────────────────────────
def make_search_result(
    chunk_id: int = 100,
    sec_id: int = 10,
    act: str = "Indian Contract Act",
    sec_num: str = "10",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        section_id=sec_id,
        document_id=1,
        act_name=act,
        section_number=sec_num,
        heading="What agreements are contracts",
        content="All agreements are contracts if they are made by free consent...",
        chunk_text="All agreements are contracts if they are made by free consent...",
        jurisdiction="IN",
        effective_from=datetime.date(1872, 9, 1),
        repealed_on=None,
        is_repealed=False,
        score=0.95,
        vector_rank=1,
        bm25_rank=1,
        source_type="hybrid",
    )


# ── SituationExtractor Tests ─────────────────────────────────────────────────
class TestSituationExtractor:
    def test_extract_empty_query(self):
        mock_llm = MagicMock()
        extractor = SituationExtractor(mock_llm)
        res = extractor.extract("   ")
        assert res.summary == "Empty query"
        assert res.jurisdiction == "IN"

    def test_extract_valid_response(self):
        mock_llm = MagicMock()
        llm_output = {
            "summary": "Breach of supply agreement",
            "facts": ["Vendor delayed delivery", "Buyer incurred loss"],
            "legal_issues": ["Is buyer entitled to compensation?"],
            "jurisdiction": "IN",
            "act_names": ["Indian Contract Act"],
            "search_keywords": ["breach", "compensation"],
        }
        mock_llm.chat.return_value = json.dumps(llm_output)

        extractor = SituationExtractor(mock_llm)
        res = extractor.extract("Vendor delayed delivery under contract")

        assert res.summary == "Breach of supply agreement"
        assert len(res.facts) == 2
        assert res.act_names == ["Indian Contract Act"]
        assert res.search_keywords == ["breach", "compensation"]

    def test_extract_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = LLMResponseError("Malformed output")

        extractor = SituationExtractor(mock_llm)
        res = extractor.extract("I bought a defective car yesterday")

        assert res.summary == "I bought a defective car yesterday"
        assert res.jurisdiction == "IN"
        assert len(res.facts) == 1


# ── AnswerPipeline Tests ─────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestAnswerPipeline:
    async def test_run_happy_path(self):
        mock_retriever = AsyncMock()
        mock_llm = MagicMock()

        # 1. Retriever returns 1 matching chunk (Section 10 of Indian Contract Act)
        retrieved_chunk = make_search_result(100, 10, "Indian Contract Act", "10")
        mock_retriever.search.return_value = [retrieved_chunk]

        # 2. LLM mock returns valid JSON legal answer citing Section 10 of Indian Contract Act
        llm_response_json = {
            "summary": "Agreements made with free consent are valid contracts.",
            "legal_analysis": [
                {
                    "issue": "Validity of contract without consent",
                    "reasoning": "Under Section 10, free consent of competent parties is essential.",
                    "citations": [{"act": "Indian Contract Act", "section": "10"}],
                }
            ],
            "risk_awareness": ["Ensure consent is recorded in writing."],
        }

        # First call is for SituationExtractor (JSON schema), second call for Answer synthesis
        extractor_json = {
            "summary": "Contract validity query",
            "facts": ["Agreement entered"],
            "legal_issues": ["Is consent required?"],
            "jurisdiction": "IN",
            "act_names": ["Indian Contract Act"],
            "search_keywords": ["contract", "consent"],
        }

        mock_llm.chat.side_effect = [
            json.dumps(extractor_json),
            json.dumps(llm_response_json),
        ]

        pipeline = AnswerPipeline(retriever=mock_retriever, llm_client=mock_llm)
        resp: LegalAnswerResponse = await pipeline.run("What makes a contract valid?")

        assert resp.all_citations_verified is True
        assert len(resp.legal_analysis) == 1
        assert resp.legal_analysis[0].citations[0].act == "Indian Contract Act"
        assert resp.legal_analysis[0].citations[0].section == "10"
        assert "AI legal guidance" in resp.disclaimer
        assert len(resp.retrieved_sources) == 1

    async def test_run_unverified_citation_raises_verification_error(self):
        mock_retriever = AsyncMock()
        mock_llm = MagicMock()

        # Retriever returns Indian Contract Act Section 10
        retrieved_chunk = make_search_result(100, 10, "Indian Contract Act", "10")
        mock_retriever.search.return_value = [retrieved_chunk]

        # LLM hallucinates citation for Fake Act Section 999
        hallucinated_llm_json = {
            "summary": "Invalid claim",
            "legal_analysis": [
                {
                    "issue": "Invented issue",
                    "reasoning": "Unfounded reasoning.",
                    "citations": [{"act": "Fake Nonexistent Act", "section": "999"}],
                }
            ],
            "risk_awareness": [],
        }

        extractor_json = {
            "summary": "Query",
            "facts": [],
            "legal_issues": [],
            "jurisdiction": "IN",
            "act_names": [],
            "search_keywords": [],
        }

        mock_llm.chat.side_effect = [
            json.dumps(extractor_json),
            json.dumps(hallucinated_llm_json),
        ]

        pipeline = AnswerPipeline(retriever=mock_retriever, llm_client=mock_llm)

        with pytest.raises(CitationVerificationError):
            await pipeline.run("Some legal question")

    async def test_run_no_retrieved_results_handled_gracefully(self):
        mock_retriever = AsyncMock()
        mock_llm = MagicMock()

        mock_retriever.search.return_value = []

        no_sources_json = {
            "summary": "No specific statutory provision found in knowledge base.",
            "legal_analysis": [],
            "risk_awareness": ["Consult local counsel."],
        }

        extractor_json = {
            "summary": "Unusual question",
            "facts": [],
            "legal_issues": [],
            "jurisdiction": "IN",
            "act_names": [],
            "search_keywords": [],
        }

        mock_llm.chat.side_effect = [
            json.dumps(extractor_json),
            json.dumps(no_sources_json),
        ]

        pipeline = AnswerPipeline(retriever=mock_retriever, llm_client=mock_llm)
        resp = await pipeline.run("Query on rare ordinance")

        assert resp.summary == "No specific statutory provision found in knowledge base."
        assert len(resp.retrieved_sources) == 0
        assert resp.all_citations_verified is True
