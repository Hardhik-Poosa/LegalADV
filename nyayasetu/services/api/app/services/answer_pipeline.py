"""Answer Generation Pipeline for NyayaSetu combining extraction, hybrid retrieval, LLM synthesis, and citation verification."""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.llm_client import ChatMessage, LLMClient, LLMResponseError
from app.models.legal import LegalDocument, LegalSection
from app.rag.citation_verifier import (
    Citation,
    CitationVerificationError,
    CitationVerifier,
    Provision,
    ProvisionRepository,
    VerifierConfig,
)
from app.rag.retriever import HybridRetriever, RetrievalFilter, SearchResult
from app.services.extractor import ExtractedSituation, SituationExtractor

logger = logging.getLogger(__name__)

STANDARD_DISCLAIMER: str = (
    "NyayaSetu provides AI legal guidance and risk awareness for educational "
    "and decision-support purposes. It is not a lawyer replacement and does not "
    "constitute formal legal advice."
)


@dataclass
class LegalAnalysisPoint:
    """A single legal issue analysis with supporting statute citations."""

    issue: str
    reasoning: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class LegalAnswerResponse:
    """Complete response returned by the NyayaSetu Answer Pipeline."""

    disclaimer: str
    summary: str
    facts_considered: list[str]
    legal_analysis: list[LegalAnalysisPoint]
    risk_awareness: list[str]
    retrieved_sources: list[dict[str, Any]]
    verified_citations: list[dict[str, Any]]
    all_citations_verified: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class DBProvisionRepository:
    """ProvisionRepository implementation using SQLAlchemy database queries and retrieved provisions cache."""

    def __init__(
        self,
        session: AsyncSession | None = None,
        cached_provisions: list[Provision] | None = None,
    ) -> None:
        self.session = session
        self._cache: dict[tuple[str, str, str], Provision] = {}
        if cached_provisions:
            for p in cached_provisions:
                key = (p.act.strip().lower(), p.section.strip().lower(), p.jurisdiction.strip().upper())
                self._cache[key] = p

    def get(self, act: str, section: str, jurisdiction: str) -> Provision | None:
        key = (act.strip().lower(), section.strip().lower(), jurisdiction.strip().upper())
        if key in self._cache:
            return self._cache[key]

        # If sync fallback is called without session, return None
        return None

    async def get_async(self, act: str, section: str, jurisdiction: str) -> Provision | None:
        """Async lookup against PostgreSQL database if not in cache."""
        key = (act.strip().lower(), section.strip().lower(), jurisdiction.strip().upper())
        if key in self._cache:
            return self._cache[key]

        if not self.session:
            return None

        stmt = (
            select(LegalSection, LegalDocument)
            .join(LegalDocument, LegalSection.document_id == LegalDocument.id)
            .where(
                func.lower(LegalDocument.act_name) == act.strip().lower(),
                func.lower(LegalSection.section_number) == section.strip().lower(),
                func.upper(LegalDocument.jurisdiction) == jurisdiction.strip().upper(),
            )
        )
        result = await self.session.execute(stmt)
        row = result.first()
        if not row:
            return None

        sec, doc = row
        prov = Provision(
            act=doc.act_name,
            section=sec.section_number,
            jurisdiction=doc.jurisdiction,
            effective_from=sec.effective_from,
            repealed_on=sec.repealed_on,
        )
        self._cache[key] = prov
        return prov


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Clear high-level answer summary"},
        "legal_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "act": {"type": "string"},
                                "section": {"type": "string"},
                            },
                            "required": ["act", "section"],
                        },
                    },
                },
                "required": ["issue", "reasoning", "citations"],
            },
        },
        "risk_awareness": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Limitation periods, statutory risks, or procedural notice requirements",
        },
    },
    "required": ["summary", "legal_analysis", "risk_awareness"],
}


SYSTEM_PROMPT = """\
You are NyayaSetu, an expert AI Legal Guidance Assistant for Indian Civil Law.
Your goal is to provide accurate, grounded risk-awareness and legal guidance based ONLY on the provided legal provisions.

STRICT GROUNDING RULES:
1. Ground EVERY legal statement and claim directly in the provided retrieved legal sections.
2. NEVER invent citations, sections, or statutes that are not explicitly present in the retrieved context.
3. Every citation must include exact 'act' and 'section' fields matching the context.
4. If the retrieved sections do not contain enough information to fully resolve an issue, explicitly note the gap.
5. Provide practical risk awareness (e.g. limitation periods, mandatory notice provisions, evidentiary requirements).
"""


class AnswerPipeline:
    """Orchestrates extraction -> hybrid retrieval -> LLM answer generation -> citation verification."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        verifier: CitationVerifier | None = None,
        extractor: SituationExtractor | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.extractor = extractor or SituationExtractor(llm_client)
        
        # Default verifier allowing 0 unverified citations
        default_config = VerifierConfig(
            allowed_jurisdictions=frozenset({"IN", "IN-MH", "IN-DL", "IN-KA"}),
            max_unverified_ratio=0.0,
        )
        self.verifier = verifier or CitationVerifier(
            repo=DBProvisionRepository(), config=default_config
        )

    def _build_context_prompt(
        self, situation: ExtractedSituation, search_results: list[SearchResult]
    ) -> str:
        """Format retrieved search results into a clean, unambiguous context block."""
        context_lines = [
            f"=== USER SITUATION ===",
            f"Summary: {situation.summary}",
            f"Facts: {', '.join(situation.facts)}",
            f"Legal Issues: {', '.join(situation.legal_issues)}",
            f"Jurisdiction: {situation.jurisdiction}",
            "",
            f"=== RETRIEVED LEGAL STATUTES & SECTIONS ({len(search_results)} items) ===",
        ]

        if not search_results:
            context_lines.append("No matching statutory sections were found in the knowledge base.")
        else:
            for idx, res in enumerate(search_results, start=1):
                heading_str = f" - {res.heading}" if res.heading else ""
                context_lines.append(
                    f"[{idx}] ACT: {res.act_name} | SECTION: {res.section_number}{heading_str}\n"
                    f"    Jurisdiction: {res.jurisdiction} | In force from: {res.effective_from}\n"
                    f"    Text: {res.chunk_text}\n"
                )

        return "\n".join(context_lines)

    async def run(
        self,
        user_query: str,
        jurisdiction: str = "IN",
        as_of_date: datetime.date | None = None,
        top_k: int = 10,
    ) -> LegalAnswerResponse:
        """Run full pipeline: extract -> retrieve -> synthesize -> verify -> enforce."""
        target_date = as_of_date or datetime.date.today()

        # 1. Extract situation parameters
        situation = self.extractor.extract(user_query, default_jurisdiction=jurisdiction)

        # 2. Build retrieval filter & execute search
        retrieval_filter = RetrievalFilter(
            jurisdiction=situation.jurisdiction,
            as_of_date=target_date,
            act_names=situation.act_names or None,
        )
        retrieved_results = await self.retriever.search(
            query=user_query,
            filter_=retrieval_filter,
            top_k=top_k,
        )

        # 3. Assemble prompt context
        context_prompt = self._build_context_prompt(situation, retrieved_results)

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=context_prompt),
        ]

        # 4. Generate structured response from LLM
        try:
            raw_json = self.llm_client.chat(messages, json_schema=ANSWER_SCHEMA)
            data = json.loads(raw_json)
        except (LLMResponseError, json.JSONDecodeError) as exc:
            logger.error("Answer synthesis failed: %s", exc)
            data = {
                "summary": "Unable to synthesize legal response due to model error.",
                "legal_analysis": [],
                "risk_awareness": ["System encountered an error during response generation."],
            }

        # 5. Extract citations and prepare verification context
        extracted_citations: list[Citation] = []
        analysis_points: list[LegalAnalysisPoint] = []

        for item in data.get("legal_analysis", []):
            item_citations: list[Citation] = []
            for c_dict in item.get("citations", []):
                act_str = c_dict.get("act", "")
                sec_str = c_dict.get("section", "")
                if act_str and sec_str:
                    c_obj = Citation(
                        act=act_str,
                        section=sec_str,
                        jurisdiction=situation.jurisdiction,
                    )
                    item_citations.append(c_obj)
                    extracted_citations.append(c_obj)

            analysis_points.append(
                LegalAnalysisPoint(
                    issue=item.get("issue", ""),
                    reasoning=item.get("reasoning", ""),
                    citations=item_citations,
                )
            )

        # Build set of retrieved provision keys: (act_name.lower(), section_number.lower())
        retrieved_keys = {
            (r.act_name.strip().lower(), r.section_number.strip().lower())
            for r in retrieved_results
        }

        # Build provision repository for verification
        provisions_list = [
            Provision(
                act=r.act_name,
                section=r.section_number,
                jurisdiction=r.jurisdiction,
                effective_from=r.effective_from,
                repealed_on=r.repealed_on,
            )
            for r in retrieved_results
        ]
        repo = DBProvisionRepository(cached_provisions=provisions_list)
        verifier = CitationVerifier(repo=repo, config=self.verifier._config)

        # 6. Verify citations & enforce
        report = verifier.verify(
            citations=extracted_citations,
            retrieved_keys=retrieved_keys,
            on=target_date,
        )
        verifier.enforce(report)

        # Build output structure
        verified_citations_meta = [
            {
                "act": r.citation.act,
                "section": r.citation.section,
                "jurisdiction": r.citation.jurisdiction,
                "status": r.status.value,
            }
            for r in report.results
        ]

        retrieved_sources_meta = [
            {
                "act_name": r.act_name,
                "section_number": r.section_number,
                "heading": r.heading,
                "jurisdiction": r.jurisdiction,
                "score": r.score,
            }
            for r in retrieved_results
        ]

        return LegalAnswerResponse(
            disclaimer=STANDARD_DISCLAIMER,
            summary=data.get("summary", ""),
            facts_considered=situation.facts,
            legal_analysis=analysis_points,
            risk_awareness=data.get("risk_awareness", []),
            retrieved_sources=retrieved_sources_meta,
            verified_citations=verified_citations_meta,
            all_citations_verified=report.all_verified,
        )
