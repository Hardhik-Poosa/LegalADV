"""Fact and legal issue extractor using LLMClient structured output."""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.llm.llm_client import ChatMessage, LLMClient, LLMResponseError

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSituation:
    """Structured extraction of facts, legal issues, and metadata from a user query."""

    summary: str
    facts: list[str] = field(default_factory=list)
    legal_issues: list[str] = field(default_factory=list)
    jurisdiction: str = "IN"
    as_of_date: datetime.date = field(default_factory=datetime.date.today)
    act_names: list[str] = field(default_factory=list)
    search_keywords: list[str] = field(default_factory=list)


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Short summary of user's situation"},
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Extracted factual statements",
        },
        "legal_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Core legal questions or disputes",
        },
        "jurisdiction": {
            "type": "string",
            "description": "ISO country/state code, e.g. IN or IN-MH",
        },
        "act_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Potential acts/statutes implied",
        },
        "search_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key legal terms for retrieval",
        },
    },
    "required": ["summary", "facts", "legal_issues", "jurisdiction", "search_keywords"],
}


EXTRACTION_SYSTEM_PROMPT = """\
You are an expert Indian legal assistant. Analyze the user's factual query and extract:
1. Summary: Concise statement of the situation.
2. Facts: Key facts stated by the user.
3. Legal Issues: Specific legal questions raised by these facts.
4. Jurisdiction: Code such as 'IN' (Union of India) or state codes like 'IN-MH', 'IN-DL'. Default to 'IN'.
5. Act Names: Mentioned or relevant Indian Acts (e.g. 'Indian Contract Act', 'Limitation Act').
6. Search Keywords: Terms to help search legal statutes.

Output valid JSON strictly adhering to the schema.
"""


class SituationExtractor:
    """Extracts structured legal facts and issues from unstructured user text."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def extract(self, query: str, default_jurisdiction: str = "IN") -> ExtractedSituation:
        """Extract structured situation parameters from a user prompt."""
        if not query.strip():
            return ExtractedSituation(
                summary="Empty query",
                jurisdiction=default_jurisdiction,
            )

        messages = [
            ChatMessage(role="system", content=EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"User Query:\n{query}"),
        ]

        try:
            raw_response = self.llm_client.chat(
                messages, json_schema=EXTRACTION_SCHEMA
            )
            data = json.loads(raw_response)
        except (LLMResponseError, json.JSONDecodeError) as exc:
            logger.warning("Structured extraction failed (%s), using fallback", exc)
            return ExtractedSituation(
                summary=query[:100],
                facts=[query],
                legal_issues=["General legal guidance request"],
                jurisdiction=default_jurisdiction,
                search_keywords=[k for k in query.split() if len(k) > 3][:5],
            )

        jurisdiction = data.get("jurisdiction") or default_jurisdiction

        return ExtractedSituation(
            summary=data.get("summary", query[:100]),
            facts=data.get("facts", []),
            legal_issues=data.get("legal_issues", []),
            jurisdiction=jurisdiction,
            as_of_date=datetime.date.today(),
            act_names=data.get("act_names", []),
            search_keywords=data.get("search_keywords", []),
        )
