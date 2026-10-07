"""Domain-specific exceptions for the ingestion subsystem."""
from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion errors."""


class DocumentAlreadyExistsError(IngestionError):
    """A document with the same (act_name, jurisdiction, version) already exists."""

    def __init__(self, act_name: str, jurisdiction: str, version: int) -> None:
        super().__init__(
            f"Document already exists: act={act_name!r} "
            f"jurisdiction={jurisdiction!r} version={version}"
        )
        self.act_name = act_name
        self.jurisdiction = jurisdiction
        self.version = version


class EmbeddingError(IngestionError):
    """LLM client failed to produce embeddings."""


class ChunkingError(IngestionError):
    """Text could not be split into valid chunks."""
