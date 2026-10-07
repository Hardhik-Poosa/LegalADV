"""Deterministic citation verification for legal answers.

The LLM may only cite provisions that (a) were retrieved for this query,
(b) exist in the legal DB, (c) are in force on the relevant date, and
(d) match an allowed jurisdiction. Everything else is flagged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class CitationError(Exception):
    """Base class for citation-related errors."""


class InvalidConfigError(CitationError):
    """Raised when verifier configuration is invalid."""


class CitationVerificationError(CitationError):
    """Raised when too many citations fail verification."""


@dataclass(frozen=True)
class VerifierConfig:
    allowed_jurisdictions: frozenset[str] = frozenset({"IN"})
    max_unverified_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_unverified_ratio <= 1.0:
            raise InvalidConfigError("max_unverified_ratio must be in [0, 1]")
        if not self.allowed_jurisdictions:
            raise InvalidConfigError("allowed_jurisdictions must not be empty")


@dataclass(frozen=True)
class Citation:
    act: str
    section: str
    jurisdiction: str = "IN"

    @property
    def key(self) -> tuple[str, str]:
        return (self.act.strip().lower(), self.section.strip().lower())


@dataclass(frozen=True)
class Provision:
    act: str
    section: str
    jurisdiction: str
    effective_from: date
    repealed_on: date | None = None

    def in_force(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.repealed_on is None or on < self.repealed_on


class ProvisionRepository(Protocol):
    def get(self, act: str, section: str, jurisdiction: str) -> Provision | None:
        ...


class Status(str, Enum):
    VERIFIED = "verified"
    NOT_RETRIEVED = "not_retrieved"
    NOT_FOUND = "not_found"
    NOT_IN_FORCE = "not_in_force"
    JURISDICTION_MISMATCH = "jurisdiction_mismatch"


@dataclass(frozen=True)
class CitationResult:
    citation: Citation
    status: Status


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[CitationResult, ...]

    @property
    def unverified_ratio(self) -> float:
        if not self.results:
            return 0.0
        bad = sum(1 for r in self.results if r.status is not Status.VERIFIED)
        return bad / len(self.results)

    @property
    def all_verified(self) -> bool:
        return self.unverified_ratio == 0.0


class CitationVerifier:
    def __init__(self, repo: ProvisionRepository, config: VerifierConfig) -> None:
        self._repo = repo
        self._config = config

    def verify(
        self,
        citations: list[Citation],
        retrieved_keys: set[tuple[str, str]],
        on: date,
    ) -> VerificationReport:
        results = tuple(
            CitationResult(c, self._check(c, retrieved_keys, on)) for c in citations
        )
        for r in results:
            if r.status is not Status.VERIFIED:
                logger.warning(
                    "Citation failed: %s s.%s -> %s",
                    r.citation.act,
                    r.citation.section,
                    r.status.value,
                )
        return VerificationReport(results)

    def enforce(self, report: VerificationReport) -> None:
        if report.unverified_ratio > self._config.max_unverified_ratio:
            raise CitationVerificationError(
                f"unverified ratio {report.unverified_ratio:.2f} exceeds "
                f"{self._config.max_unverified_ratio:.2f}"
            )

    def _check(
        self, c: Citation, retrieved: set[tuple[str, str]], on: date
    ) -> Status:
        if c.jurisdiction not in self._config.allowed_jurisdictions:
            return Status.JURISDICTION_MISMATCH
        if c.key not in retrieved:
            return Status.NOT_RETRIEVED
        provision = self._repo.get(c.act, c.section, c.jurisdiction)
        if provision is None:
            return Status.NOT_FOUND
        if not provision.in_force(on):
            return Status.NOT_IN_FORCE
        return Status.VERIFIED
