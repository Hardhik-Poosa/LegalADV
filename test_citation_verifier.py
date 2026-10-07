from datetime import date

import pytest

from citation_verifier import (
    Citation,
    CitationVerificationError,
    CitationVerifier,
    InvalidConfigError,
    Provision,
    Status,
    VerifierConfig,
)

TODAY = date(2026, 10, 7)


class FakeRepo:
    def __init__(self, provisions: list[Provision]) -> None:
        self._data = {(p.act.lower(), p.section.lower()): p for p in provisions}

    def get(self, act: str, section: str, jurisdiction: str) -> Provision | None:
        return self._data.get((act.strip().lower(), section.strip().lower()))


@pytest.fixture
def verifier() -> CitationVerifier:
    repo = FakeRepo(
        [
            Provision("Consumer Protection Act, 2019", "35", "IN", date(2020, 7, 20)),
            Provision("Old Act", "1", "IN", date(1950, 1, 1), date(2000, 1, 1)),
        ]
    )
    return CitationVerifier(repo, VerifierConfig())


def test_verified(verifier: CitationVerifier) -> None:
    c = Citation("Consumer Protection Act, 2019", "35")
    report = verifier.verify([c], {c.key}, TODAY)
    assert report.all_verified


def test_not_retrieved(verifier: CitationVerifier) -> None:
    c = Citation("Consumer Protection Act, 2019", "35")
    report = verifier.verify([c], set(), TODAY)
    assert report.results[0].status is Status.NOT_RETRIEVED


def test_not_found(verifier: CitationVerifier) -> None:
    c = Citation("Fake Act", "999")
    report = verifier.verify([c], {c.key}, TODAY)
    assert report.results[0].status is Status.NOT_FOUND


def test_repealed(verifier: CitationVerifier) -> None:
    c = Citation("Old Act", "1")
    report = verifier.verify([c], {c.key}, TODAY)
    assert report.results[0].status is Status.NOT_IN_FORCE


def test_jurisdiction_mismatch(verifier: CitationVerifier) -> None:
    c = Citation("Consumer Protection Act, 2019", "35", jurisdiction="US")
    report = verifier.verify([c], {c.key}, TODAY)
    assert report.results[0].status is Status.JURISDICTION_MISMATCH


def test_enforce_raises(verifier: CitationVerifier) -> None:
    c = Citation("Fake Act", "1")
    report = verifier.verify([c], {c.key}, TODAY)
    with pytest.raises(CitationVerificationError):
        verifier.enforce(report)


def test_empty_report_is_ok(verifier: CitationVerifier) -> None:
    assert verifier.verify([], set(), TODAY).all_verified


def test_invalid_config() -> None:
    with pytest.raises(InvalidConfigError):
        VerifierConfig(max_unverified_ratio=1.5)
