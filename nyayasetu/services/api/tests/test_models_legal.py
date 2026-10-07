"""Unit tests for app/models/legal.py.

All tests are pure Python — no DB, no async needed.
We test model construction, __repr__, is_in_force() logic, and relationship
declarations via inspect().
"""
from __future__ import annotations

import datetime

import pytest
from sqlalchemy import inspect as sa_inspect

from app.models.legal import (
    EMBEDDING_DIM,
    LegalDocument,
    LegalSection,
    SectionChunk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _doc(**kwargs: object) -> LegalDocument:
    defaults: dict[str, object] = dict(
        act_name="Indian Contract Act",
        jurisdiction="IN",
        version=1,
        effective_from=datetime.date(1872, 9, 1),
    )
    defaults.update(kwargs)
    return LegalDocument(**defaults)  # type: ignore[arg-type]


def _section(doc_id: int = 1, **kwargs: object) -> LegalSection:
    defaults: dict[str, object] = dict(
        document_id=doc_id,
        section_number="10",
        content="All agreements are contracts if…",
        effective_from=datetime.date(1872, 9, 1),
        is_repealed=False,  # explicit — SA2 mapped_column(default=) is INSERT-time only
    )
    defaults.update(kwargs)
    return LegalSection(**defaults)  # type: ignore[arg-type]


def _chunk(section_id: int = 1, **kwargs: object) -> SectionChunk:
    defaults: dict[str, object] = dict(
        section_id=section_id,
        chunk_index=0,
        text="All agreements are contracts if…",
    )
    defaults.update(kwargs)
    return SectionChunk(**defaults)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# LegalDocument
# ─────────────────────────────────────────────────────────────────────────────
class TestLegalDocument:
    def test_repr_contains_act_name(self) -> None:
        doc = _doc()
        assert "Indian Contract Act" in repr(doc)
        assert "IN" in repr(doc)

    def test_in_force_before_effective(self) -> None:
        doc = _doc(effective_from=datetime.date(1872, 9, 1))
        assert not doc.is_in_force(on=datetime.date(1872, 8, 31))

    def test_in_force_on_effective_date(self) -> None:
        doc = _doc(effective_from=datetime.date(1872, 9, 1))
        assert doc.is_in_force(on=datetime.date(1872, 9, 1))

    def test_in_force_no_repeal(self) -> None:
        doc = _doc(effective_from=datetime.date(1872, 9, 1), repealed_on=None)
        assert doc.is_in_force(on=datetime.date(2025, 1, 1))

    def test_not_in_force_after_repeal(self) -> None:
        doc = _doc(
            effective_from=datetime.date(1872, 9, 1),
            repealed_on=datetime.date(2000, 1, 1),
        )
        assert not doc.is_in_force(on=datetime.date(2000, 1, 1))

    def test_in_force_day_before_repeal(self) -> None:
        doc = _doc(
            effective_from=datetime.date(1872, 9, 1),
            repealed_on=datetime.date(2000, 1, 1),
        )
        assert doc.is_in_force(on=datetime.date(1999, 12, 31))

    def test_table_name(self) -> None:
        assert LegalDocument.__tablename__ == "legal_documents"

    def test_relationship_sections_declared(self) -> None:
        mapper = sa_inspect(LegalDocument)
        assert "sections" in {r.key for r in mapper.relationships}


# ─────────────────────────────────────────────────────────────────────────────
# LegalSection
# ─────────────────────────────────────────────────────────────────────────────
class TestLegalSection:
    def test_repr(self) -> None:
        sec = _section()
        assert "10" in repr(sec)
        assert "repealed=False" in repr(sec)

    def test_is_repealed_defaults_false(self) -> None:
        # is_repealed=False is passed explicitly in the helper because
        # SQLAlchemy 2 mapped_column(default=) is an INSERT-time Column
        # default, not a Python __init__ default.
        sec = _section()
        assert sec.is_repealed is False

    def test_in_force_without_section_repeal(self) -> None:
        sec = _section(effective_from=datetime.date(1872, 9, 1), repealed_on=None)
        assert sec.is_in_force(on=datetime.date(2025, 1, 1))

    def test_not_in_force_after_section_repeal(self) -> None:
        sec = _section(
            effective_from=datetime.date(1872, 9, 1),
            repealed_on=datetime.date(1950, 1, 26),
        )
        assert not sec.is_in_force(on=datetime.date(1950, 1, 26))

    def test_relationship_chunks_declared(self) -> None:
        mapper = sa_inspect(LegalSection)
        assert "chunks" in {r.key for r in mapper.relationships}

    def test_table_name(self) -> None:
        assert LegalSection.__tablename__ == "legal_sections"


# ─────────────────────────────────────────────────────────────────────────────
# SectionChunk
# ─────────────────────────────────────────────────────────────────────────────
class TestSectionChunk:
    def test_repr(self) -> None:
        chunk = _chunk(char_start=0, char_end=100)
        assert "0-100" in repr(chunk)

    def test_embedding_column_dim(self) -> None:
        """Embedding column type must match EMBEDDING_DIM."""
        col = SectionChunk.__table__.c["embedding"]
        assert col.type.dim == EMBEDDING_DIM

    def test_hnsw_index_present(self) -> None:
        index_names = {idx.name for idx in SectionChunk.__table__.indexes}
        assert "ix_section_chunks_embedding_hnsw" in index_names

    def test_gin_index_present(self) -> None:
        index_names = {idx.name for idx in SectionChunk.__table__.indexes}
        assert "ix_section_chunks_tsv_gin" in index_names

    def test_table_name(self) -> None:
        assert SectionChunk.__tablename__ == "section_chunks"

    def test_embedding_nullable(self) -> None:
        col = SectionChunk.__table__.c["embedding"]
        assert col.nullable is True  # NULL until embedded

    def test_relationship_section_declared(self) -> None:
        mapper = sa_inspect(SectionChunk)
        assert "section" in {r.key for r in mapper.relationships}
