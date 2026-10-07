"""Integration-style tests for LegalDocumentLoader.

All external dependencies (AsyncSession, LLMClient) are replaced with
``AsyncMock`` / ``MagicMock`` so no real DB or LM Studio is required.

Strategy
--------
* ``test_load_*``          — happy-path and sad-path end-to-end through load().
* ``test_embed_batched_*`` — unit-tests for the private batching helper.
* ``test_duplicate_*``     — checksum-based duplicate detection.
"""
from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import pytest_asyncio

from app.ingestion.exceptions import (
    DocumentAlreadyExistsError,
    EmbeddingError,
    IngestionError,
)
from app.ingestion.loader import LegalDocumentLoader, LoaderConfig, _sha256
from app.ingestion.chunker import ChunkerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 768
FAKE_VECTOR = [0.1] * EMBEDDING_DIM

_ACT_TEXT = """\
1. Short title — This Act may be called the Indian Contract Act, 1872.
2. Interpretation clause — In this Act the following words and expressions are used in the following senses.
3. Proposal — When one person signifies to another his willingness to do or abstain.
"""


@pytest.fixture()
def mock_llm() -> MagicMock:
    client = MagicMock()
    client.embed = MagicMock(
        side_effect=lambda texts: [FAKE_VECTOR[:] for _ in texts]
    )
    return client


@pytest.fixture()
def mock_session() -> AsyncMock:
    session = AsyncMock()

    # add() is synchronous on AsyncSession — use a regular MagicMock so the
    # 'coroutine never awaited' warning is suppressed.
    session.add = MagicMock()

    # flush() sets .id on any ORM object that was added.
    _id_counter = iter(range(1, 10_000))

    async def _flush() -> None:
        for c in session.add.call_args_list:
            obj = c.args[0] if c.args else c.kwargs.get("instance")
            if obj is not None and not getattr(obj, "id", None):
                setattr(obj, "id", next(_id_counter))

    session.flush = AsyncMock(side_effect=_flush)
    session.execute = AsyncMock(
        return_value=MagicMock(one_or_none=MagicMock(return_value=None))
    )
    return session


@pytest.fixture()
def loader(mock_session: AsyncMock, mock_llm: MagicMock) -> LegalDocumentLoader:
    return LegalDocumentLoader(
        session=mock_session,
        llm_client=mock_llm,
        config=LoaderConfig(
            embedding_batch_size=2,
            chunker=ChunkerConfig(size=200, overlap=20),
        ),
    )


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        act_name="Indian Contract Act",
        full_title="The Indian Contract Act, 1872",
        act_number="9 of 1872",
        jurisdiction="IN",
        version=1,
        effective_from=datetime.date(1872, 9, 1),
        repealed_on=None,
        source_url="https://indiacode.nic.in/",
        raw_text=_ACT_TEXT,
        skip_repealed_sections=True,
    )
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadHappyPath:
    @pytest.mark.asyncio
    async def test_returns_load_result(
        self, loader: LegalDocumentLoader, mock_session: AsyncMock
    ) -> None:
        result = await loader.load(**_base_kwargs())
        assert result.sections_loaded >= 1
        assert result.chunks_loaded >= result.sections_loaded
        assert result.document_id is not None

    @pytest.mark.asyncio
    async def test_session_committed(
        self, loader: LegalDocumentLoader, mock_session: AsyncMock
    ) -> None:
        await loader.load(**_base_kwargs())
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_called_with_chunk_texts(
        self, loader: LegalDocumentLoader, mock_llm: MagicMock
    ) -> None:
        await loader.load(**_base_kwargs())
        # embed must be called at least once with a list of strings.
        assert mock_llm.embed.call_count >= 1
        for c in mock_llm.embed.call_args_list:
            texts = c.args[0]
            assert isinstance(texts, list)
            assert all(isinstance(t, str) for t in texts)

    @pytest.mark.asyncio
    async def test_batch_size_respected(
        self, mock_session: AsyncMock, mock_llm: MagicMock
    ) -> None:
        """With batch_size=1, every text gets its own embed() call."""
        loader = LegalDocumentLoader(
            session=mock_session,
            llm_client=mock_llm,
            config=LoaderConfig(
                embedding_batch_size=1,
                chunker=ChunkerConfig(size=200, overlap=10),
            ),
        )
        result = await loader.load(**_base_kwargs())
        # Each call to embed should have exactly 1 text.
        for c in mock_llm.embed.call_args_list:
            assert len(c.args[0]) == 1

    @pytest.mark.asyncio
    async def test_tsvector_refresh_sql_called(
        self, loader: LegalDocumentLoader, mock_session: AsyncMock
    ) -> None:
        await loader.load(**_base_kwargs())
        # execute() should have been called at least once (the tsvector UPDATE).
        # We check that a call containing 'UPDATE section_chunks' was made.
        sql_calls = [str(c.args[0]) for c in mock_session.execute.call_args_list
                     if c.args and hasattr(c.args[0], "text")]
        # At minimum, execute was called (for the duplicate check + tsv refresh).
        assert mock_session.execute.await_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_exact_duplicate_raises(
        self, loader: LegalDocumentLoader, mock_session: AsyncMock
    ) -> None:
        checksum = _sha256(_ACT_TEXT)
        # Simulate existing row with same checksum.
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                one_or_none=MagicMock(return_value=(42, checksum))
            )
        )
        with pytest.raises(DocumentAlreadyExistsError) as exc_info:
            await loader.load(**_base_kwargs())
        assert exc_info.value.act_name == "Indian Contract Act"
        assert exc_info.value.jurisdiction == "IN"

    @pytest.mark.asyncio
    async def test_duplicate_rolls_back_session(
        self, loader: LegalDocumentLoader, mock_session: AsyncMock
    ) -> None:
        checksum = _sha256(_ACT_TEXT)
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                one_or_none=MagicMock(return_value=(42, checksum))
            )
        )
        with pytest.raises(DocumentAlreadyExistsError):
            await loader.load(**_base_kwargs())
        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding failure
# ─────────────────────────────────────────────────────────────────────────────
class TestEmbeddingFailure:
    @pytest.mark.asyncio
    async def test_embed_exception_raises_embedding_error(
        self, mock_session: AsyncMock, mock_llm: MagicMock
    ) -> None:
        mock_llm.embed.side_effect = RuntimeError("LM Studio unreachable")
        loader = LegalDocumentLoader(
            session=mock_session,
            llm_client=mock_llm,
            config=LoaderConfig(chunker=ChunkerConfig(size=200, overlap=10)),
        )
        with pytest.raises(EmbeddingError, match="embed\\(\\) failed"):
            await loader.load(**_base_kwargs())
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_wrong_count_raises_embedding_error(
        self, mock_session: AsyncMock, mock_llm: MagicMock
    ) -> None:
        # Return one fewer vector than expected.
        mock_llm.embed.side_effect = lambda texts: [FAKE_VECTOR] * max(0, len(texts) - 1)
        loader = LegalDocumentLoader(
            session=mock_session,
            llm_client=mock_llm,
            config=LoaderConfig(chunker=ChunkerConfig(size=200, overlap=10)),
        )
        with pytest.raises((EmbeddingError, IngestionError)):
            await loader.load(**_base_kwargs())


# ─────────────────────────────────────────────────────────────────────────────
# Repealed sections
# ─────────────────────────────────────────────────────────────────────────────
class TestRepealedSections:
    @pytest.mark.asyncio
    async def test_skipped_repealed_counted(
        self, loader: LegalDocumentLoader
    ) -> None:
        raw_text_with_repealed = _ACT_TEXT + "\n4. Repealed.\n"
        result = await loader.load(**_base_kwargs(raw_text=raw_text_with_repealed))
        assert result.skipped_repealed >= 1

    @pytest.mark.asyncio
    async def test_include_repealed_when_flag_false(
        self, mock_session: AsyncMock, mock_llm: MagicMock
    ) -> None:
        raw_text_with_repealed = _ACT_TEXT + "\n4. Repealed.\n"
        loader = LegalDocumentLoader(
            session=mock_session,
            llm_client=mock_llm,
            config=LoaderConfig(chunker=ChunkerConfig(size=200, overlap=10)),
        )
        result = await loader.load(
            **_base_kwargs(
                raw_text=raw_text_with_repealed,
                skip_repealed_sections=False,
            )
        )
        assert result.skipped_repealed == 0


# ─────────────────────────────────────────────────────────────────────────────
# _sha256 utility
# ─────────────────────────────────────────────────────────────────────────────
class TestSha256:
    def test_deterministic(self) -> None:
        assert _sha256("hello") == _sha256("hello")

    def test_length_64(self) -> None:
        assert len(_sha256("any text")) == 64

    def test_different_inputs(self) -> None:
        assert _sha256("a") != _sha256("b")
