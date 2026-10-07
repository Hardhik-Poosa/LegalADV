"""Character-based sliding-window text chunker.

Designed to be:
- Pure (no I/O, no DB, no LLM calls).
- Fully deterministic — same input always yields the same chunks.
- Easily unit-tested in isolation.

The chunker respects sentence and word boundaries where possible; it falls
back to a hard character split only if a single sentence exceeds ``size``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.ingestion.exceptions import ChunkingError

logger = logging.getLogger(__name__)

# Sentence boundary: period/! /? followed by whitespace or end-of-string.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    char_start: int
    char_end: int
    index: int  # 0-based position within the section


@dataclass(frozen=True)
class ChunkerConfig:
    size: int = 512        # target characters per chunk
    overlap: int = 64      # overlap with previous chunk

    def __post_init__(self) -> None:
        if self.overlap >= self.size:
            raise ChunkingError(
                f"overlap ({self.overlap}) must be less than size ({self.size})"
            )
        if self.size < 1:
            raise ChunkingError("size must be >= 1")


def chunk_text(text: str, config: ChunkerConfig) -> list[TextChunk]:
    """Split *text* into overlapping chunks respecting sentence boundaries.

    Returns an empty list when *text* is blank.
    Any sentence (or text segment with no sentence boundary) that exceeds
    ``config.size`` is hard-split at ``size - overlap`` step so the caller
    always receives chunks no larger than the configured size.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences for boundary-aware chunking.
    sentences = _SENT_BOUNDARY.split(text)

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_len: int = 0
    # We track absolute char offsets by walking the original text.
    global_pos: int = 0
    chunk_start: int = 0

    def _flush(parts: list[str], start: int, end: int) -> TextChunk:
        body = " ".join(parts).strip()
        return TextChunk(
            text=body,
            char_start=start,
            char_end=end,
            index=len(chunks),
        )

    for sentence in sentences:
        sent_len = len(sentence)

        # Hard-split any sentence that exceeds the target chunk size.
        if sent_len > config.size:
            logger.debug(
                "Sentence of %d chars exceeds chunk size %d — hard-splitting.",
                sent_len,
                config.size,
            )
            # Flush current accumulation first.
            if current_parts:
                chunks.append(_flush(current_parts, chunk_start, global_pos))
                current_parts = []
                current_len = 0
            step = max(1, config.size - config.overlap)
            for i in range(0, sent_len, step):
                fragment = sentence[i : i + config.size]
                abs_start = global_pos + i
                abs_end = abs_start + len(fragment)
                chunks.append(
                    TextChunk(
                        text=fragment,
                        char_start=abs_start,
                        char_end=abs_end,
                        index=len(chunks),
                    )
                )
            global_pos += sent_len + 1  # +1 for the whitespace consumed by split
            chunk_start = global_pos
            continue

        if current_len + sent_len > config.size and current_parts:
            # Emit current accumulation.
            chunk_end = global_pos
            chunks.append(_flush(current_parts, chunk_start, chunk_end))

            # Carry over tail for overlap.
            overlap_text = " ".join(current_parts)[-config.overlap :]
            current_parts = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
            chunk_start = chunk_end - config.overlap if overlap_text else global_pos

        current_parts.append(sentence)
        current_len += sent_len + 1  # +1 for space separator
        global_pos += sent_len + 1

    # Flush remaining content.
    if current_parts:
        chunks.append(_flush(current_parts, chunk_start, len(text)))

    logger.debug("Chunked %d chars into %d chunks.", len(text), len(chunks))
    return chunks
