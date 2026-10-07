"""Split plain-text Indian Acts (e.g. from India Code) into sections.

Assumes the table of contents has already been removed. Section numbers must
be non-decreasing, which filters most numbered lists inside section bodies.
Always spot-check output per Act before loading it into the database.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^\s*(\d{1,4}[A-Z]{0,3})\.\s+(\S.*)$")
_HEADING_SPLIT_RE = re.compile(r"\.\s*[—–-]{1,2}\s*")
_REPEALED_RE = re.compile(r"\b(repealed|omitted)\b", re.IGNORECASE)


class ParseError(Exception):
    """Raised when no sections can be extracted."""


@dataclass(frozen=True)
class ParserConfig:
    max_heading_chars: int = 120
    repealed_max_chars: int = 160


@dataclass(frozen=True)
class ParsedSection:
    act: str
    number: str
    heading: str
    text: str
    repealed: bool


class ActSectionParser:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self._config = config or ParserConfig()

    def parse(self, act_name: str, raw_text: str) -> list[ParsedSection]:
        blocks: list[tuple[str, list[str]]] = []
        last_base = 0
        for line in raw_text.splitlines():
            match = _SECTION_RE.match(line)
            if match:
                base = int(re.match(r"\d+", match.group(1)).group())  # type: ignore[union-attr]
                if base >= last_base:
                    blocks.append((match.group(1), [match.group(2).strip()]))
                    last_base = base
                    continue
            if blocks and line.strip():
                blocks[-1][1].append(line.strip())

        if not blocks:
            raise ParseError(f"No sections found for {act_name!r}")

        sections = [self._build(act_name, num, " ".join(parts)) for num, parts in blocks]
        logger.info("Parsed %d sections from %s", len(sections), act_name)
        return sections

    def _build(self, act: str, number: str, text: str) -> ParsedSection:
        parts = _HEADING_SPLIT_RE.split(text, maxsplit=1)
        heading = parts[0] if len(parts) == 2 else text.split(".")[0]
        heading = heading.strip()[: self._config.max_heading_chars]
        repealed = (
            len(text) <= self._config.repealed_max_chars
            and _REPEALED_RE.search(text) is not None
        )
        return ParsedSection(act, number, heading, text, repealed)
