import pytest

from app.ingestion.section_parser import ActSectionParser, ParseError

SAMPLE = """\
73. Compensation for loss or damage caused by breach of contract.—When a contract has been broken,
the party who suffers by such breach is entitled to receive compensation.
74. Compensation for breach of contract where penalty stipulated for.—When a contract has been
broken, if a sum is named in the contract as the amount to be paid.
74A. Repealed by Act 10 of 2000.
75. Party rightfully rescinding contract entitled to compensation.—A person who rightfully
rescinds a contract is entitled to compensation.
"""


def test_parses_sections_in_order() -> None:
    sections = ActSectionParser().parse("Indian Contract Act, 1872", SAMPLE)
    assert [s.number for s in sections] == ["73", "74", "74A", "75"]


def test_heading_and_continuation_lines() -> None:
    s = ActSectionParser().parse("ICA", SAMPLE)[0]
    assert s.heading == "Compensation for loss or damage caused by breach of contract"
    assert "entitled to receive compensation" in s.text


def test_repealed_flag() -> None:
    sections = ActSectionParser().parse("ICA", SAMPLE)
    assert sections[2].repealed is True
    assert sections[0].repealed is False


def test_ignores_out_of_order_numbered_lines() -> None:
    text = "10. Heading.—Body text.\n1. an illustration item\n11. Next.—More."
    sections = ActSectionParser().parse("X", text)
    assert [s.number for s in sections] == ["10", "11"]
    assert "illustration item" in sections[0].text


def test_no_sections_raises() -> None:
    with pytest.raises(ParseError):
        ActSectionParser().parse("X", "just some prose without numbering")
