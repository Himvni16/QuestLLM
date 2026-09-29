"""Unit tests for pasted-text ingestion."""

import pytest

from questllm.document import SourceType
from questllm.exceptions import EmptyTextError, InsufficientTextError
from questllm.ingestion.text import ingest_text


def test_ingest_text_returns_a_normalized_document() -> None:
    document = ingest_text(
        "This is a useful source passage with enough words to create a later quiz."
    )

    assert document.source_type is SourceType.TEXT
    assert document.pages == ()
    assert document.text.startswith("This is a useful")


def test_ingest_text_normalizes_whitespace_and_preserves_paragraphs() -> None:
    document = ingest_text(
        "First paragraph has   extra spacing\r\nand a wrapped line.\r\n\r\n\r\n"
        "Second paragraph remains distinct and is long enough for ingestion."
    )

    assert document.text == (
        "First paragraph has extra spacing and a wrapped line.\n\n"
        "Second paragraph remains distinct and is long enough for ingestion."
    )


@pytest.mark.parametrize("text", ["", " \n\t\r ", "\x00\x01\x02"])
def test_ingest_text_rejects_empty_content(text: str) -> None:
    with pytest.raises(EmptyTextError):
        ingest_text(text)


def test_ingest_text_rejects_insufficient_content() -> None:
    with pytest.raises(InsufficientTextError):
        ingest_text("Too short to create a useful quiz.")


def test_ingest_text_removes_control_characters() -> None:
    document = ingest_text(
        "This\x00 passage\x07 contains control characters but enough readable content "
        "for processing."
    )

    assert "\x00" not in document.text
    assert "\x07" not in document.text
    assert "This passage contains" in document.text
