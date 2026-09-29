"""Pasted-text ingestion and conservative text normalization."""

import re

from questllm.config import MINIMUM_TEXT_CHARACTERS
from questllm.document import Document, SourceType
from questllm.exceptions import EmptyTextError, InsufficientTextError

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
_LINE_BREAK_HYPHENATION = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")


def normalize_text(text: str, *, repair_hyphenation: bool = False) -> str:
    """Normalize pasted or extracted text while retaining paragraph breaks."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = _CONTROL_CHARACTERS.sub("", normalized)

    if repair_hyphenation:
        normalized = _LINE_BREAK_HYPHENATION.sub("", normalized)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in paragraph.split("\n")]
        collapsed_paragraph = " ".join(line for line in lines if line)
        if collapsed_paragraph:
            paragraphs.append(collapsed_paragraph)

    return _EXCESSIVE_NEWLINES.sub("\n\n", "\n\n".join(paragraphs)).strip()


def ingest_text(text: str) -> Document:
    """Validate and normalize pasted text into a QuestLLM document."""

    normalized = normalize_text(text)
    if not normalized:
        raise EmptyTextError("Paste some readable text before processing content.")
    if len(normalized) < MINIMUM_TEXT_CHARACTERS:
        raise InsufficientTextError(
            f"Provide at least {MINIMUM_TEXT_CHARACTERS} characters of readable text."
        )

    return Document(source_type=SourceType.TEXT, text=normalized)
