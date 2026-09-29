"""Simple internal document records shared by ingestion and later quiz features."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class SourceType(StrEnum):
    """Supported document sources."""

    TEXT = "text"
    PDF = "pdf"
    TOPIC = "topic"


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Text extracted from one PDF page, using a one-based page number."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    """Normalized source material available to later processing phases."""

    source_type: SourceType
    text: str
    pages: tuple[DocumentPage, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        """Return the number of characters in the normalized document text."""

        return len(self.text)

    @property
    def page_count(self) -> int:
        """Return the number of source pages, or zero for pasted text."""

        return len(self.pages)
