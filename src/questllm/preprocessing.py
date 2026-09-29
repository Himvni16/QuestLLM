"""Sentence-level preprocessing that preserves source provenance."""

import re
from dataclasses import dataclass

from nltk.tokenize import sent_tokenize

from questllm.config import MINIMUM_SENTENCE_CHARACTERS
from questllm.document import Document
from questllm.ingestion.text import normalize_text
from questllm.nltk_resources import ensure_punkt_resources

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_SENTENCE_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ProcessedSentence:
    """A normalized model-facing sentence with enough provenance for later source references."""

    index: int
    text: str
    page_number: int | None
    paragraph_index: int


def _usable_sentence(text: str, minimum_length: int) -> bool:
    """Reject punctuation-only and trivially short sentence fragments."""

    return len(text) >= minimum_length and any(character.isalnum() for character in text)


def preprocess_document(
    document: Document,
    *,
    minimum_sentence_length: int = MINIMUM_SENTENCE_CHARACTERS,
) -> tuple[ProcessedSentence, ...]:
    """Split a document into readable, ordered sentences without aggressive NLP rewriting."""

    if minimum_sentence_length < 1:
        raise ValueError("minimum_sentence_length must be at least 1.")

    ensure_punkt_resources()
    source_units = (
        tuple((page.page_number, page.text) for page in document.pages)
        if document.pages
        else ((None, document.text),)
    )

    sentences = []
    sentence_index = 0
    paragraph_index = 0
    for page_number, source_text in source_units:
        normalized_source = normalize_text(source_text)
        for paragraph in _PARAGRAPH_BREAK.split(normalized_source):
            if not paragraph:
                continue
            for sentence in sent_tokenize(paragraph, language="english"):
                normalized_sentence = _SENTENCE_WHITESPACE.sub(" ", sentence).strip()
                if _usable_sentence(normalized_sentence, minimum_sentence_length):
                    sentences.append(
                        ProcessedSentence(
                            index=sentence_index,
                            text=normalized_sentence,
                            page_number=page_number,
                            paragraph_index=paragraph_index,
                        )
                    )
                    sentence_index += 1
            paragraph_index += 1

    return tuple(sentences)
