"""Unit tests for sentence-level preprocessing without downloading NLTK data."""

import re

import pytest

from questllm import preprocessing
from questllm.document import Document, DocumentPage, SourceType
from questllm.exceptions import NltkResourceError


@pytest.fixture
def tokenizer_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preprocessing, "ensure_punkt_resources", lambda: None)
    monkeypatch.setattr(
        preprocessing,
        "sent_tokenize",
        lambda text, language: [piece for piece in re.split(r"(?<=[.!?])\s+", text) if piece],
    )


def test_preprocess_document_splits_sentences_and_preserves_punctuation(
    tokenizer_ready: None,
) -> None:
    document = Document(
        source_type=SourceType.TEXT,
        text="First sentence keeps punctuation. Does the second sentence keep its question mark?",
    )

    sentences = preprocessing.preprocess_document(document)

    assert [sentence.text for sentence in sentences] == [
        "First sentence keeps punctuation.",
        "Does the second sentence keep its question mark?",
    ]
    assert [sentence.index for sentence in sentences] == [0, 1]


def test_preprocess_document_tracks_paragraphs_and_filters_short_fragments(
    tokenizer_ready: None,
) -> None:
    document = Document(
        source_type=SourceType.TEXT,
        text="Ok. ...! This is the first usable sentence.\n\nThis is another usable sentence.",
    )

    sentences = preprocessing.preprocess_document(document)

    assert [sentence.text for sentence in sentences] == [
        "This is the first usable sentence.",
        "This is another usable sentence.",
    ]
    assert [sentence.paragraph_index for sentence in sentences] == [0, 1]


def test_preprocess_document_preserves_pdf_page_provenance(tokenizer_ready: None) -> None:
    document = Document(
        source_type=SourceType.PDF,
        text="First page has a complete sentence.\n\nSecond page has a complete sentence.",
        pages=(
            DocumentPage(page_number=1, text="First page has a complete sentence."),
            DocumentPage(page_number=2, text="Second page has a complete sentence."),
        ),
    )

    sentences = preprocessing.preprocess_document(document)

    assert [sentence.page_number for sentence in sentences] == [1, 2]
    assert [sentence.text for sentence in sentences] == [
        "First page has a complete sentence.",
        "Second page has a complete sentence.",
    ]


def test_preprocess_document_raises_when_nltk_data_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_resource() -> None:
        raise NltkResourceError("tokenizer data missing")

    monkeypatch.setattr(preprocessing, "ensure_punkt_resources", raise_missing_resource)
    document = Document(
        source_type=SourceType.TEXT,
        text="This sentence is long enough to process safely.",
    )

    with pytest.raises(NltkResourceError, match="tokenizer data missing"):
        preprocessing.preprocess_document(document)
