"""Sentence-aware chunking with a replaceable size-estimation function."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from questllm.config import CHUNK_OVERLAP_SIZE, TARGET_CHUNK_SIZE
from questllm.preprocessing import ProcessedSentence

SizeEstimator = Callable[[str], int]
_WORD_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """An ordered collection of complete sentences for a later model input."""

    index: int
    text: str
    sentences: tuple[ProcessedSentence, ...]
    page_numbers: tuple[int, ...]
    estimated_token_count: int


def estimate_word_tokens(text: str) -> int:
    """Estimate model input size by counting whitespace-delimited terms for this phase."""

    return len(_WORD_PATTERN.findall(text))


def make_tokenizer_estimator(tokenizer: object) -> SizeEstimator:
    """Return a tokenizer-based size estimator without coupling chunking to Transformers."""

    def estimate_tokens(text: str) -> int:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=True)  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "The tokenizer must provide encode(text, add_special_tokens=True)."
            ) from error
        return len(encoded)

    return estimate_tokens


def _sentence_size(sentence: ProcessedSentence, estimator: SizeEstimator) -> int:
    return estimator(sentence.text)


def _tail_for_overlap(
    sentences: list[ProcessedSentence],
    overlap_size: int,
    estimator: SizeEstimator,
) -> list[ProcessedSentence]:
    """Return the largest sentence-aligned tail that fits the configured overlap budget."""

    overlap = []
    used_size = 0
    for sentence in reversed(sentences):
        sentence_size = _sentence_size(sentence, estimator)
        if used_size + sentence_size > overlap_size:
            break
        overlap.append(sentence)
        used_size += sentence_size
    overlap.reverse()
    return overlap


def _build_chunk(
    index: int,
    sentences: list[ProcessedSentence],
    estimator: SizeEstimator,
) -> TextChunk:
    """Create one non-empty text chunk from complete sentence records."""

    sentence_records = tuple(sentences)
    text = " ".join(sentence.text for sentence in sentence_records)
    page_numbers = tuple(
        dict.fromkeys(
            sentence.page_number
            for sentence in sentence_records
            if sentence.page_number is not None
        )
    )
    return TextChunk(
        index=index,
        text=text,
        sentences=sentence_records,
        page_numbers=page_numbers,
        estimated_token_count=estimator(text),
    )


def chunk_sentences(
    sentences: tuple[ProcessedSentence, ...],
    *,
    target_size: int = TARGET_CHUNK_SIZE,
    overlap_size: int = CHUNK_OVERLAP_SIZE,
    estimator: SizeEstimator = estimate_word_tokens,
) -> tuple[TextChunk, ...]:
    """Group ordered sentences into bounded chunks without splitting a sentence."""

    if target_size < 1:
        raise ValueError("target_size must be at least 1.")
    if overlap_size < 0 or overlap_size >= target_size:
        raise ValueError("overlap_size must be non-negative and smaller than target_size.")
    if not sentences:
        return ()

    chunks = []
    current_sentences: list[ProcessedSentence] = []
    current_size = 0

    for sentence in sentences:
        sentence_size = _sentence_size(sentence, estimator)
        if current_sentences and current_size + sentence_size > target_size:
            chunks.append(_build_chunk(len(chunks), current_sentences, estimator))
            current_sentences = _tail_for_overlap(current_sentences, overlap_size, estimator)
            current_size = sum(_sentence_size(item, estimator) for item in current_sentences)

            while current_sentences and current_size + sentence_size > target_size:
                removed_sentence = current_sentences.pop(0)
                current_size -= _sentence_size(removed_sentence, estimator)

        current_sentences.append(sentence)
        current_size += sentence_size

    if current_sentences:
        chunk = _build_chunk(len(chunks), current_sentences, estimator)
        if not chunks or chunk.sentences != chunks[-1].sentences:
            chunks.append(chunk)

    return tuple(chunks)
