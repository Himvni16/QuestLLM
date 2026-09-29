"""Unit tests for sentence-aware approximate chunking."""

from questllm.chunking import chunk_sentences, make_tokenizer_estimator
from questllm.preprocessing import ProcessedSentence


def _sentences(
    *texts: str,
    page_numbers: tuple[int | None, ...] | None = None,
) -> tuple[ProcessedSentence, ...]:
    pages = page_numbers or tuple(None for _ in texts)
    return tuple(
        ProcessedSentence(index=index, text=text, page_number=pages[index], paragraph_index=0)
        for index, text in enumerate(texts)
    )


def test_chunk_sentences_preserves_order_and_size_limits() -> None:
    sentences = _sentences(
        "One two.",
        "Three four.",
        "Five six.",
        "Seven eight.",
    )

    chunks = chunk_sentences(sentences, target_size=4, overlap_size=0)

    assert [chunk.text for chunk in chunks] == ["One two. Three four.", "Five six. Seven eight."]
    assert [chunk.estimated_token_count for chunk in chunks] == [4, 4]


def test_chunk_sentences_uses_sentence_aligned_overlap_without_duplicate_chunks() -> None:
    sentences = _sentences(
        "One two.",
        "Three four.",
        "Five six.",
        "Seven eight.",
    )

    chunks = chunk_sentences(sentences, target_size=6, overlap_size=2)

    assert chunks[0].sentences[-1] == chunks[1].sentences[0]
    assert len({chunk.text for chunk in chunks}) == len(chunks)
    assert all(len(chunk.sentences) == len(set(chunk.sentences)) for chunk in chunks)


def test_chunk_sentences_handles_a_very_long_sentence_without_splitting_it() -> None:
    sentences = _sentences("One two three four five six seven eight nine ten.")

    chunks = chunk_sentences(sentences, target_size=4, overlap_size=1)

    assert len(chunks) == 1
    assert chunks[0].text == sentences[0].text
    assert chunks[0].estimated_token_count > 4


def test_chunk_sentences_tracks_page_provenance_and_avoids_empty_chunks() -> None:
    sentences = _sentences(
        "First page sentence.",
        "Second page sentence.",
        page_numbers=(1, 2),
    )

    chunks = chunk_sentences(sentences, target_size=6, overlap_size=0)

    assert chunks[0].page_numbers == (1, 2)
    assert all(chunk.text for chunk in chunks)


def test_short_documents_produce_one_chunk() -> None:
    sentences = _sentences("A short document still has one complete sentence.")

    chunks = chunk_sentences(sentences, target_size=20, overlap_size=0)

    assert len(chunks) == 1
    assert chunks[0].sentences == sentences


def test_tokenizer_estimator_keeps_chunking_injectable() -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is True
            return ["<s>", *text.split(), "</s>"]

    estimator = make_tokenizer_estimator(FakeTokenizer())

    assert estimator("One short sentence.") == 5
