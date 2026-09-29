"""Offline tests for grounded T5 prompt building and preview orchestration."""

from dataclasses import dataclass

import pytest

from questllm.candidates import CandidateAnswer, CandidateType
from questllm.chunking import TextChunk
from questllm.exceptions import GenerationError
from questllm.generation import (
    FocusedContext,
    QuestionGenerationResult,
    QuestionGenerationSettings,
    QuestionPreviewService,
    T5QuestionGenerator,
    build_focused_context,
    build_question_prompt,
    validate_generated_question,
)
from questllm.model_loader import LoadedQuestionGenerationModel
from questllm.preprocessing import ProcessedSentence


def _sentence(index: int, text: str, page_number: int | None = 1) -> ProcessedSentence:
    return ProcessedSentence(index=index, text=text, paragraph_index=index, page_number=page_number)


def _candidate(
    text: str = "photosynthesis",
    source_sentence: str = "Photosynthesis captures light energy in plant cells.",
    index: int = 1,
) -> CandidateAnswer:
    return CandidateAnswer(
        text=text,
        normalized_text=text.lower(),
        source_sentence=source_sentence,
        sentence_index=index,
        paragraph_index=index,
        page_number=2,
        importance_score=0.8,
        candidate_type=CandidateType.DEFINITION_TERM,
    )


def _chunk(*sentences: ProcessedSentence) -> TextChunk:
    return TextChunk(
        index=0,
        text=" ".join(sentence.text for sentence in sentences),
        sentences=sentences,
        page_numbers=tuple(sentence.page_number for sentence in sentences if sentence.page_number),
        estimated_token_count=20,
    )


def test_builds_focused_context_and_highlights_only_one_target_occurrence() -> None:
    source = "DNA helps DNA repair damage in cells."
    candidate = _candidate("DNA", source, 1)
    chunk = _chunk(
        _sentence(0, "Cells contain genetic material."),
        _sentence(1, source, page_number=2),
        _sentence(2, "Repair protects the organism."),
    )

    context = build_focused_context(candidate, (chunk,), max_tokens=30)
    prompt = build_question_prompt(candidate, context)

    assert context.sentence_index == 1
    assert context.page_number == 2
    assert prompt.count("<hl>") == 2
    assert "<hl> DNA <hl> helps DNA repair" in prompt


def test_rejects_missing_candidate_and_oversized_source_sentence() -> None:
    candidate = _candidate("chlorophyll")
    chunk = _chunk(_sentence(1, candidate.source_sentence, page_number=2))

    with pytest.raises(GenerationError, match="not present"):
        build_focused_context(candidate, (chunk,))

    long_source = "photosynthesis " + "important " * 20
    long_candidate = _candidate("photosynthesis", long_source)
    long_chunk = _chunk(_sentence(1, long_source, page_number=2))
    with pytest.raises(GenerationError, match="exceeds"):
        build_focused_context(long_candidate, (long_chunk,), max_tokens=3)


def test_question_validation_rejects_artifacts_answers_and_source_repetition() -> None:
    candidate = _candidate()

    assert validate_generated_question(
        "Which process captures light energy?", candidate
    ).endswith("?")
    with pytest.raises(GenerationError, match="directly exposes"):
        validate_generated_question("What is photosynthesis?", candidate)
    with pytest.raises(GenerationError, match="prompt artifact"):
        validate_generated_question("<hl> What process captures light?", candidate)
    with pytest.raises(GenerationError, match="too similar"):
        validate_generated_question(candidate.source_sentence[:-1] + "?", candidate)


class _FakeTensor:
    def to(self, _device: str) -> "_FakeTensor":
        return self


class _FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **_kwargs: object) -> dict[str, _FakeTensor]:
        self.prompts.append(prompt)
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}

    def decode(self, _output: object, **_kwargs: object) -> str:
        return "Which process captures light energy?"

    def encode(self, text: str, **_kwargs: object) -> list[str]:
        return text.split()


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.kwargs = kwargs
        return [[1, 2]]


def test_t5_generator_returns_structured_deterministic_result() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    generator = T5QuestionGenerator(
        LoadedQuestionGenerationModel(
            tokenizer=tokenizer,
            model=model,
            device="cpu",
            model_id="fake/t5",
        )
    )
    candidate = _candidate()
    context = FocusedContext(candidate.source_sentence, candidate.source_sentence, 1, 2)

    result = generator.generate(candidate, context, QuestionGenerationSettings())

    assert result.question == "Which process captures light energy?"
    assert result.candidate == candidate
    assert result.model_id == "fake/t5"
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["num_beams"] == 4
    assert tokenizer.prompts[0].startswith("generate question: ")


@dataclass
class _PreviewFakeGenerator:
    tokenizer: object
    model_id: str = "fake/t5"
    device: str = "cpu"

    def generate(
        self,
        candidate: CandidateAnswer,
        source_context: FocusedContext,
        _settings: QuestionGenerationSettings,
    ) -> QuestionGenerationResult:
        if candidate.text == "bad":
            raise GenerationError("bad candidate")
        return QuestionGenerationResult(
            question=f"Which term relates to {candidate.text[0]}?",
            candidate=candidate,
            source_context=source_context.text,
            source_sentence=source_context.source_sentence,
            sentence_index=candidate.sentence_index,
            paragraph_index=candidate.paragraph_index,
            page_number=candidate.page_number,
            model_id=self.model_id,
        )


def test_preview_service_skips_failed_candidates_and_respects_order_and_limit() -> None:
    first = _candidate("bad", "Bad appears in the source material.", 0)
    second = _candidate("alpha", "Alpha appears in the source material.", 1)
    third = _candidate("beta", "Beta appears in the source material.", 2)
    chunk = _chunk(
        _sentence(0, first.source_sentence),
        _sentence(1, second.source_sentence),
        _sentence(2, third.source_sentence),
    )

    results = QuestionPreviewService(_PreviewFakeGenerator(_FakeTokenizer())).generate_preview(
        (chunk,),
        (first, second, third),
        settings=QuestionGenerationSettings(preview_limit=2),
    )

    assert [result.candidate.text for result in results] == ["alpha", "beta"]
