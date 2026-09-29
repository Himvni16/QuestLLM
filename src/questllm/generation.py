"""Grounded, deterministic T5 question-stem generation for ranked candidates."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Protocol

import torch

from questllm.candidates import CandidateAnswer
from questllm.chunking import SizeEstimator, TextChunk, estimate_word_tokens
from questllm.config import (
    MAXIMUM_QUESTION_CHARACTERS,
    MINIMUM_QUESTION_CHARACTERS,
    MODEL_CONTEXT_TOKEN_BUDGET,
    QUESTION_GENERATION_MAX_NEW_TOKENS,
    QUESTION_GENERATION_MODEL_ID,
    QUESTION_GENERATION_NUM_BEAMS,
    QUESTION_PREVIEW_LIMIT,
)
from questllm.exceptions import GenerationError
from questllm.model_loader import LoadedQuestionGenerationModel


@dataclass(frozen=True, slots=True)
class QuestionGenerationSettings:
    """Small deterministic decoding configuration for the preview workflow."""

    num_beams: int = QUESTION_GENERATION_NUM_BEAMS
    max_new_tokens: int = QUESTION_GENERATION_MAX_NEW_TOKENS
    max_context_tokens: int = MODEL_CONTEXT_TOKEN_BUDGET
    preview_limit: int = QUESTION_PREVIEW_LIMIT


@dataclass(frozen=True, slots=True)
class FocusedContext:
    """Candidate-centred source text that remains small enough for T5 input."""

    text: str
    source_sentence: str
    sentence_index: int
    page_number: int | None


@dataclass(frozen=True, slots=True)
class QuestionGenerationResult:
    """A validated question stem and the source information needed to inspect it."""

    question: str
    candidate: CandidateAnswer
    source_context: str
    source_sentence: str
    sentence_index: int
    paragraph_index: int
    page_number: int | None
    model_id: str
    metadata: Mapping[str, int | str] = field(default_factory=dict)


class QuestionGenerator(Protocol):
    """Replaceable generator boundary used by the service and offline tests."""

    tokenizer: object
    model_id: str
    device: str

    def generate(
        self,
        candidate: CandidateAnswer,
        source_context: FocusedContext,
        settings: QuestionGenerationSettings,
    ) -> QuestionGenerationResult:
        """Generate one validated question for a grounded answer candidate."""


def _candidate_match(text: str, candidate_text: str) -> re.Match[str] | None:
    """Find a standalone candidate occurrence without changing source text."""

    return re.search(rf"(?<!\w){re.escape(candidate_text)}(?!\w)", text, flags=re.IGNORECASE)


def build_focused_context(
    candidate: CandidateAnswer,
    chunks: Sequence[TextChunk],
    *,
    max_tokens: int = MODEL_CONTEXT_TOKEN_BUDGET,
    estimator: SizeEstimator = estimate_word_tokens,
) -> FocusedContext:
    """Use the candidate's sentence and nearby source sentences without unsafe truncation."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    if not _candidate_match(candidate.source_sentence, candidate.text):
        raise GenerationError(
            "The candidate answer is not present in its recorded source sentence."
        )

    matching_chunk = next(
        (
            chunk
            for chunk in chunks
            if any(sentence.index == candidate.sentence_index for sentence in chunk.sentences)
        ),
        None,
    )
    if matching_chunk is None:
        raise GenerationError("No source chunk contains the candidate's recorded source sentence.")

    source_position = next(
        index
        for index, sentence in enumerate(matching_chunk.sentences)
        if sentence.index == candidate.sentence_index
    )
    selected = [matching_chunk.sentences[source_position]]
    if estimator(selected[0].text) > max_tokens:
        raise GenerationError(
            "The candidate's source sentence exceeds the safe model context budget."
        )

    left = source_position - 1
    right = source_position + 1
    while left >= 0 or right < len(matching_chunk.sentences):
        added = False
        for position in (left, right):
            if position < 0 or position >= len(matching_chunk.sentences):
                continue
            possible = (
                [matching_chunk.sentences[position], *selected]
                if position == left
                else [*selected, matching_chunk.sentences[position]]
            )
            possible_text = " ".join(sentence.text for sentence in possible)
            if estimator(possible_text) <= max_tokens:
                selected = possible
                added = True
        left -= 1
        right += 1
        if not added and left < 0 and right >= len(matching_chunk.sentences):
            break

    return FocusedContext(
        text=" ".join(sentence.text for sentence in selected),
        source_sentence=candidate.source_sentence,
        sentence_index=candidate.sentence_index,
        page_number=candidate.page_number,
    )


def build_question_prompt(candidate: CandidateAnswer, context: FocusedContext) -> str:
    """Build model-card-compatible input with exactly one targeted answer span highlighted."""

    if "<hl>" in context.text:
        raise GenerationError("The source context contains reserved highlight markers.")
    source_start = context.text.find(context.source_sentence)
    if source_start < 0:
        raise GenerationError("The source sentence is not present in the focused source context.")
    source_end = source_start + len(context.source_sentence)
    match = _candidate_match(context.source_sentence, candidate.text)
    if not match:
        raise GenerationError("The candidate answer is not present in the supplied source context.")

    start = source_start + match.start()
    end = source_start + match.end()
    if start < source_start or end > source_end:
        raise GenerationError("The candidate answer could not be highlighted safely.")
    highlighted_context = (
        f"{context.text[:start]}<hl> {context.text[start:end]} <hl>{context.text[end:]}"
    )
    return f"generate question: {highlighted_context} </s>"


def validate_generated_question(question: str, candidate: CandidateAnswer) -> str:
    """Reject obviously unusable stems before they reach the Streamlit preview."""

    cleaned = " ".join(question.split())
    if not cleaned:
        raise GenerationError("The model returned an empty question.")
    if len(cleaned) < MINIMUM_QUESTION_CHARACTERS or len(cleaned) > MAXIMUM_QUESTION_CHARACTERS:
        raise GenerationError("The generated question is outside the supported length range.")
    if not cleaned.endswith("?"):
        raise GenerationError("The generated output is not a complete question.")
    lowered = cleaned.lower()
    if any(artifact in lowered for artifact in ("<hl>", "</s>", "generate question:")):
        raise GenerationError("The generated output contains a model prompt artifact.")
    if SequenceMatcher(None, lowered, candidate.source_sentence.lower()).ratio() >= 0.85:
        raise GenerationError("The generated output is too similar to its source sentence.")
    if _candidate_match(cleaned, candidate.text):
        raise GenerationError("The generated question directly exposes its expected answer.")
    return cleaned


class T5QuestionGenerator:
    """A local T5 implementation of the replaceable QuestionGenerator interface."""

    def __init__(self, resources: LoadedQuestionGenerationModel) -> None:
        self.tokenizer = resources.tokenizer
        self.model = resources.model
        self.device = resources.device
        self.model_id = resources.model_id

    def generate(
        self,
        candidate: CandidateAnswer,
        source_context: FocusedContext,
        settings: QuestionGenerationSettings,
    ) -> QuestionGenerationResult:
        """Run deterministic local inference and return a validated structured result."""

        prompt = build_question_prompt(candidate, source_context)
        try:
            model_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            model_inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in model_inputs.items()
            }
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    num_beams=settings.num_beams,
                    do_sample=False,
                    max_new_tokens=settings.max_new_tokens,
                    early_stopping=True,
                )
            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise GenerationError(
                    "QuestLLM ran out of memory during question generation."
                ) from error
            raise GenerationError(
                "QuestLLM could not generate a question from this source context."
            ) from error
        except (TypeError, ValueError, OSError) as error:
            raise GenerationError(
                "QuestLLM could not prepare local model input for generation."
            ) from error

        question = validate_generated_question(generated, candidate)
        return QuestionGenerationResult(
            question=question,
            candidate=candidate,
            source_context=source_context.text,
            source_sentence=source_context.source_sentence,
            sentence_index=candidate.sentence_index,
            paragraph_index=candidate.paragraph_index,
            page_number=candidate.page_number,
            model_id=self.model_id or QUESTION_GENERATION_MODEL_ID,
            metadata={"num_beams": settings.num_beams, "max_new_tokens": settings.max_new_tokens},
        )


class QuestionPreviewService:
    """Generate a small, resilient preview without forcing every candidate to succeed."""

    def __init__(
        self,
        generator: QuestionGenerator,
        *,
        estimator: SizeEstimator = estimate_word_tokens,
    ) -> None:
        self.generator = generator
        self.estimator = estimator

    def generate_preview(
        self,
        chunks: Sequence[TextChunk],
        candidates: Sequence[CandidateAnswer],
        *,
        settings: QuestionGenerationSettings = QuestionGenerationSettings(),
    ) -> tuple[QuestionGenerationResult, ...]:
        """Generate ordered previews, skipping individual weak candidates safely."""

        if settings.preview_limit < 1:
            raise ValueError("preview_limit must be at least 1.")
        results = []
        for candidate in candidates:
            if len(results) >= settings.preview_limit:
                break
            try:
                context = build_focused_context(
                    candidate,
                    chunks,
                    max_tokens=settings.max_context_tokens,
                    estimator=self.estimator,
                )
                results.append(self.generator.generate(candidate, context, settings))
            except GenerationError:
                continue
        return tuple(results)
