"""Explainable quality validation, scoring, and deduplication for built questions."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from questllm.config import (
    FILL_IN_THE_BLANK_TOKEN,
    MAXIMUM_QUESTION_CHARACTERS,
    MINIMUM_QUESTION_CHARACTERS,
    QUESTION_NEAR_DUPLICATE_THRESHOLD,
)
from questllm.questions import Question, QuestionType

_NORMALIZATION = re.compile(r"[^\w\s]+")


@dataclass(frozen=True, slots=True)
class QuestionValidation:
    """The deterministic decision and human-readable reasons for one question."""

    is_valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoredQuestion:
    """A validated question paired with its explainable selection score."""

    question: Question
    quality_score: float
    validation: QuestionValidation


def normalize_question_text(text: str) -> str:
    """Normalize text for exact duplicate and source-grounding comparisons."""

    return " ".join(_NORMALIZATION.sub(" ", text.casefold()).split())


def _answer_in_source(question: Question) -> bool:
    return normalize_question_text(question.correct_answer) in normalize_question_text(
        question.source_sentence
    )


def _has_answer_leakage(question: Question) -> bool:
    if question.question_type in {QuestionType.TRUE_FALSE, QuestionType.FILL_IN_THE_BLANK}:
        return False
    answer = re.escape(question.correct_answer)
    return bool(re.search(rf"(?<!\w){answer}(?!\w)", question.prompt, flags=re.IGNORECASE))


def _has_question_answer_type_mismatch(question: Question) -> bool:
    """Reject obvious wh-question/answer mismatches without semantic-model grading."""

    prompt = normalize_question_text(question.prompt)
    answer = normalize_question_text(question.correct_answer)
    answer_terms = answer.split()
    action_like = answer.startswith("to ") or any(term.endswith("ing") for term in answer_terms)
    if ("what task" in prompt or "common task" in prompt) and not action_like:
        return True
    if ("purpose" in prompt or "goal" in prompt) and not action_like:
        return True
    if "how many" in prompt and not re.search(r"\d", answer):
        return True
    if prompt.startswith("who ") and not any(
        token[:1].isupper() for token in question.correct_answer.split()
    ):
        return True
    return False


def validate_question(question: Question) -> QuestionValidation:
    """Apply common and type-specific quality rules without depending on Streamlit."""

    errors = []
    prompt = question.prompt.strip()
    if not prompt:
        errors.append("Question prompt is empty.")
    if not question.correct_answer.strip():
        errors.append("Correct answer is empty.")
    if len(prompt) < MINIMUM_QUESTION_CHARACTERS or len(prompt) > MAXIMUM_QUESTION_CHARACTERS:
        errors.append("Question prompt is outside the supported length range.")
    if not question.source_sentence.strip() or question.sentence_index < 0:
        errors.append("Question provenance is incomplete.")
    if any(artifact in prompt.casefold() for artifact in ("<hl>", "</s>", "generate question:")):
        errors.append("Question contains a generation artifact.")
    if question.question_type not in {QuestionType.TRUE_FALSE, QuestionType.FILL_IN_THE_BLANK} and (
        normalize_question_text(prompt) == normalize_question_text(question.source_sentence)
    ):
        errors.append("Question prompt is effectively identical to its source sentence.")
    if question.question_type is not QuestionType.TRUE_FALSE and not _answer_in_source(question):
        errors.append("Correct answer is not grounded in the recorded source sentence.")
    if _has_answer_leakage(question):
        errors.append("Question prompt directly reveals its expected answer.")
    if _has_question_answer_type_mismatch(question):
        errors.append("Question wording does not match the semantic type of its answer.")

    if question.question_type is QuestionType.MULTIPLE_CHOICE:
        normalized_choices = [normalize_question_text(choice.text) for choice in question.choices]
        if len(question.choices) != 4:
            errors.append("Multiple Choice requires exactly four choices.")
        if any(not choice.text.strip() for choice in question.choices):
            errors.append("Multiple Choice contains an empty choice.")
        if len(set(normalized_choices)) != len(normalized_choices):
            errors.append("Multiple Choice contains duplicate choices.")
        if sum(choice.is_correct for choice in question.choices) != 1:
            errors.append("Multiple Choice requires exactly one correct choice.")
        correct_choice_count = sum(
            normalize_question_text(choice.text) == normalize_question_text(question.correct_answer)
            for choice in question.choices
        )
        if correct_choice_count != 1:
            errors.append("Multiple Choice must include the correct answer exactly once.")
        for choice in question.choices:
            if not choice.is_correct and (
                normalize_question_text(choice.text)
                in normalize_question_text(question.correct_answer)
                or normalize_question_text(question.correct_answer)
                in normalize_question_text(choice.text)
            ):
                errors.append("A distractor overlaps the correct answer.")
                break
    elif question.question_type is QuestionType.TRUE_FALSE:
        if not isinstance(question.boolean_answer, bool):
            errors.append("True/False requires a boolean answer.")
        if question.boolean_answer is True and (
            normalize_question_text(prompt)
            != normalize_question_text(question.source_sentence)
        ):
            errors.append("True statement is not grounded in the original source statement.")
        if question.boolean_answer is False and (
            normalize_question_text(prompt) == normalize_question_text(question.source_sentence)
            or question.metadata.get("source") != "controlled replacement"
        ):
            errors.append("False statement is not a valid controlled factual replacement.")
    elif question.question_type is QuestionType.FILL_IN_THE_BLANK:
        if prompt.count(FILL_IN_THE_BLANK_TOKEN) != 1:
            errors.append("Fill-in-the-Blank requires exactly one blank.")
        if not _answer_in_source(question):
            errors.append("Fill-in-the-Blank answer is absent from the original source.")
        if len(prompt.replace(FILL_IN_THE_BLANK_TOKEN, "").split()) < 4:
            errors.append("Fill-in-the-Blank removes too much source context.")
    elif question.question_type is QuestionType.SHORT_ANSWER:
        if not prompt.endswith("?"):
            errors.append("Short Answer requires a complete question stem.")
        if not question.accepted_answer_variants:
            errors.append("Short Answer requires accepted answer variants.")
        if any(
            normalize_question_text(variant) != normalize_question_text(question.correct_answer)
            for variant in question.accepted_answer_variants
        ):
            errors.append("Short Answer contains an unsupported accepted-answer variant.")

    return QuestionValidation(is_valid=not errors, errors=tuple(errors))


def score_question(
    question: Question,
    validation: QuestionValidation | None = None,
) -> ScoredQuestion:
    """Score valid questions with transparent grounding, provenance, and source-quality signals."""

    validation = validation or validate_question(question)
    if not validation.is_valid:
        return ScoredQuestion(question=question, quality_score=0.0, validation=validation)

    score = 0.35  # Grounding and successful type validation.
    score += 0.15  # Non-empty source sentence and sentence provenance.
    score += 0.15 if question.page_number is not None else 0.1
    score += 0.15 if MINIMUM_QUESTION_CHARACTERS <= len(question.prompt) <= 160 else 0.08
    score += 0.1 if len(question.correct_answer.split()) <= 5 else 0.04
    importance = question.metadata.get("candidate_importance", 0.0)
    if isinstance(importance, (int, float)):
        score += min(max(float(importance), 0.0), 1.0) * 0.1
    return ScoredQuestion(
        question=question,
        quality_score=round(min(score, 1.0), 6),
        validation=validation,
    )


def _near_duplicate_pairs(
    questions: Sequence[ScoredQuestion],
    threshold: float,
) -> set[frozenset[str]]:
    if len(questions) < 2:
        return set()
    prompts = [question.question.prompt for question in questions]
    try:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(prompts)
    except ValueError:
        return set()
    similarity = matrix * matrix.T
    duplicates = set()
    for first in range(len(questions)):
        for second in range(first + 1, len(questions)):
            if float(similarity[first, second]) >= threshold:
                duplicates.add(
                    frozenset((questions[first].question.id, questions[second].question.id))
                )
    return duplicates


def is_duplicate_question(
    candidate: ScoredQuestion,
    accepted: Sequence[ScoredQuestion],
    *,
    near_duplicate_threshold: float = QUESTION_NEAR_DUPLICATE_THRESHOLD,
) -> bool:
    """Check exact prompts, same facts, and TF-IDF near-duplicate wording."""

    for existing in accepted:
        if normalize_question_text(candidate.question.prompt) == normalize_question_text(
            existing.question.prompt
        ):
            return True
        if (
            normalize_question_text(candidate.question.correct_answer)
            == normalize_question_text(existing.question.correct_answer)
            and normalize_question_text(candidate.question.source_sentence)
            == normalize_question_text(existing.question.source_sentence)
        ):
            return True
    pair_duplicates = _near_duplicate_pairs((*accepted, candidate), near_duplicate_threshold)
    return any(candidate.question.id in pair for pair in pair_duplicates)


def deduplicate_questions(
    questions: Sequence[ScoredQuestion],
    *,
    near_duplicate_threshold: float = QUESTION_NEAR_DUPLICATE_THRESHOLD,
) -> tuple[ScoredQuestion, ...]:
    """Keep the stronger deterministically ranked question for every duplicate family."""

    accepted = []
    for scored in sorted(
        questions,
        key=lambda item: (
            -item.quality_score,
            item.question.question_type.value,
            item.question.id,
        ),
    ):
        if not scored.validation.is_valid:
            continue
        if not is_duplicate_question(
            scored,
            accepted,
            near_duplicate_threshold=near_duplicate_threshold,
        ):
            accepted.append(scored)
    return tuple(accepted)
