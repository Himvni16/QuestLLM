"""Conservative, deterministic grading of QuestLLM quizzes without Streamlit dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from questllm.config import (
    MINIMUM_SHORT_ANSWER_OVERLAP_TOKENS,
    SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD,
)
from questllm.questions import Question, QuestionType
from questllm.quiz import Quiz
from questllm.validation import normalize_question_text

AnswerValue = str | bool | None


class AnswerStatus(StrEnum):
    """Review labels that distinguish missing responses from incorrect responses."""

    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    UNANSWERED = "Unanswered"


@dataclass(frozen=True, slots=True)
class QuestionScore:
    """The stored grading decision for one stable question ID."""

    question_id: str
    question_type: QuestionType
    user_answer: AnswerValue
    expected_answer: str
    is_correct: bool
    status: AnswerStatus
    normalized_user_answer: str | None


@dataclass(frozen=True, slots=True)
class TypeScore:
    """A compact per-question-type result summary."""

    total: int
    correct: int
    incorrect: int
    unanswered: int


@dataclass(frozen=True, slots=True)
class QuizScore:
    """The immutable overall score and all review-ready per-question results."""

    quiz_id: str
    total_questions: int
    correct_count: int
    incorrect_count: int
    unanswered_count: int
    percentage: float
    by_type: Mapping[QuestionType, TypeScore]
    question_results: tuple[QuestionScore, ...]


def normalize_answer(answer: str) -> str:
    """Normalize only safe superficial differences for concise factual answers."""

    return normalize_question_text(answer)


def _is_unanswered(answer: AnswerValue) -> bool:
    return answer is None or (isinstance(answer, str) and not answer.strip())


def _matches_variants(answer: str, question: Question) -> bool:
    normalized = normalize_answer(answer)
    variants = {normalize_answer(question.correct_answer)}
    variants.update(normalize_answer(variant) for variant in question.accepted_answer_variants)
    return normalized in variants


def _conservative_token_overlap(answer: str, question: Question) -> bool:
    """Allow close concise factual phrases without turning grading into semantic evaluation."""

    answer_terms = set(normalize_answer(answer).split())
    expected_terms = set(normalize_answer(question.correct_answer).split())
    if (
        len(answer_terms) < MINIMUM_SHORT_ANSWER_OVERLAP_TOKENS
        or len(expected_terms) < MINIMUM_SHORT_ANSWER_OVERLAP_TOKENS
    ):
        return False
    overlap = len(answer_terms & expected_terms)
    return (
        overlap / len(answer_terms) >= SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD
        and overlap / len(expected_terms) >= SHORT_ANSWER_TOKEN_OVERLAP_THRESHOLD
    )


def score_question(question: Question, user_answer: AnswerValue) -> QuestionScore:
    """Score one answer according to its question type and return a review-ready result."""

    if _is_unanswered(user_answer):
        return QuestionScore(
            question_id=question.id,
            question_type=question.question_type,
            user_answer=None,
            expected_answer=question.correct_answer,
            is_correct=False,
            status=AnswerStatus.UNANSWERED,
            normalized_user_answer=None,
        )

    is_correct = False
    normalized_answer: str | None = None
    if question.question_type is QuestionType.MULTIPLE_CHOICE:
        correct_choice = next((choice for choice in question.choices if choice.is_correct), None)
        is_correct = isinstance(user_answer, str) and (
            correct_choice is not None and user_answer == correct_choice.id
        )
        normalized_answer = user_answer if isinstance(user_answer, str) else None
    elif question.question_type is QuestionType.TRUE_FALSE:
        is_correct = isinstance(user_answer, bool) and user_answer == question.boolean_answer
        normalized_answer = str(user_answer) if isinstance(user_answer, bool) else None
    elif isinstance(user_answer, str):
        normalized_answer = normalize_answer(user_answer)
        is_correct = _matches_variants(user_answer, question)
        if not is_correct and question.question_type is QuestionType.SHORT_ANSWER:
            is_correct = _conservative_token_overlap(user_answer, question)

    return QuestionScore(
        question_id=question.id,
        question_type=question.question_type,
        user_answer=user_answer,
        expected_answer=question.correct_answer,
        is_correct=is_correct,
        status=AnswerStatus.CORRECT if is_correct else AnswerStatus.INCORRECT,
        normalized_user_answer=normalized_answer,
    )


def score_quiz(quiz: Quiz, answers: Mapping[str, AnswerValue]) -> QuizScore:
    """Score all questions once; absent responses intentionally remain unanswered and incorrect."""

    results = tuple(
        score_question(question, answers.get(question.id)) for question in quiz.questions
    )
    correct_count = sum(result.is_correct for result in results)
    unanswered_count = sum(result.status is AnswerStatus.UNANSWERED for result in results)
    incorrect_count = len(results) - correct_count
    by_type = {}
    for question_type in QuestionType:
        type_results = tuple(result for result in results if result.question_type is question_type)
        if type_results:
            type_correct = sum(result.is_correct for result in type_results)
            type_unanswered = sum(
                result.status is AnswerStatus.UNANSWERED for result in type_results
            )
            by_type[question_type] = TypeScore(
                total=len(type_results),
                correct=type_correct,
                incorrect=len(type_results) - type_correct,
                unanswered=type_unanswered,
            )
    percentage = round((correct_count / len(results) * 100) if results else 0.0, 2)
    return QuizScore(
        quiz_id=quiz.id,
        total_questions=len(results),
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        percentage=percentage,
        by_type=by_type,
        question_results=results,
    )
