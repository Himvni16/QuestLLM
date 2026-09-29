"""Minimal, provenance-preserving domain models for QuestLLM question types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class QuestionType(StrEnum):
    """Question types supported by QuestLLM's first builder phase."""

    MULTIPLE_CHOICE = "Multiple Choice"
    TRUE_FALSE = "True/False"
    FILL_IN_THE_BLANK = "Fill-in-the-Blank"
    SHORT_ANSWER = "Short Answer"


class Difficulty(StrEnum):
    """Practical labels that guide lightweight builder heuristics."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


@dataclass(frozen=True, slots=True)
class AnswerChoice:
    """A stable MCQ choice whose identity is independent of display position."""

    id: str
    text: str
    is_correct: bool = False


@dataclass(frozen=True, slots=True)
class Question:
    """A type-safe, source-grounded question ready for a later quiz interface."""

    id: str
    question_type: QuestionType
    prompt: str
    correct_answer: str
    accepted_answer_variants: tuple[str, ...]
    choices: tuple[AnswerChoice, ...]
    boolean_answer: bool | None
    source_excerpt: str
    source_sentence: str
    sentence_index: int
    paragraph_index: int
    page_number: int | None
    difficulty: Difficulty
    metadata: Mapping[str, int | str | bool] = field(default_factory=dict)
