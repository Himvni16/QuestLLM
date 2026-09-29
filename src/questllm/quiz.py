"""The minimal final-assembly representation for a generated QuestLLM quiz."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from questllm.questions import Difficulty, Question, QuestionType


@dataclass(frozen=True, slots=True)
class Quiz:
    """An ordered, provenance-preserving quiz that may safely contain a partial result."""

    id: str
    questions: tuple[Question, ...]
    requested_question_count: int
    actual_question_count: int
    selected_question_types: tuple[QuestionType, ...]
    difficulty: Difficulty
    source_metadata: Mapping[str, str]
    generation_seed: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
