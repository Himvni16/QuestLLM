"""Deterministic builders that turn grounded candidates and stems into question types."""

import random
import re
import uuid
from collections.abc import Sequence

from questllm.candidates import CandidateAnswer, normalize_candidate
from questllm.config import FILL_IN_THE_BLANK_TOKEN, MCQ_CHOICE_ORDER_SEED
from questllm.distractors import select_distractors
from questllm.exceptions import GenerationError
from questllm.generation import QuestionGenerationResult, validate_generated_question
from questllm.questions import AnswerChoice, Difficulty, Question, QuestionType

_QUESTION_NAMESPACE = uuid.UUID("82d42780-60ca-4ecf-a293-8d628d96e6da")


def _candidate_occurrences(text: str, candidate_text: str) -> list[re.Match[str]]:
    pattern = rf"(?<!\w){re.escape(candidate_text)}(?!\w)"
    return list(re.finditer(pattern, text, flags=re.IGNORECASE))


def _stable_question_id(
    question_type: QuestionType,
    candidate: CandidateAnswer,
    prompt: str,
) -> str:
    identity = "|".join(
        (
            question_type.value,
            candidate.normalized_text,
            str(candidate.sentence_index),
            candidate.source_sentence,
            prompt,
        )
    )
    return str(uuid.uuid5(_QUESTION_NAMESPACE, identity))


def _accepted_variants(candidate: CandidateAnswer) -> tuple[str, ...]:
    """Keep only exact and normalized forms; semantic aliases remain intentionally deferred."""

    return tuple(dict.fromkeys((candidate.text, candidate.normalized_text)))


def _make_question(
    *,
    question_type: QuestionType,
    candidate: CandidateAnswer,
    prompt: str,
    difficulty: Difficulty,
    choices: tuple[AnswerChoice, ...] = (),
    boolean_answer: bool | None = None,
    metadata: dict[str, int | float | str | bool] | None = None,
) -> Question:
    question_metadata: dict[str, int | float | str | bool] = {
        "candidate_importance": candidate.importance_score,
    }
    question_metadata.update(metadata or {})
    return Question(
        id=_stable_question_id(question_type, candidate, prompt),
        question_type=question_type,
        prompt=prompt,
        correct_answer=candidate.text if boolean_answer is None else str(boolean_answer),
        accepted_answer_variants=(
            _accepted_variants(candidate) if boolean_answer is None else (str(boolean_answer),)
        ),
        choices=choices,
        boolean_answer=boolean_answer,
        source_excerpt=candidate.source_sentence,
        source_sentence=candidate.source_sentence,
        sentence_index=candidate.sentence_index,
        paragraph_index=candidate.paragraph_index,
        page_number=candidate.page_number,
        difficulty=difficulty,
        metadata=question_metadata,
    )


def build_short_answer(
    stem: QuestionGenerationResult,
    *,
    difficulty: Difficulty,
) -> Question | None:
    """Build a concise, answer-aware question from an already grounded T5 result."""

    candidate = stem.candidate
    relation_prompts = {
        "common task": f"What is a common task of {candidate.subject}?",
        "stores": f"What does {candidate.subject} store?",
        "converts": f"What does {candidate.subject} convert energy into?",
        "are": f"What are {candidate.subject} used to treat?",
    }
    prompt = relation_prompts.get(candidate.relation or "") if candidate.subject else None
    if prompt is None:
        try:
            prompt = validate_generated_question(stem.question, candidate)
        except GenerationError:
            return None
    return _make_question(
        question_type=QuestionType.SHORT_ANSWER,
        candidate=candidate,
        prompt=prompt,
        difficulty=difficulty,
        metadata={"model_id": stem.model_id, "source": "t5"},
    )


def build_fill_in_the_blank(
    candidate: CandidateAnswer,
    *,
    difficulty: Difficulty,
) -> Question | None:
    """Blank exactly one unambiguous answer span while retaining the original punctuation."""

    matches = _candidate_occurrences(candidate.source_sentence, candidate.text)
    if len(matches) != 1:
        return None
    match = matches[0]
    surrounding_terms = (
        candidate.source_sentence[: match.start()] + candidate.source_sentence[match.end() :]
    ).split()
    if len(surrounding_terms) < 4:
        return None
    prompt = (
        candidate.source_sentence[: match.start()]
        + FILL_IN_THE_BLANK_TOKEN
        + candidate.source_sentence[match.end() :]
    )
    if prompt.count(FILL_IN_THE_BLANK_TOKEN) != 1:
        return None
    return _make_question(
        question_type=QuestionType.FILL_IN_THE_BLANK,
        candidate=candidate,
        prompt=prompt,
        difficulty=difficulty,
        metadata={"blank_token": FILL_IN_THE_BLANK_TOKEN, "source": "deterministic"},
    )


def _false_replacement(
    candidate: CandidateAnswer,
    candidates: Sequence[CandidateAnswer],
    difficulty: Difficulty,
) -> CandidateAnswer | None:
    """Choose a compatible documented alternative, never a negation-based fabrication."""

    distractors = select_distractors(candidate, candidates, difficulty=difficulty, limit=1)
    return distractors[0] if distractors else None


def build_true_false(
    candidate: CandidateAnswer,
    candidates: Sequence[CandidateAnswer],
    *,
    truth_value: bool,
    difficulty: Difficulty,
) -> Question | None:
    """Build either a source fact or a controlled same-type factual replacement."""

    matches = _candidate_occurrences(candidate.source_sentence, candidate.text)
    if len(matches) != 1 or not candidate.source_sentence.strip():
        return None
    if truth_value:
        return _make_question(
            question_type=QuestionType.TRUE_FALSE,
            candidate=candidate,
            prompt=candidate.source_sentence,
            difficulty=difficulty,
            boolean_answer=True,
            metadata={"source": "original statement", "statement_truth": True},
        )

    replacement = _false_replacement(candidate, candidates, difficulty)
    if replacement is None:
        return None
    match = matches[0]
    prompt = (
        candidate.source_sentence[: match.start()]
        + replacement.text
        + candidate.source_sentence[match.end() :]
    )
    if normalize_candidate(prompt) == normalize_candidate(candidate.source_sentence):
        return None
    return _make_question(
        question_type=QuestionType.TRUE_FALSE,
        candidate=candidate,
        prompt=prompt,
        difficulty=difficulty,
        boolean_answer=False,
        metadata={
            "source": "controlled replacement",
            "statement_truth": False,
            "replaced_with": replacement.text,
        },
    )


def _build_choices(
    question_id: str,
    candidate: CandidateAnswer,
    distractors: Sequence[CandidateAnswer],
    *,
    seed: int,
) -> tuple[AnswerChoice, ...]:
    raw_choices = [(candidate.text, True), *((item.text, False) for item in distractors)]
    chooser = random.Random(f"{seed}:{question_id}")
    chooser.shuffle(raw_choices)
    return tuple(
        AnswerChoice(
            id=str(uuid.uuid5(_QUESTION_NAMESPACE, f"{question_id}|{text.casefold()}")),
            text=text,
            is_correct=is_correct,
        )
        for text, is_correct in raw_choices
    )


def build_multiple_choice(
    stem: QuestionGenerationResult,
    candidates: Sequence[CandidateAnswer],
    *,
    difficulty: Difficulty,
    choice_seed: int = MCQ_CHOICE_ORDER_SEED,
) -> Question | None:
    """Build a four-choice MCQ only when three unique, defensible distractors exist."""

    try:
        prompt = validate_generated_question(stem.question, stem.candidate)
    except GenerationError:
        return None
    distractors = select_distractors(stem.candidate, candidates, difficulty=difficulty, limit=3)
    if len(distractors) != 3:
        return None
    question_id = _stable_question_id(QuestionType.MULTIPLE_CHOICE, stem.candidate, prompt)
    choices = _build_choices(question_id, stem.candidate, distractors, seed=choice_seed)
    if len(choices) != 4 or sum(choice.is_correct for choice in choices) != 1:
        return None
    if len({normalize_candidate(choice.text) for choice in choices}) != 4:
        return None
    return _make_question(
        question_type=QuestionType.MULTIPLE_CHOICE,
        candidate=stem.candidate,
        prompt=prompt,
        difficulty=difficulty,
        choices=choices,
        metadata={"choice_seed": choice_seed, "model_id": stem.model_id, "source": "t5"},
    )


class QuestionBuilderService:
    """Build selected question types while preserving order and safely skipping weak material."""

    def __init__(
        self,
        candidates: Sequence[CandidateAnswer],
        stems: Sequence[QuestionGenerationResult],
    ) -> None:
        self.candidates = tuple(candidates)
        self.stems = {stem.candidate.normalized_text: stem for stem in stems}

    def build_preview(
        self,
        question_types: Sequence[QuestionType],
        *,
        difficulty: Difficulty,
        per_type_limit: int = 1,
    ) -> tuple[Question, ...]:
        """Return a small grouped preview in the caller's requested type order."""

        if per_type_limit < 1:
            raise ValueError("per_type_limit must be at least 1.")
        questions = []
        for question_type in question_types:
            built_for_type = 0
            false_next = False
            for candidate in self.candidates:
                if built_for_type >= per_type_limit:
                    break
                stem = self.stems.get(candidate.normalized_text)
                if question_type is QuestionType.SHORT_ANSWER and stem:
                    question = build_short_answer(stem, difficulty=difficulty)
                elif question_type is QuestionType.MULTIPLE_CHOICE and stem:
                    question = build_multiple_choice(
                        stem,
                        self.candidates,
                        difficulty=difficulty,
                    )
                elif question_type is QuestionType.FILL_IN_THE_BLANK:
                    question = build_fill_in_the_blank(candidate, difficulty=difficulty)
                elif question_type is QuestionType.TRUE_FALSE:
                    question = build_true_false(
                        candidate,
                        self.candidates,
                        truth_value=not false_next,
                        difficulty=difficulty,
                    )
                    if question is None and not false_next:
                        question = build_true_false(
                            candidate,
                            self.candidates,
                            truth_value=False,
                            difficulty=difficulty,
                        )
                else:
                    question = None
                if question:
                    questions.append(question)
                    built_for_type += 1
                    if question_type is QuestionType.TRUE_FALSE:
                        false_next = question.boolean_answer is True
        return tuple(questions)
