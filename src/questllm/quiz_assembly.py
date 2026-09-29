"""Bounded orchestration from processed QuestLLM material to a quality-filtered Quiz."""

import random
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from questllm.candidates import CandidateAnswer, extract_and_rank_candidates
from questllm.chunking import TextChunk, chunk_sentences
from questllm.config import (
    MAXIMUM_CANDIDATES_ATTEMPTED,
    MAXIMUM_QUIZ_QUESTION_COUNT,
    MAXIMUM_STEM_GENERATION_ATTEMPTS,
    MINIMUM_QUIZ_QUESTION_COUNT,
    QUESTION_QUALITY_CUTOFF,
    QUIZ_COVERAGE_BONUS,
)
from questllm.document import Document
from questllm.generation import (
    QuestionGenerationResult,
    QuestionGenerationSettings,
    QuestionGenerator,
    QuestionPreviewService,
)
from questllm.preprocessing import ProcessedSentence, preprocess_document
from questllm.question_builders import (
    build_fill_in_the_blank,
    build_multiple_choice,
    build_short_answer,
    build_true_false,
)
from questllm.questions import Difficulty, Question, QuestionType
from questllm.quiz import Quiz
from questllm.validation import ScoredQuestion, is_duplicate_question, score_question

_QUIZ_NAMESPACE = uuid.UUID("1dafb7f2-7432-42cb-9b18-5cb76dc52e1f")


def allocate_question_quotas(
    requested_count: int,
    question_types: Sequence[QuestionType],
) -> dict[QuestionType, int]:
    """Split an initial target evenly across selected types in the supplied order."""

    selected = tuple(dict.fromkeys(question_types))
    if requested_count < 1 or not selected:
        return {}
    base, remainder = divmod(requested_count, len(selected))
    return {
        question_type: base + (1 if index < remainder else 0)
        for index, question_type in enumerate(selected)
    }


class QuizAssemblyService:
    """Build, validate, deduplicate, allocate, and backfill a bounded quiz result."""

    def __init__(
        self,
        candidates: Sequence[CandidateAnswer],
        stems: Sequence[QuestionGenerationResult],
        *,
        source_metadata: Mapping[str, str],
    ) -> None:
        self.candidates = tuple(candidates)
        self.stems = {stem.candidate.normalized_text: stem for stem in stems}
        self.source_metadata = dict(source_metadata)

    def _build_pool(
        self,
        question_type: QuestionType,
        difficulty: Difficulty,
    ) -> tuple[ScoredQuestion, ...]:
        """Build a bounded source-order pool for one type without forcing weak questions."""

        pool = []
        false_next = False
        for candidate in self.candidates[:MAXIMUM_CANDIDATES_ATTEMPTED]:
            stem = self.stems.get(candidate.normalized_text)
            if question_type is QuestionType.MULTIPLE_CHOICE and stem:
                question = build_multiple_choice(stem, self.candidates, difficulty=difficulty)
            elif question_type is QuestionType.SHORT_ANSWER and stem:
                question = build_short_answer(stem, difficulty=difficulty)
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
                if question:
                    false_next = question.boolean_answer is True
            else:
                question = None
            if question:
                scored = score_question(question)
                if scored.quality_score >= QUESTION_QUALITY_CUTOFF:
                    pool.append(scored)
        return tuple(pool)

    @staticmethod
    def _coverage_key(question: Question) -> tuple[int | None, int, int]:
        return (question.page_number, question.paragraph_index, question.sentence_index)

    def _select_one(
        self,
        pool: Sequence[ScoredQuestion],
        accepted: Sequence[ScoredQuestion],
        coverage: Counter[tuple[int | None, int, int]],
        tie_breakers: Mapping[str, float],
    ) -> ScoredQuestion | None:
        eligible = [
            scored
            for scored in pool
            if not is_duplicate_question(scored, accepted) and scored.question.id not in {
                item.question.id for item in accepted
            }
        ]
        if not eligible:
            return None

        def selection_key(scored: ScoredQuestion) -> tuple[float, float, float, str]:
            coverage_bonus = (
                QUIZ_COVERAGE_BONUS if coverage[self._coverage_key(scored.question)] == 0 else 0.0
            )
            return (
                -(scored.quality_score + coverage_bonus),
                -scored.quality_score,
                -tie_breakers[scored.question.id],
                scored.question.id,
            )

        return min(eligible, key=selection_key)

    def assemble(
        self,
        *,
        requested_count: int,
        question_types: Sequence[QuestionType],
        difficulty: Difficulty,
        seed: int,
        initial_warnings: Sequence[str] = (),
    ) -> Quiz:
        """Allocate initial quotas, backfill safely, and return a complete or partial quiz."""

        selected_types = tuple(dict.fromkeys(question_types))
        if not MINIMUM_QUIZ_QUESTION_COUNT <= requested_count <= MAXIMUM_QUIZ_QUESTION_COUNT:
            raise ValueError(
                f"requested_count must be between {MINIMUM_QUIZ_QUESTION_COUNT} and "
                f"{MAXIMUM_QUIZ_QUESTION_COUNT}."
            )
        if not selected_types:
            raise ValueError("At least one question type must be selected.")

        pools = {
            question_type: self._build_pool(question_type, difficulty)
            for question_type in selected_types
        }
        quotas = allocate_question_quotas(requested_count, selected_types)
        randomizer = random.Random(seed)
        all_ids = sorted(
            scored.question.id for pool in pools.values() for scored in pool
        )
        tie_breakers = {question_id: randomizer.random() for question_id in all_ids}
        accepted: list[ScoredQuestion] = []
        coverage: Counter[tuple[int | None, int, int]] = Counter()

        def accept_from_pool(pool: Sequence[ScoredQuestion]) -> bool:
            selected = self._select_one(pool, accepted, coverage, tie_breakers)
            if selected is None:
                return False
            accepted.append(selected)
            coverage[self._coverage_key(selected.question)] += 1
            return True

        for question_type in selected_types:
            for _ in range(quotas[question_type]):
                if not accept_from_pool(pools[question_type]):
                    break

        while len(accepted) < requested_count:
            added = False
            for question_type in selected_types:
                if len(accepted) >= requested_count:
                    break
                added = accept_from_pool(pools[question_type]) or added
            if not added:
                break

        ordered = list(accepted)
        random.Random(seed).shuffle(ordered)
        warnings = list(initial_warnings)
        if len(ordered) < requested_count:
            warnings.append(
                f"Requested {requested_count} questions, but only {len(ordered)} high-quality "
                "grounded questions could be generated from this source."
            )
        source_fingerprint = "|".join(
            [
                *sorted(f"{key}={value}" for key, value in self.source_metadata.items()),
                *sorted(question.question.id for question in ordered),
            ]
        )
        quiz_identity = "|".join(
            (
                source_fingerprint,
                str(requested_count),
                ",".join(question_type.value for question_type in selected_types),
                difficulty.value,
                str(seed),
            )
        )
        return Quiz(
            id=str(uuid.uuid5(_QUIZ_NAMESPACE, quiz_identity)),
            questions=tuple(scored.question for scored in ordered),
            requested_question_count=requested_count,
            actual_question_count=len(ordered),
            selected_question_types=selected_types,
            difficulty=difficulty,
            source_metadata=self.source_metadata,
            generation_seed=seed,
            warnings=tuple(warnings),
        )


class QuizGenerationPipeline:
    """Coordinate existing phases while keeping expensive model calls bounded and reusable."""

    def __init__(
        self,
        question_generator: QuestionGenerator | None = None,
        *,
        document_preprocessor: Callable[
            [Document], tuple[ProcessedSentence, ...]
        ] = preprocess_document,
        sentence_chunker: Callable[
            [tuple[ProcessedSentence, ...]], tuple[TextChunk, ...]
        ] = chunk_sentences,
        candidate_extractor: Callable[
            [tuple[ProcessedSentence, ...]], tuple[CandidateAnswer, ...]
        ] = (
            extract_and_rank_candidates
        ),
    ) -> None:
        self.question_generator = question_generator
        self.document_preprocessor = document_preprocessor
        self.sentence_chunker = sentence_chunker
        self.candidate_extractor = candidate_extractor

    def build_from_document(
        self,
        document: Document,
        *,
        requested_count: int,
        question_types: Sequence[QuestionType],
        difficulty: Difficulty,
        seed: int,
    ) -> Quiz:
        """Execute the documented phase pipeline from an ingested document to a final Quiz."""

        sentences = self.document_preprocessor(document)
        chunks = self.sentence_chunker(sentences)
        candidates = self.candidate_extractor(sentences)
        return self.build_from_processed(
            document,
            chunks=chunks,
            candidates=candidates,
            requested_count=requested_count,
            question_types=question_types,
            difficulty=difficulty,
            seed=seed,
        )

    def build_from_processed(
        self,
        document: Document,
        *,
        chunks: Sequence[TextChunk],
        candidates: Sequence[CandidateAnswer],
        requested_count: int,
        question_types: Sequence[QuestionType],
        difficulty: Difficulty,
        seed: int,
        pre_generated_stems: Sequence[QuestionGenerationResult] = (),
    ) -> Quiz:
        """Reuse existing processed content, creating T5 stems only for types that need them."""

        warnings = []
        requires_stems = any(
            question_type in {QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER}
            for question_type in question_types
        )
        stems = tuple(pre_generated_stems)
        if requires_stems and self.question_generator is None:
            warnings.append(
                "Question stems were unavailable, so stem-based question types were skipped."
            )
        elif requires_stems:
            completed_candidates = {stem.candidate.normalized_text for stem in stems}
            remaining_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.normalized_text not in completed_candidates
            )
            new_stems = QuestionPreviewService(self.question_generator).generate_preview(
                chunks,
                remaining_candidates[:MAXIMUM_STEM_GENERATION_ATTEMPTS],
                settings=QuestionGenerationSettings(
                    preview_limit=min(
                        len(remaining_candidates),
                        MAXIMUM_STEM_GENERATION_ATTEMPTS,
                    )
                ),
            )
            stems = (*stems, *new_stems)
        source_metadata = {"source_type": document.source_type.value, **dict(document.metadata)}
        return QuizAssemblyService(
            candidates,
            stems,
            source_metadata=source_metadata,
        ).assemble(
            requested_count=requested_count,
            question_types=question_types,
            difficulty=difficulty,
            seed=seed,
            initial_warnings=warnings,
        )
