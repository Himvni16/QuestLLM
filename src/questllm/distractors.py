"""Conservative, deterministic distractor selection for grounded QuestLLM MCQs."""

from collections.abc import Sequence

from questllm.candidates import CandidateAnswer, CandidateType
from questllm.questions import Difficulty

_CONCEPT_TYPES = frozenset(
    {
        CandidateType.DEFINITION_TERM,
        CandidateType.NOUN_PHRASE,
        CandidateType.GENERAL_CONCEPT,
    }
)


def _compatible_type(target: CandidateAnswer, alternative: CandidateAnswer) -> bool:
    if target.candidate_type == alternative.candidate_type:
        return True
    return target.candidate_type in _CONCEPT_TYPES and alternative.candidate_type in _CONCEPT_TYPES


def _has_unsafe_overlap(target: str, alternative: str) -> bool:
    target_normalized = target.casefold().strip()
    alternative_normalized = alternative.casefold().strip()
    return (
        target_normalized == alternative_normalized
        or target_normalized in alternative_normalized
        or alternative_normalized in target_normalized
    )


def _context_similarity(first: str, second: str) -> float:
    """Return a small lexical signal without introducing another model dependency."""

    first_terms = {term.casefold().strip(".,;:!?") for term in first.split()}
    second_terms = {term.casefold().strip(".,;:!?") for term in second.split()}
    first_terms.discard("")
    second_terms.discard("")
    if not first_terms or not second_terms:
        return 0.0
    return len(first_terms & second_terms) / len(first_terms | second_terms)


def select_distractors(
    correct_answer: CandidateAnswer,
    candidates: Sequence[CandidateAnswer],
    *,
    difficulty: Difficulty,
    limit: int = 3,
) -> tuple[CandidateAnswer, ...]:
    """Return only defensible same-type alternatives, in a stable rank order."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    eligible = []
    for candidate in candidates:
        if not _compatible_type(correct_answer, candidate):
            continue
        if _has_unsafe_overlap(correct_answer.normalized_text, candidate.normalized_text):
            continue
        # An alternative named in the same statement may be another supported answer.
        if candidate.text.casefold() in correct_answer.source_sentence.casefold():
            continue
        eligible.append(candidate)

    def sort_key(candidate: CandidateAnswer) -> tuple[float, float, int, str, int]:
        similarity = _context_similarity(correct_answer.source_sentence, candidate.source_sentence)
        length_distance = abs(len(correct_answer.text) - len(candidate.text))
        if difficulty is Difficulty.EASY:
            plausibility = -similarity
        elif difficulty is Difficulty.HARD:
            plausibility = similarity
        else:
            plausibility = similarity * 0.5
        return (
            -plausibility,
            length_distance,
            -candidate.importance_score,
            candidate.normalized_text,
            candidate.sentence_index,
        )

    unique: dict[str, CandidateAnswer] = {}
    for candidate in sorted(eligible, key=sort_key):
        unique.setdefault(candidate.normalized_text, candidate)
    return tuple(unique.values())[:limit]
