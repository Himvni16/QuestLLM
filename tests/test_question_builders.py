"""Offline tests for QuestLLM's four grounded question-type builders."""

from questllm.candidates import CandidateAnswer, CandidateType
from questllm.generation import QuestionGenerationResult
from questllm.question_builders import (
    QuestionBuilderService,
    build_fill_in_the_blank,
    build_multiple_choice,
    build_short_answer,
    build_true_false,
)
from questllm.questions import Difficulty, QuestionType


def _candidate(
    text: str,
    sentence: str,
    *,
    candidate_type: CandidateType = CandidateType.GENERAL_CONCEPT,
    index: int = 0,
    page: int | None = 2,
) -> CandidateAnswer:
    return CandidateAnswer(
        text=text,
        normalized_text=text.casefold(),
        source_sentence=sentence,
        sentence_index=index,
        paragraph_index=index,
        page_number=page,
        importance_score=0.9 - index / 100,
        candidate_type=candidate_type,
    )


def _stem(
    candidate: CandidateAnswer,
    question: str = "Which substance absorbs light energy?",
) -> QuestionGenerationResult:
    return QuestionGenerationResult(
        question=question,
        candidate=candidate,
        source_context=candidate.source_sentence,
        source_sentence=candidate.source_sentence,
        sentence_index=candidate.sentence_index,
        paragraph_index=candidate.paragraph_index,
        page_number=candidate.page_number,
        model_id="fake/t5",
    )


def test_short_answer_keeps_validated_stem_answer_variants_and_provenance() -> None:
    candidate = _candidate("Chlorophyll", "Chlorophyll absorbs light energy in plant cells.")

    question = build_short_answer(_stem(candidate), difficulty=Difficulty.EASY)

    assert question is not None
    assert question.question_type is QuestionType.SHORT_ANSWER
    assert question.correct_answer == "Chlorophyll"
    assert question.accepted_answer_variants == ("Chlorophyll", "chlorophyll")
    assert question.page_number == 2
    assert question.source_sentence == candidate.source_sentence


def test_fill_in_the_blank_replaces_one_span_and_rejects_ambiguous_repetition() -> None:
    candidate = _candidate("chlorophyll", "Chlorophyll absorbs light energy in plant cells.")

    question = build_fill_in_the_blank(candidate, difficulty=Difficulty.EASY)

    assert question is not None
    assert question.prompt == "________ absorbs light energy in plant cells."
    assert question.prompt.count("________") == 1
    assert question.page_number == 2

    ambiguous = _candidate("DNA", "DNA helps DNA repair damage in cells.")
    assert build_fill_in_the_blank(ambiguous, difficulty=Difficulty.MEDIUM) is None


def test_true_false_builds_grounded_true_and_controlled_same_type_false() -> None:
    paris = _candidate(
        "Paris",
        "Paris is the capital of France and a major European city.",
        candidate_type=CandidateType.PROPER_NOUN,
    )
    rome = _candidate(
        "Rome",
        "Rome is an Italian city with ancient landmarks.",
        candidate_type=CandidateType.PROPER_NOUN,
        index=1,
    )

    true_question = build_true_false(
        paris,
        (paris, rome),
        truth_value=True,
        difficulty=Difficulty.EASY,
    )
    false_question = build_true_false(
        paris,
        (paris, rome),
        truth_value=False,
        difficulty=Difficulty.EASY,
    )

    assert true_question is not None and true_question.boolean_answer is True
    assert true_question.prompt == paris.source_sentence
    assert false_question is not None and false_question.boolean_answer is False
    assert false_question.prompt.startswith("Rome is the capital of France")
    assert false_question.metadata["replaced_with"] == "Rome"
    assert build_true_false(paris, (paris,), truth_value=False, difficulty=Difficulty.EASY) is None


def test_mcq_requires_three_unique_distractors_and_uses_stable_seeded_choices() -> None:
    correct = _candidate("chlorophyll", "Chlorophyll absorbs light energy in plant cells.")
    alternatives = (
        _candidate("mitochondria", "Mitochondria release energy in cells.", index=1),
        _candidate("ribosome", "A ribosome builds proteins in cells.", index=2),
        _candidate("nucleus", "The nucleus stores genetic material in cells.", index=3),
    )
    stem = _stem(correct)

    first = build_multiple_choice(stem, (correct, *alternatives), difficulty=Difficulty.MEDIUM)
    second = build_multiple_choice(stem, (correct, *alternatives), difficulty=Difficulty.MEDIUM)

    assert first is not None and second is not None
    assert len(first.choices) == 4
    assert sum(choice.is_correct for choice in first.choices) == 1
    assert sum(choice.text == "chlorophyll" for choice in first.choices) == 1
    assert len({choice.id for choice in first.choices}) == 4
    assert first.choices == second.choices
    assert (
        build_multiple_choice(stem, (correct, *alternatives[:2]), difficulty=Difficulty.MEDIUM)
        is None
    )


def test_mixed_builder_returns_selected_types_in_predictable_order_and_skips_failures() -> None:
    good = _candidate("chlorophyll", "Chlorophyll absorbs light energy in plant cells.")
    repeated = _candidate("DNA", "DNA helps DNA repair damage in cells.", index=1)
    service = QuestionBuilderService((good, repeated), (_stem(good),))

    questions = service.build_preview(
        (QuestionType.FILL_IN_THE_BLANK, QuestionType.SHORT_ANSWER),
        difficulty=Difficulty.EASY,
    )

    assert [question.question_type for question in questions] == [
        QuestionType.FILL_IN_THE_BLANK,
        QuestionType.SHORT_ANSWER,
    ]
    assert all(question.question_type is not QuestionType.MULTIPLE_CHOICE for question in questions)


def test_true_false_preview_alternates_truth_values_when_safe_replacements_exist() -> None:
    paris = _candidate(
        "Paris",
        "Paris is the capital of France and a major European city.",
        candidate_type=CandidateType.PROPER_NOUN,
    )
    rome = _candidate(
        "Rome",
        "Rome is an Italian city with ancient landmarks.",
        candidate_type=CandidateType.PROPER_NOUN,
        index=1,
    )
    service = QuestionBuilderService((paris, rome), ())

    questions = service.build_preview(
        (QuestionType.TRUE_FALSE,),
        difficulty=Difficulty.MEDIUM,
        per_type_limit=2,
    )

    assert [question.boolean_answer for question in questions] == [True, False]


def test_mixed_builder_safely_skips_candidates_that_cannot_form_requested_type() -> None:
    repeated = _candidate("DNA", "DNA helps DNA repair damage in cells.")

    questions = QuestionBuilderService((repeated,), ()).build_preview(
        (QuestionType.FILL_IN_THE_BLANK,),
        difficulty=Difficulty.EASY,
    )

    assert questions == ()
