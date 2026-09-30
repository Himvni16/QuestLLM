"""Offline tests for conservative, type-aware quiz scoring."""

from questllm.questions import AnswerChoice, Difficulty, Question, QuestionType
from questllm.quiz import Quiz
from questllm.scoring import AnswerStatus, QuizScore, score_question, score_quiz


def _question(
    question_type: QuestionType,
    *,
    identifier: str,
    answer: str,
    variants: tuple[str, ...] = (),
    choices: tuple[AnswerChoice, ...] = (),
    boolean_answer: bool | None = None,
) -> Question:
    return Question(
        id=identifier,
        question_type=question_type,
        prompt="Which concise factual response is correct?",
        correct_answer=answer,
        accepted_answer_variants=variants or (answer,),
        choices=choices,
        boolean_answer=boolean_answer,
        source_excerpt=f"{answer} is stated in the source material.",
        source_sentence=f"{answer} is stated in the source material.",
        sentence_index=0,
        paragraph_index=0,
        page_number=1,
        difficulty=Difficulty.EASY,
    )


def _quiz(*questions: Question) -> Quiz:
    return Quiz(
        id="quiz-1",
        questions=questions,
        requested_question_count=len(questions),
        actual_question_count=len(questions),
        selected_question_types=tuple(dict.fromkeys(q.question_type for q in questions)),
        difficulty=Difficulty.EASY,
        source_metadata={"source_type": "text"},
        generation_seed=17,
    )


def _mixed_questions() -> tuple[Question, ...]:
    return (
        _question(
            QuestionType.MULTIPLE_CHOICE,
            identifier="mcq",
            answer="chlorophyll",
            choices=(
                AnswerChoice("mcq-correct", "chlorophyll", True),
                AnswerChoice("mcq-wrong", "mitochondria", False),
                AnswerChoice("mcq-two", "ribosome", False),
                AnswerChoice("mcq-three", "nucleus", False),
            ),
        ),
        _question(
            QuestionType.TRUE_FALSE,
            identifier="tf",
            answer="True",
            boolean_answer=True,
        ),
        _question(
            QuestionType.FILL_IN_THE_BLANK,
            identifier="blank",
            answer="DNA",
        ),
        _question(
            QuestionType.SHORT_ANSWER,
            identifier="short",
            answer="cellular respiration",
        ),
    )


def _assert_counts_partition_total(result: QuizScore) -> None:
    assert result.correct_count + result.incorrect_count + result.unanswered_count == (
        result.total_questions
    )


def test_mcq_and_true_false_scoring_use_stable_values_and_handle_missing_answers() -> None:
    mcq = _question(
        QuestionType.MULTIPLE_CHOICE,
        identifier="mcq",
        answer="chlorophyll",
        choices=(
            AnswerChoice("choice-correct", "chlorophyll", True),
            AnswerChoice("choice-wrong", "mitochondria", False),
            AnswerChoice("choice-two", "ribosome", False),
            AnswerChoice("choice-three", "nucleus", False),
        ),
    )
    true_false = _question(
        QuestionType.TRUE_FALSE,
        identifier="tf",
        answer="True",
        boolean_answer=True,
    )

    assert score_question(mcq, "choice-correct").status is AnswerStatus.CORRECT
    assert score_question(mcq, "choice-wrong").status is AnswerStatus.INCORRECT
    assert score_question(mcq, None).status is AnswerStatus.UNANSWERED
    assert score_question(true_false, True).status is AnswerStatus.CORRECT
    assert score_question(true_false, False).status is AnswerStatus.INCORRECT
    assert score_question(true_false, None).status is AnswerStatus.UNANSWERED


def test_fill_in_the_blank_normalizes_safe_variants_without_fuzzy_grading() -> None:
    question = _question(
        QuestionType.FILL_IN_THE_BLANK,
        identifier="blank",
        answer="75 percent",
        variants=("75 percent", "75%"),
    )

    assert score_question(question, "75 percent").is_correct is True
    assert score_question(question, "  75   PERCENT! ").is_correct is True
    assert score_question(question, "75%").is_correct is True
    assert score_question(question, "70 percent").is_correct is False


def test_short_answer_supports_exact_variants_and_conservative_token_overlap() -> None:
    question = _question(
        QuestionType.SHORT_ANSWER,
        identifier="short",
        answer="cellular respiration",
        variants=("cellular respiration", "respiration in cells"),
    )
    short_answer = _question(
        QuestionType.SHORT_ANSWER,
        identifier="dna",
        answer="DNA",
    )

    assert score_question(question, "cellular respiration").is_correct is True
    assert score_question(question, "respiration in cells").is_correct is True
    assert score_question(question, "respiration cellular").is_correct is True
    assert score_question(question, "photosynthesis process").is_correct is False
    assert score_question(short_answer, "RNA").is_correct is False


def test_six_question_summary_counts_unanswered_exclusively() -> None:
    questions = tuple(
        _question(
            QuestionType.FILL_IN_THE_BLANK,
            identifier=f"blank-{index}",
            answer="DNA",
        )
        for index in range(6)
    )
    answers = {f"blank-{index}": "DNA" for index in range(4)}
    answers["blank-4"] = "RNA"

    result = score_quiz(_quiz(*questions), answers)

    assert result.correct_count == 4
    assert result.incorrect_count == 1
    assert result.unanswered_count == 1
    assert result.total_questions == 6
    assert result.percentage == 66.67
    _assert_counts_partition_total(result)


def test_all_unanswered_questions_do_not_increment_incorrect() -> None:
    result = score_quiz(_quiz(*_mixed_questions()), {})

    assert result.correct_count == 0
    assert result.incorrect_count == 0
    assert result.unanswered_count == result.total_questions == 4
    assert result.percentage == 0.0
    _assert_counts_partition_total(result)


def test_all_incorrect_answered_questions_do_not_increment_unanswered() -> None:
    result = score_quiz(
        _quiz(*_mixed_questions()),
        {
            "mcq": "mcq-wrong",
            "tf": False,
            "blank": "RNA",
            "short": "photosynthesis",
        },
    )

    assert result.correct_count == 0
    assert result.incorrect_count == result.total_questions == 4
    assert result.unanswered_count == 0
    assert result.percentage == 0.0
    _assert_counts_partition_total(result)


def test_mixed_question_types_have_mutually_exclusive_overall_and_type_counts() -> None:
    result = score_quiz(
        _quiz(*_mixed_questions()),
        {
            "mcq": "mcq-correct",
            "tf": False,
            "short": "cellular respiration",
        },
    )

    assert result.correct_count == 2
    assert result.incorrect_count == 1
    assert result.unanswered_count == 1
    assert result.percentage == 50.0
    _assert_counts_partition_total(result)

    assert result.by_type[QuestionType.MULTIPLE_CHOICE].correct == 1
    assert result.by_type[QuestionType.TRUE_FALSE].incorrect == 1
    assert result.by_type[QuestionType.FILL_IN_THE_BLANK].unanswered == 1
    assert result.by_type[QuestionType.SHORT_ANSWER].correct == 1
    for type_score in result.by_type.values():
        assert type_score.correct + type_score.incorrect + type_score.unanswered == type_score.total
