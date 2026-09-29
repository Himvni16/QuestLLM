"""Offline tests for conservative, type-aware quiz scoring."""

from questllm.questions import AnswerChoice, Difficulty, Question, QuestionType
from questllm.quiz import Quiz
from questllm.scoring import AnswerStatus, score_question, score_quiz


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


def test_overall_score_counts_unanswered_as_incorrect_and_breaks_down_types() -> None:
    mcq = _question(
        QuestionType.MULTIPLE_CHOICE,
        identifier="mcq",
        answer="chlorophyll",
        choices=(
            AnswerChoice("correct", "chlorophyll", True),
            AnswerChoice("wrong", "mitochondria", False),
            AnswerChoice("two", "ribosome", False),
            AnswerChoice("three", "nucleus", False),
        ),
    )
    blank = _question(QuestionType.FILL_IN_THE_BLANK, identifier="blank", answer="DNA")
    quiz = Quiz(
        id="quiz-1",
        questions=(mcq, blank),
        requested_question_count=2,
        actual_question_count=2,
        selected_question_types=(QuestionType.MULTIPLE_CHOICE, QuestionType.FILL_IN_THE_BLANK),
        difficulty=Difficulty.EASY,
        source_metadata={"source_type": "text"},
        generation_seed=17,
    )

    result = score_quiz(quiz, {"mcq": "correct"})

    assert result.correct_count == 1
    assert result.incorrect_count == 1
    assert result.unanswered_count == 1
    assert result.percentage == 50.0
    assert result.by_type[QuestionType.FILL_IN_THE_BLANK].unanswered == 1
