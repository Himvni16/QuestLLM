"""Offline tests for question quality validation, scoring, and deduplication."""

from questllm.questions import AnswerChoice, Difficulty, Question, QuestionType
from questllm.validation import deduplicate_questions, score_question, validate_question


def _question(
    *,
    identifier: str = "question-1",
    prompt: str = "Which pigment absorbs light energy?",
    answer: str = "chlorophyll",
    source: str = "Chlorophyll absorbs light energy in plant cells.",
    question_type: QuestionType = QuestionType.SHORT_ANSWER,
    page: int | None = 2,
    choices: tuple[AnswerChoice, ...] = (),
    boolean_answer: bool | None = None,
    metadata: dict[str, int | float | str | bool] | None = None,
) -> Question:
    return Question(
        id=identifier,
        question_type=question_type,
        prompt=prompt,
        correct_answer=answer,
        accepted_answer_variants=(answer,),
        choices=choices,
        boolean_answer=boolean_answer,
        source_excerpt=source,
        source_sentence=source,
        sentence_index=0,
        paragraph_index=0,
        page_number=page,
        difficulty=Difficulty.EASY,
        metadata=metadata or {"candidate_importance": 0.8},
    )


def test_common_and_type_specific_validation_rejects_malformed_questions() -> None:
    valid = _question()
    assert validate_question(valid).is_valid is True

    leaked = _question(prompt="What is chlorophyll?")
    assert "directly reveals" in " ".join(validate_question(leaked).errors)

    malformed_mcq = _question(
        identifier="mcq",
        question_type=QuestionType.MULTIPLE_CHOICE,
        choices=(
            AnswerChoice("a", "chlorophyll", True),
            AnswerChoice("b", "chlorophyll", False),
        ),
    )
    errors = " ".join(validate_question(malformed_mcq).errors)
    assert "exactly four" in errors
    assert "duplicate choices" in errors

    invalid_blank = _question(
        identifier="blank",
        question_type=QuestionType.FILL_IN_THE_BLANK,
        prompt="________ absorbs ________.",
    )
    assert "exactly one blank" in " ".join(validate_question(invalid_blank).errors)

    false_without_replacement = _question(
        identifier="false",
        question_type=QuestionType.TRUE_FALSE,
        prompt="Chlorophyll absorbs light energy in plant cells.",
        answer="False",
        boolean_answer=False,
    )
    assert "controlled factual replacement" in " ".join(
        validate_question(false_without_replacement).errors
    )


def test_quality_scoring_rewards_grounded_provenance_and_is_deterministic() -> None:
    strong = _question()
    incomplete = _question(identifier="incomplete", page=None)
    incomplete = Question(
        **{
            **{field: getattr(incomplete, field) for field in incomplete.__dataclass_fields__},
            "sentence_index": -1,
        }
    )

    strong_score = score_question(strong)
    incomplete_score = score_question(incomplete)

    assert strong_score.quality_score > incomplete_score.quality_score
    assert strong_score == score_question(strong)


def test_deduplication_handles_exact_same_fact_and_near_duplicate_wording() -> None:
    exact = score_question(_question(identifier="exact"))
    casing = score_question(
        _question(identifier="casing", prompt="WHICH pigment absorbs   light energy?")
    )
    same_fact = score_question(
        _question(identifier="fact", prompt="Name the pigment that absorbs light energy.")
    )
    near_first = score_question(
        _question(
            identifier="near-first",
            prompt="Which organelle produces energy inside cells?",
            answer="mitochondria",
            source="Mitochondria produce energy inside cells.",
        )
    )
    near_second = score_question(
        _question(
            identifier="near-second",
            prompt="What organelle produces energy within cells?",
            answer="ribosome",
            source="Ribosomes build proteins within cells.",
        )
    )
    distinct = score_question(
        _question(
            identifier="distinct",
            prompt="Which molecule stores genetic information?",
            answer="DNA",
            source="DNA stores genetic information in cells.",
        )
    )

    retained = deduplicate_questions(
        (exact, casing, same_fact, near_first, near_second, distinct),
        near_duplicate_threshold=0.55,
    )

    retained_ids = {item.question.id for item in retained}
    assert len({"exact", "casing", "fact"} & retained_ids) == 1
    assert len({"near-first", "near-second"} & retained_ids) == 1
    assert "distinct" in retained_ids
