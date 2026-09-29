"""Offline tests for immutable quiz workflow and generic session-state transitions."""

import pytest

from questllm.document import Document, SourceType
from questllm.exceptions import WorkflowError
from questllm.questions import Difficulty, Question, QuestionType
from questllm.quiz import Quiz
from questllm.workflow import (
    WorkflowStage,
    begin_generation,
    complete_generation,
    get_workflow,
    record_answers,
    reset_workflow,
    save_workflow,
    submit_quiz,
)


def _quiz() -> Quiz:
    question = Question(
        id="question-1",
        question_type=QuestionType.FILL_IN_THE_BLANK,
        prompt="________ stores hereditary information in cells.",
        correct_answer="DNA",
        accepted_answer_variants=("DNA",),
        choices=(),
        boolean_answer=None,
        source_excerpt="DNA stores hereditary information in cells.",
        source_sentence="DNA stores hereditary information in cells.",
        sentence_index=0,
        paragraph_index=0,
        page_number=None,
        difficulty=Difficulty.EASY,
    )
    return Quiz(
        id="quiz-1",
        questions=(question,),
        requested_question_count=1,
        actual_question_count=1,
        selected_question_types=(QuestionType.FILL_IN_THE_BLANK,),
        difficulty=Difficulty.EASY,
        source_metadata={"source_type": "text"},
        generation_seed=17,
    )


def test_workflow_transitions_retain_answers_then_freeze_after_submission() -> None:
    document = Document(
        source_type=SourceType.TEXT,
        text="DNA stores hereditary information in cells.",
    )
    workflow = begin_generation(
        reset_workflow(),
        document=document,
        configuration={"requested_count": 1},
        seed=17,
    )
    assert workflow.stage is WorkflowStage.GENERATING
    assert workflow.quiz_configuration == {"requested_count": 1}
    assert workflow.generation_seed == 17
    assert workflow.source_identity is not None

    attempt = complete_generation(workflow, _quiz())
    assert attempt.stage is WorkflowStage.ATTEMPT
    saved = record_answers(attempt, {"question-1": "DNA", "unknown": "ignored"})
    assert saved.answers == {"question-1": "DNA"}
    submitted = submit_quiz(saved, saved.answers)

    assert submitted.stage is WorkflowStage.REVIEW
    assert submitted.answers == {"question-1": "DNA"}
    assert submitted.result is not None and submitted.result.correct_count == 1
    with pytest.raises(WorkflowError):
        record_answers(submitted, {"question-1": "RNA"})


def test_session_helper_initializes_saves_and_resets_cleanly() -> None:
    session: dict[str, object] = {}
    initial = get_workflow(session)
    assert initial.stage is WorkflowStage.CREATE

    document = Document(source_type=SourceType.TEXT, text="A stable source.")
    generating = begin_generation(
        initial,
        document=document,
        configuration={"difficulty": "Easy"},
        seed=3,
    )
    save_workflow(session, generating)

    restored = get_workflow(session)
    assert restored.source_identity is not None
    assert restored.quiz_configuration == {"difficulty": "Easy"}
    assert restored.generation_seed == 3

    reset = reset_workflow()
    assert reset.stage is WorkflowStage.CREATE
    assert reset.quiz is None
    assert reset.answers == {}
    assert reset.result is None
