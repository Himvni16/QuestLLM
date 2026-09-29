"""Immutable, Streamlit-independent workflow and session-state helpers for quiz attempts."""

import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from questllm.document import Document
from questllm.exceptions import WorkflowError
from questllm.quiz import Quiz
from questllm.scoring import AnswerValue, QuizScore, score_quiz

WORKFLOW_SESSION_KEY = "questllm_workflow"


class WorkflowStage(StrEnum):
    """The only supported states for the create-to-review quiz lifecycle."""

    CREATE = "CREATE"
    GENERATING = "GENERATING"
    ATTEMPT = "ATTEMPT"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class QuizWorkflow:
    """All active-quiz state kept together so Streamlit reruns cannot mutate it implicitly."""

    stage: WorkflowStage = WorkflowStage.CREATE
    quiz: Quiz | None = None
    source_identity: str | None = None
    quiz_configuration: Mapping[str, str | int] = field(default_factory=dict)
    answers: Mapping[str, AnswerValue] = field(default_factory=dict)
    result: QuizScore | None = None
    generation_seed: int | None = None


def source_identity(document: Document) -> str:
    """Return a stable content hash that binds a generated quiz to its processed source."""

    digest = hashlib.sha256()
    digest.update(document.source_type.value.encode())
    digest.update(b"\0")
    digest.update(document.text.encode("utf-8"))
    return digest.hexdigest()


def begin_generation(
    workflow: QuizWorkflow,
    *,
    document: Document,
    configuration: Mapping[str, str | int],
    seed: int,
) -> QuizWorkflow:
    """Freeze source/configuration identity before an explicit generation request begins."""

    if workflow.stage is not WorkflowStage.CREATE:
        raise WorkflowError("Start over before generating a new quiz.")
    return replace(
        workflow,
        stage=WorkflowStage.GENERATING,
        source_identity=source_identity(document),
        quiz_configuration=dict(configuration),
        generation_seed=seed,
    )


def complete_generation(workflow: QuizWorkflow, quiz: Quiz) -> QuizWorkflow:
    """Store the generated quiz exactly once and enter the answer-hidden attempt stage."""

    if workflow.stage is not WorkflowStage.GENERATING:
        raise WorkflowError("A quiz can only be stored while generation is in progress.")
    return replace(workflow, stage=WorkflowStage.ATTEMPT, quiz=quiz)


def record_answers(workflow: QuizWorkflow, answers: Mapping[str, AnswerValue]) -> QuizWorkflow:
    """Return an updated attempt state; answers cannot change once the quiz is submitted."""

    if workflow.stage is not WorkflowStage.ATTEMPT:
        raise WorkflowError("Answers can only be recorded during an active quiz attempt.")
    if workflow.quiz is None:
        raise WorkflowError("No active quiz is available for answers.")
    valid_ids = {question.id for question in workflow.quiz.questions}
    filtered = {
        question_id: answer
        for question_id, answer in answers.items()
        if question_id in valid_ids
    }
    return replace(workflow, answers=filtered)


def submit_quiz(workflow: QuizWorkflow, answers: Mapping[str, AnswerValue]) -> QuizWorkflow:
    """Freeze submitted answers, score once, and transition directly to review."""

    attempted = record_answers(workflow, answers)
    if attempted.quiz is None:
        raise WorkflowError("No active quiz is available for submission.")
    return replace(
        attempted,
        stage=WorkflowStage.REVIEW,
        result=score_quiz(attempted.quiz, attempted.answers),
    )


def reset_workflow() -> QuizWorkflow:
    """Create a clean state with no stale source, answers, quiz, or result."""

    return QuizWorkflow()


def get_workflow(session_state: MutableMapping[str, object]) -> QuizWorkflow:
    """Read or initialize one workflow object in a generic session-state mapping."""

    workflow = session_state.get(WORKFLOW_SESSION_KEY)
    if isinstance(workflow, QuizWorkflow):
        return workflow
    initialized = reset_workflow()
    session_state[WORKFLOW_SESSION_KEY] = initialized
    return initialized


def save_workflow(session_state: MutableMapping[str, object], workflow: QuizWorkflow) -> None:
    """Persist the complete immutable workflow as one stable session-state value."""

    session_state[WORKFLOW_SESSION_KEY] = workflow
