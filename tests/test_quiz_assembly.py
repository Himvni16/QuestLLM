"""Offline tests for quota allocation, coverage, partial results, and orchestration."""

from dataclasses import dataclass

from questllm.candidates import CandidateAnswer, CandidateType
from questllm.chunking import TextChunk
from questllm.document import Document, SourceType
from questllm.generation import FocusedContext, QuestionGenerationResult, QuestionGenerationSettings
from questllm.preprocessing import ProcessedSentence
from questllm.questions import Difficulty, QuestionType
from questllm.quiz_assembly import (
    QuizAssemblyService,
    QuizGenerationPipeline,
    allocate_question_quotas,
)


def _candidate(index: int, *, page: int | None = None) -> CandidateAnswer:
    name = ("chlorophyll", "mitochondria", "ribosome", "nucleus", "DNA", "glucose")[index]
    facts = (
        "absorbs light energy during photosynthesis in plant cells.",
        "releases energy from nutrients for cell activities.",
        "builds proteins by joining amino acids together.",
        "stores genetic material and coordinates cell activities.",
        "stores hereditary instructions for living organisms.",
        "provides chemical energy for cellular processes.",
    )
    return CandidateAnswer(
        text=name,
        normalized_text=name.casefold(),
        source_sentence=f"{name.title()} {facts[index]}",
        sentence_index=index,
        paragraph_index=index,
        page_number=page if page is not None else index // 2 + 1,
        importance_score=0.9 - index / 100,
        candidate_type=CandidateType.GENERAL_CONCEPT,
    )


def _stem(candidate: CandidateAnswer) -> QuestionGenerationResult:
    return QuestionGenerationResult(
        question="Which cell component has an important role?",
        candidate=candidate,
        source_context=candidate.source_sentence,
        source_sentence=candidate.source_sentence,
        sentence_index=candidate.sentence_index,
        paragraph_index=candidate.paragraph_index,
        page_number=candidate.page_number,
        model_id="fake/t5",
    )


def _chunks(candidates: tuple[CandidateAnswer, ...]) -> tuple[TextChunk, ...]:
    sentences = tuple(
        ProcessedSentence(
            index=candidate.sentence_index,
            text=candidate.source_sentence,
            page_number=candidate.page_number,
            paragraph_index=candidate.paragraph_index,
        )
        for candidate in candidates
    )
    return (
        TextChunk(
            index=0,
            text=" ".join(sentence.text for sentence in sentences),
            sentences=sentences,
            page_numbers=tuple(dict.fromkeys(sentence.page_number for sentence in sentences)),
            estimated_token_count=80,
        ),
    )


def test_quota_allocation_is_even_and_respects_selected_order() -> None:
    quotas = allocate_question_quotas(
        5,
        (QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE, QuestionType.FILL_IN_THE_BLANK),
    )

    assert list(quotas.values()) == [2, 2, 1]


def test_assembly_backfills_available_types_and_returns_partial_warning_when_needed() -> None:
    candidates = tuple(_candidate(index) for index in range(4))
    assembler = QuizAssemblyService(candidates, (), source_metadata={"source_type": "text"})

    quiz = assembler.assemble(
        requested_count=4,
        question_types=(QuestionType.MULTIPLE_CHOICE, QuestionType.FILL_IN_THE_BLANK),
        difficulty=Difficulty.EASY,
        seed=7,
    )

    assert quiz.actual_question_count == 4
    assert all(
        question.question_type is QuestionType.FILL_IN_THE_BLANK for question in quiz.questions
    )

    partial = assembler.assemble(
        requested_count=6,
        question_types=(QuestionType.FILL_IN_THE_BLANK,),
        difficulty=Difficulty.EASY,
        seed=7,
    )
    assert partial.actual_question_count == 4
    assert "Requested 6 questions" in partial.warnings[0]


def test_assembly_prefers_page_coverage_and_is_seed_deterministic() -> None:
    candidates = tuple(_candidate(index) for index in range(4))
    assembler = QuizAssemblyService(candidates, (), source_metadata={"source_type": "pdf"})

    first = assembler.assemble(
        requested_count=3,
        question_types=(QuestionType.FILL_IN_THE_BLANK,),
        difficulty=Difficulty.MEDIUM,
        seed=99,
    )
    second = assembler.assemble(
        requested_count=3,
        question_types=(QuestionType.FILL_IN_THE_BLANK,),
        difficulty=Difficulty.MEDIUM,
        seed=99,
    )

    assert first == second
    assert {question.page_number for question in first.questions} == {1, 2}


@dataclass
class _FakeGenerator:
    tokenizer: object = object()
    model_id: str = "fake/t5"
    device: str = "cpu"

    def generate(
        self,
        candidate: CandidateAnswer,
        context: FocusedContext,
        _settings: QuestionGenerationSettings,
    ) -> QuestionGenerationResult:
        prompts = (
            "Which pigment absorbs light energy during photosynthesis?",
            "Which organelle releases energy from nutrients?",
            "Which structure builds proteins from amino acids?",
            "Which structure stores genetic material in a cell?",
            "Which molecule stores hereditary instructions?",
            "Which substance provides chemical energy for cells?",
        )
        return QuestionGenerationResult(
            question=prompts[candidate.sentence_index],
            candidate=candidate,
            source_context=context.text,
            source_sentence=context.source_sentence,
            sentence_index=candidate.sentence_index,
            paragraph_index=candidate.paragraph_index,
            page_number=candidate.page_number,
            model_id=self.model_id,
        )


def test_pipeline_builds_quiz_from_document_with_fake_t5_generation() -> None:
    candidates = tuple(_candidate(index) for index in range(3))
    document = Document(
        source_type=SourceType.TEXT,
        text="Source material",
        metadata={"title": "Cells"},
    )
    chunks = _chunks(candidates)
    sentences = chunks[0].sentences
    pipeline = QuizGenerationPipeline(
        _FakeGenerator(),
        document_preprocessor=lambda _document: sentences,
        sentence_chunker=lambda _sentences: chunks,
        candidate_extractor=lambda _sentences: candidates,
    )

    quiz = pipeline.build_from_document(
        document,
        requested_count=2,
        question_types=(QuestionType.SHORT_ANSWER,),
        difficulty=Difficulty.EASY,
        seed=3,
    )

    assert quiz.actual_question_count == 2
    assert all(question.question_type is QuestionType.SHORT_ANSWER for question in quiz.questions)
    assert quiz.source_metadata["title"] == "Cells"
