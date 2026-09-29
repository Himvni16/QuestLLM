"""Streamlit entry point for QuestLLM."""

import streamlit as st

from questllm.candidates import CandidateAnswer, extract_and_rank_candidates
from questllm.chunking import TextChunk, chunk_sentences, make_tokenizer_estimator
from questllm.config import (
    APP_DESCRIPTION,
    APP_NAME,
    CHUNK_OVERLAP_SIZE,
    MAXIMUM_QUIZ_QUESTION_COUNT,
    MINIMUM_QUIZ_QUESTION_COUNT,
    MODEL_CONTEXT_TOKEN_BUDGET,
)
from questllm.document import Document
from questllm.exceptions import QuestLLMError
from questllm.generation import QuestionPreviewService, T5QuestionGenerator
from questllm.ingestion import ingest_pdf, ingest_text
from questllm.model_loader import load_question_generation_model
from questllm.preprocessing import ProcessedSentence, preprocess_document
from questllm.questions import Difficulty, QuestionType
from questllm.quiz import Quiz
from questllm.quiz_assembly import QuizGenerationPipeline

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="centered")

st.title(APP_NAME)
st.write(APP_DESCRIPTION)


def show_document_summary(document: Document) -> None:
    """Render a concise summary of successfully ingested content."""

    st.success("Content processed successfully.")
    summary = {"Source type": document.source_type.value, "Characters": document.character_count}
    if document.page_count:
        summary["Pages"] = document.page_count
    st.json(summary)
    st.text_area("Processed-text preview", document.text[:800], height=180, disabled=True)


def show_chunk_summary(
    sentences: tuple[ProcessedSentence, ...], chunks: tuple[TextChunk, ...]
) -> None:
    """Render the sentence and chunk information relevant to this processing phase."""

    st.caption(f"{len(sentences)} usable sentences · {len(chunks)} text chunks")
    for chunk in chunks:
        label = f"Chunk {chunk.index + 1} · approximately {chunk.estimated_token_count} words"
        with st.expander(label):
            st.write(chunk.text)


def show_candidate_preview(candidates: tuple[CandidateAnswer, ...]) -> None:
    """Render a compact, intermediate preview of the strongest answer candidates."""

    st.subheader("Candidate-answer preview")
    if not candidates:
        st.info("No strong answer candidates were found in this material.")
        return

    st.caption(f"{len(candidates)} ranked candidates selected across the source material")
    preview = []
    for candidate in candidates:
        item = {
            "Candidate": candidate.text,
            "Type": candidate.candidate_type.value,
            "Score": round(candidate.importance_score, 2),
        }
        if candidate.page_number is not None:
            item["Page"] = candidate.page_number
        preview.append(item)
    st.table(preview)


@st.cache_resource(show_spinner=False)
def load_cached_question_generator() -> T5QuestionGenerator:
    """Cache the heavyweight local model at the UI boundary after a user requests it."""

    return T5QuestionGenerator(load_question_generation_model())


def show_question_preview() -> None:
    """Render development-only question stems with their expected answers and source references."""

    results = st.session_state.get("question_preview", ())
    if not results:
        return
    st.subheader("Question-stem preview")
    for index, result in enumerate(results, start=1):
        st.markdown(f"**{index}. {result.question}**")
        st.write(f"Expected answer: `{result.candidate.text}`")
        if result.page_number is not None:
            st.caption(f"Source page {result.page_number}")
        with st.expander("Source excerpt"):
            st.write(result.source_sentence)


def show_quiz_preview(quiz: Quiz | None) -> None:
    """Render a development-only assembled quiz with transparent answers and warnings."""

    if quiz is None:
        return
    st.subheader("Quiz preview")
    st.caption(
        f"Requested {quiz.requested_question_count} · Generated {quiz.actual_question_count} · "
        f"Seed {quiz.generation_seed}"
    )
    for warning in quiz.warnings:
        st.warning(warning)
    st.caption("Development preview: correct answers and source excerpts are visible.")
    for index, question in enumerate(quiz.questions, start=1):
        st.markdown(f"**{index}. {question.question_type.value} · {question.difficulty.value}**")
        st.write(question.prompt)
        if question.choices:
            for choice in question.choices:
                st.write(f"- {choice.id[:8]}: {choice.text}")
        st.write(f"Correct answer: `{question.correct_answer}`")
        if question.page_number is not None:
            st.caption(f"Source page {question.page_number}")
        with st.expander("Source excerpt"):
            st.write(question.source_excerpt)


def process_document(document: Document) -> None:
    """Run the completed pre-generation workflow and retain it for the preview action."""

    sentences = preprocess_document(document)
    chunks = chunk_sentences(sentences)
    candidates = extract_and_rank_candidates(sentences)
    st.session_state["processed_content"] = (document, sentences, chunks, candidates)
    st.session_state.pop("question_preview", None)
    st.session_state.pop("quiz_preview", None)


source_choice = st.radio("Choose content source", ("Paste Text", "Upload PDF"), horizontal=True)

if source_choice == "Paste Text":
    pasted_text = st.text_area(
        "Paste text",
        placeholder=(
            "Paste learning material here. QuestLLM will prepare it for a later quiz phase."
        ),
        height=220,
    )
    if st.button("Process Content", type="primary"):
        try:
            process_document(ingest_text(pasted_text))
        except QuestLLMError as error:
            st.error(str(error))
else:
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process Content", type="primary"):
        if uploaded_pdf is None:
            st.error("Upload a PDF before processing content.")
        else:
            try:
                process_document(ingest_pdf(uploaded_pdf, filename=uploaded_pdf.name))
            except QuestLLMError as error:
                st.error(str(error))

processed_content = st.session_state.get("processed_content")
if processed_content:
    document, sentences, chunks, candidates = processed_content
    show_document_summary(document)
    show_chunk_summary(sentences, chunks)
    show_candidate_preview(candidates)

    if st.button("Generate Question Preview"):
        try:
            with st.spinner("Loading the local T5 model and generating question stems..."):
                generator = load_cached_question_generator()
                tokenizer_chunks = chunk_sentences(
                    sentences,
                    target_size=MODEL_CONTEXT_TOKEN_BUDGET,
                    overlap_size=CHUNK_OVERLAP_SIZE,
                    estimator=make_tokenizer_estimator(generator.tokenizer),
                )
                service = QuestionPreviewService(
                    generator,
                    estimator=make_tokenizer_estimator(generator.tokenizer),
                )
                st.session_state["question_preview"] = service.generate_preview(
                    tokenizer_chunks,
                    candidates,
                )
            st.info(f"Using local inference on {generator.device.upper()}.")
            if not st.session_state["question_preview"]:
                st.warning(
                    "No safe question stems could be generated from the selected candidates."
                )
        except QuestLLMError as error:
            st.error(str(error))

    show_question_preview()

    selected_type_labels = st.multiselect(
        "Question types for quiz preview",
        options=[question_type.value for question_type in QuestionType],
        default=[question_type.value for question_type in QuestionType],
    )
    selected_types = tuple(QuestionType(label) for label in selected_type_labels)
    difficulty_label = st.selectbox("Difficulty", options=[level.value for level in Difficulty])
    difficulty = Difficulty(difficulty_label)
    requested_count = st.number_input(
        "Requested question count",
        min_value=MINIMUM_QUIZ_QUESTION_COUNT,
        max_value=MAXIMUM_QUIZ_QUESTION_COUNT,
        value=min(6, MAXIMUM_QUIZ_QUESTION_COUNT),
        step=1,
    )
    generation_seed = st.number_input("Generation seed", min_value=0, value=17, step=1)

    if st.button("Generate Quiz Preview"):
        if not selected_types:
            st.warning("Select at least one question type for the quiz preview.")
        else:
            try:
                needs_t5_stems = any(
                    question_type in {QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER}
                    for question_type in selected_types
                )
                generator = None
                quiz_chunks = chunks
                if needs_t5_stems:
                    with st.spinner("Loading the local T5 model and generating question stems..."):
                        generator = load_cached_question_generator()
                        tokenizer_estimator = make_tokenizer_estimator(generator.tokenizer)
                        tokenizer_chunks = chunk_sentences(
                            sentences,
                            target_size=MODEL_CONTEXT_TOKEN_BUDGET,
                            overlap_size=CHUNK_OVERLAP_SIZE,
                            estimator=tokenizer_estimator,
                        )
                        quiz_chunks = tokenizer_chunks
                st.session_state["quiz_preview"] = QuizGenerationPipeline(
                    generator
                ).build_from_processed(
                    document,
                    chunks=quiz_chunks,
                    candidates=candidates,
                    requested_count=int(requested_count),
                    question_types=selected_types,
                    difficulty=difficulty,
                    seed=int(generation_seed),
                    pre_generated_stems=st.session_state.get("question_preview", ()),
                )
            except QuestLLMError as error:
                st.error(str(error))
            except ValueError as error:
                st.error(str(error))

    show_quiz_preview(st.session_state.get("quiz_preview"))

st.caption(
    "Question previews are for development only; quiz-taking and scoring are not available yet."
)
