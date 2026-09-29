"""Streamlit entry point for QuestLLM."""

import streamlit as st

from questllm.candidates import CandidateAnswer, extract_and_rank_candidates
from questllm.chunking import TextChunk, chunk_sentences, make_tokenizer_estimator
from questllm.config import (
    APP_DESCRIPTION,
    APP_NAME,
    CHUNK_OVERLAP_SIZE,
    MODEL_CONTEXT_TOKEN_BUDGET,
)
from questllm.document import Document
from questllm.exceptions import QuestLLMError
from questllm.generation import QuestionPreviewService, T5QuestionGenerator
from questllm.ingestion import ingest_pdf, ingest_text
from questllm.model_loader import load_question_generation_model
from questllm.preprocessing import ProcessedSentence, preprocess_document

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


def process_document(document: Document) -> None:
    """Run the completed pre-generation workflow and retain it for the preview action."""

    sentences = preprocess_document(document)
    chunks = chunk_sentences(sentences)
    candidates = extract_and_rank_candidates(sentences)
    st.session_state["processed_content"] = (document, sentences, chunks, candidates)
    st.session_state.pop("question_preview", None)


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

st.caption(
    "Question stems are a development preview; quiz-taking and scoring are not available yet."
)
