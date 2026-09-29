"""Streamlit entry point for QuestLLM."""

import streamlit as st

from questllm.chunking import TextChunk, chunk_sentences
from questllm.config import APP_DESCRIPTION, APP_NAME
from questllm.document import Document
from questllm.exceptions import QuestLLMError
from questllm.ingestion import ingest_pdf, ingest_text
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
            document = ingest_text(pasted_text)
            sentences = preprocess_document(document)
            show_document_summary(document)
            show_chunk_summary(sentences, chunk_sentences(sentences))
        except QuestLLMError as error:
            st.error(str(error))
else:
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process Content", type="primary"):
        if uploaded_pdf is None:
            st.error("Upload a PDF before processing content.")
        else:
            try:
                document = ingest_pdf(uploaded_pdf, filename=uploaded_pdf.name)
                sentences = preprocess_document(document)
                show_document_summary(document)
                show_chunk_summary(sentences, chunk_sentences(sentences))
            except QuestLLMError as error:
                st.error(str(error))

st.caption("Quiz generation is intentionally not part of this phase.")
