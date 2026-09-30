"""Streamlit entry point for QuestLLM's create, attempt, and review workflow."""

from html import escape

import streamlit as st

from questllm.candidates import extract_and_rank_candidates
from questllm.chunking import chunk_sentences, make_tokenizer_estimator
from questllm.config import (
    APP_NAME,
    CHUNK_OVERLAP_SIZE,
    MAXIMUM_PDF_FILE_SIZE_BYTES,
    MAXIMUM_PDF_PAGE_COUNT,
    MAXIMUM_QUIZ_QUESTION_COUNT,
    MAXIMUM_SOURCE_TEXT_CHARACTERS,
    MINIMUM_QUIZ_QUESTION_COUNT,
    MODEL_CONTEXT_TOKEN_BUDGET,
)
from questllm.document import Document
from questllm.exceptions import NltkResourceError, QuestLLMError, ResourceLimitError
from questllm.generation import T5QuestionGenerator
from questllm.ingestion import (
    TopicSearchResult,
    WikipediaClient,
    ingest_pdf,
    ingest_text,
)
from questllm.model_loader import load_question_generation_model
from questllm.nltk_resources import initialize_runtime_resources
from questllm.preprocessing import preprocess_document
from questllm.questions import Difficulty, Question, QuestionType
from questllm.quiz_assembly import QuizGenerationPipeline
from questllm.scoring import AnswerStatus, AnswerValue, QuestionScore
from questllm.topic_state import (
    get_topic_state,
    reset_topic_state,
    save_topic_state,
    select_topic_result,
    store_search_results,
    store_topic_document,
    update_topic_query,
)
from questllm.workflow import (
    QuizWorkflow,
    WorkflowStage,
    begin_generation,
    complete_generation,
    get_workflow,
    reset_workflow,
    save_workflow,
    submit_quiz,
)

st.set_page_config(page_title=f"{APP_NAME} | Quiz Generator", page_icon="🧭", layout="centered")

st.markdown(
    """
    <style>
        .stMainBlockContainer {
            max-width: 780px;
            padding-top: 3.25rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, [data-testid="stMetricValue"] {
            letter-spacing: -0.025em;
        }

        h1 {
            font-size: clamp(2.25rem, 6vw, 3.25rem) !important;
            line-height: 1.05 !important;
            margin-bottom: 0.45rem !important;
        }

        [data-testid="stCaptionContainer"] {
            opacity: 0.68;
        }

        .quest-brand {
            margin-bottom: 2.75rem;
        }

        .quest-brand__name {
            color: inherit;
            font-size: clamp(2.35rem, 7vw, 3.5rem);
            font-weight: 760;
            letter-spacing: -0.055em;
            line-height: 1;
        }

        .quest-brand__tagline {
            font-size: 1rem;
            margin-top: 0.65rem;
            opacity: 0.68;
        }

        .quest-section-label {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            margin: 2.2rem 0 0.7rem;
            opacity: 0.72;
            text-transform: uppercase;
        }

        .quest-question-meta {
            font-size: 0.73rem;
            font-weight: 650;
            letter-spacing: 0.09em;
            margin-top: 1.7rem;
            opacity: 0.62;
            text-transform: uppercase;
        }

        .quest-question-prompt {
            color: inherit;
            font-size: 1.08rem;
            font-weight: 620;
            line-height: 1.5;
            margin: 0.35rem 0 0.8rem;
        }

        .quest-question-divider {
            border-top: 1px solid color-mix(in srgb, currentColor 12%, transparent);
            margin-top: 1.75rem;
        }

        .quest-status {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .quest-status--correct {
            color: #4f9f73;
        }

        .quest-status--incorrect {
            color: #bd6666;
        }

        .quest-status--unanswered {
            color: #a9843e;
        }

        div[data-testid="stTextInputRootElement"],
        div[data-testid="stTextAreaRootElement"],
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInputContainer"] {
            border-color: color-mix(in srgb, currentColor 18%, transparent) !important;
            border-radius: 0.45rem !important;
            box-shadow: none !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: color-mix(in srgb, currentColor 3%, transparent);
            border: 1px solid color-mix(in srgb, currentColor 17%, transparent);
            border-radius: 0.45rem;
        }

        div[data-testid="stAlertContainer"] {
            border-radius: 0.45rem;
        }

        div[data-testid="stMetric"] {
            background: transparent;
            border: 0;
            padding: 0.2rem 0;
        }

        div[data-testid="stMetricLabel"] {
            opacity: 0.68;
        }

        div[data-testid="stExpander"] {
            background: transparent;
            border-color: color-mix(in srgb, currentColor 14%, transparent);
            border-radius: 0.45rem;
            box-shadow: none;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stLinkButton > a {
            min-height: 2.75rem;
            border-radius: 0.45rem;
            box-shadow: none !important;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.055em;
        }

        :is(
            button[kind="primary"],
            button[kind="primaryFormSubmit"],
            button[data-testid="stBaseButton-primaryFormSubmit"]
        ) {
            background: currentColor !important;
            border: 0 !important;
            color: inherit !important;
        }

        :is(
            button[kind="primary"],
            button[kind="primaryFormSubmit"],
            button[data-testid="stBaseButton-primaryFormSubmit"]
        ) p {
            color: inherit !important;
            filter: invert(1) brightness(2);
        }

        :is(
            button[kind="primary"],
            button[kind="primaryFormSubmit"],
            button[data-testid="stBaseButton-primaryFormSubmit"]
        ):hover {
            background: currentColor !important;
            opacity: 0.88;
        }

        :is(
            button[kind="primary"],
            button[kind="primaryFormSubmit"],
            button[data-testid="stBaseButton-primaryFormSubmit"]
        ):focus-visible {
            outline: 2px solid currentColor;
            outline-offset: 2px;
        }

        button[kind="secondary"] {
            background: color-mix(in srgb, currentColor 5%, transparent) !important;
            border-color: color-mix(in srgb, currentColor 18%, transparent) !important;
            color: inherit !important;
        }

        button[kind="secondary"]:hover {
            background: color-mix(in srgb, currentColor 9%, transparent) !important;
            border-color: color-mix(in srgb, currentColor 26%, transparent) !important;
        }

        label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
            background: currentColor !important;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        @media (max-width: 640px) {
            .stMainBlockContainer {
                padding-top: 2rem;
            }

            .quest-brand {
                margin-bottom: 2.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_brand(*, tagline: bool = True) -> None:
    """Render the restrained, reusable product masthead."""

    subtitle = (
        '<div class="quest-brand__tagline">'
        "Turn text, PDFs, or topics into an intelligent quiz."
        "</div>"
        if tagline
        else ""
    )
    st.markdown(
        f'<div class="quest-brand"><div class="quest-brand__name">QUESTLLM</div>{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    """Render one consistent small-caps section marker."""

    st.markdown(f'<div class="quest-section-label">{escape(label)}</div>', unsafe_allow_html=True)


def show_source_summary(
    document: Document,
    *,
    sentence_count: int,
    chunk_count: int,
    candidate_count: int,
) -> None:
    """Confirm a prepared source while keeping implementation details optional."""

    st.success("Source ready.")
    source_label = {
        "text": "Pasted text",
        "pdf": "PDF upload",
        "topic": "Wikipedia article",
    }[document.source_type.value]
    with st.expander("Processing details"):
        details = st.columns(3)
        details[0].metric("Source", source_label)
        details[1].metric("Characters", f"{document.character_count:,}")
        details[2].metric("Pages", document.page_count if document.page_count else "—")
        st.caption(
            f"{sentence_count} sentences · {chunk_count} chunks · "
            f"{candidate_count} candidate concepts"
        )
        st.text_area(
            "Source preview",
            document.text[:800],
            height=140,
            disabled=True,
            label_visibility="collapsed",
        )


@st.cache_resource(show_spinner=False)
def load_cached_question_generator() -> T5QuestionGenerator:
    """Cache the heavyweight local model only after a user explicitly requests generation."""

    return T5QuestionGenerator(load_question_generation_model())


@st.cache_resource(show_spinner=False)
def initialize_cached_nltk_resources() -> None:
    """Prepare NLTK data once per app process and reuse it across Streamlit reruns."""

    initialize_runtime_resources()


def process_document(document: Document) -> None:
    """Run downstream preparation once and retain it only in create state."""

    try:
        sentences = preprocess_document(document)
        chunks = chunk_sentences(sentences)
        candidates = extract_and_rank_candidates(sentences)
    except MemoryError as error:
        raise ResourceLimitError(
            "QuestLLM ran out of memory while processing this source. Try a smaller document."
        ) from error
    st.session_state["processed_content"] = (document, sentences, chunks, candidates)


@st.cache_data(ttl=600, show_spinner=False)
def search_cached_wikipedia(query: str) -> tuple[TopicSearchResult, ...]:
    """Cache successful topic searches briefly without coupling ingestion to Streamlit."""

    return WikipediaClient().search(query)


@st.cache_data(ttl=3600, show_spinner=False)
def retrieve_cached_wikipedia_article(
    page_id: int,
    title: str,
    description: str,
    article_url: str,
    requested_topic: str,
) -> Document:
    """Cache successful article retrievals by stable page ID for a reasonable period."""

    result = TopicSearchResult(
        page_id=page_id,
        title=title,
        description=description,
        article_url=article_url,
    )
    return WikipediaClient().retrieve_article(result, requested_topic=requested_topic)


def clear_processed_content() -> None:
    """Remove content derived from a source that is no longer active."""

    st.session_state.pop("processed_content", None)


def _choice_text(question: Question, choice_id: str | None) -> str:
    if choice_id is None:
        return "No answer"
    return next(
        (choice.text for choice in question.choices if choice.id == choice_id),
        "Invalid choice",
    )


def _review_answer(question: Question, result: QuestionScore) -> str:
    if result.status is AnswerStatus.UNANSWERED:
        return "No answer"
    if question.question_type is QuestionType.MULTIPLE_CHOICE:
        choice_id = result.user_answer if isinstance(result.user_answer, str) else None
        return _choice_text(question, choice_id)
    return str(result.user_answer)


def render_attempt(workflow: QuizWorkflow) -> None:
    """Render answer-only controls; answers, provenance, and correctness stay hidden here."""

    quiz = workflow.quiz
    if quiz is None:
        st.error("No quiz is available. Start over and generate a new quiz.")
        return
    render_brand(tagline=False)
    st.caption(f"{quiz.actual_question_count} questions · {quiz.difficulty.value} difficulty")
    for warning in quiz.warnings:
        st.warning(warning)

    with st.form(f"quiz-attempt-{quiz.id}"):
        answers: dict[str, AnswerValue] = {}
        for index, question in enumerate(quiz.questions, start=1):
            if index > 1:
                st.markdown('<div class="quest-question-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="quest-question-meta">'
                f"Question {index} of {quiz.actual_question_count} · "
                f"{escape(question.question_type.value)}"
                "</div>"
                f'<div class="quest-question-prompt">{escape(question.prompt)}</div>',
                unsafe_allow_html=True,
            )
            answer_key = f"answer-{quiz.id}-{question.id}"
            saved_answer = workflow.answers.get(question.id)
            if question.question_type is QuestionType.MULTIPLE_CHOICE:
                choice_ids = [choice.id for choice in question.choices]
                selected_index = (
                    choice_ids.index(saved_answer) if saved_answer in choice_ids else None
                )
                answers[question.id] = st.radio(
                    "Choose one answer",
                    options=choice_ids,
                    index=selected_index,
                    format_func=lambda choice_id, item=question: _choice_text(item, choice_id),
                    key=answer_key,
                    label_visibility="collapsed",
                )
            elif question.question_type is QuestionType.TRUE_FALSE:
                selected_index = (
                    (0 if saved_answer else 1) if isinstance(saved_answer, bool) else None
                )
                answers[question.id] = st.radio(
                    "Choose True or False",
                    options=(True, False),
                    index=selected_index,
                    format_func=lambda value: "True" if value else "False",
                    key=answer_key,
                    label_visibility="collapsed",
                )
            elif question.question_type is QuestionType.FILL_IN_THE_BLANK:
                answers[question.id] = st.text_input(
                    "Your answer",
                    value=saved_answer if isinstance(saved_answer, str) else "",
                    key=answer_key,
                    placeholder="Type your answer",
                    label_visibility="collapsed",
                )
            else:
                answers[question.id] = st.text_area(
                    "Your answer",
                    value=saved_answer if isinstance(saved_answer, str) else "",
                    key=answer_key,
                    height=90,
                    placeholder="Type your answer",
                    label_visibility="collapsed",
                )
        st.markdown('<div class="quest-question-divider"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "SUBMIT QUIZ",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            save_workflow(st.session_state, submit_quiz(workflow, answers))
            st.rerun()
        except QuestLLMError as error:
            st.error(str(error))


def render_review(workflow: QuizWorkflow) -> None:
    """Render score and review details only after frozen answers have been submitted."""

    if workflow.quiz is None or workflow.result is None:
        st.error("The submitted quiz result is unavailable. Start over to create a new quiz.")
        return
    quiz = workflow.quiz
    result = workflow.result
    st.title("RESULTS")
    headline_metrics = st.columns(2)
    headline_metrics[0].metric("Score", f"{result.correct_count} / {result.total_questions}")
    headline_metrics[1].metric("Percentage", f"{result.percentage}%")
    st.markdown('<div class="quest-question-divider"></div>', unsafe_allow_html=True)
    compact_metrics = st.columns(3)
    compact_metrics[0].metric("Correct", result.correct_count)
    compact_metrics[1].metric("Incorrect", result.incorrect_count)
    compact_metrics[2].metric("Unanswered", result.unanswered_count)

    with st.expander("Score details"):
        st.table(
            [
                {
                    "Type": question_type.value,
                    "Correct": type_score.correct,
                    "Total": type_score.total,
                    "Unanswered": type_score.unanswered,
                }
                for question_type, type_score in result.by_type.items()
            ]
        )

    results_by_id = {item.question_id: item for item in result.question_results}
    render_section_label("Review answers")
    for index, question in enumerate(quiz.questions, start=1):
        question_result = results_by_id[question.id]
        status_label = question_result.status.value.title()
        status_class = question_result.status.value.lower()
        prompt_preview = (
            question.prompt if len(question.prompt) <= 72 else question.prompt[:69] + "…"
        )
        with st.expander(f"{index:02d} · {status_label} · {prompt_preview}"):
            st.markdown(
                f'<div class="quest-status quest-status--{status_class}">'
                f"{escape(status_label)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="quest-question-prompt">{escape(question.prompt)}</div>',
                unsafe_allow_html=True,
            )
            if question.choices:
                st.caption("Choices")
                for choice in question.choices:
                    st.write(f"- {choice.text}")
            answer_columns = st.columns(2)
            answer_columns[0].caption("Your answer")
            answer_columns[0].write(_review_answer(question, question_result))
            answer_columns[1].caption("Correct answer")
            answer_columns[1].write(question.correct_answer)
            if (
                question.question_type is QuestionType.TRUE_FALSE
                and question.boolean_answer is False
            ):
                st.caption(f"Original source fact: {question.source_sentence}")
            source_parts = []
            if question.page_number is not None:
                source_parts.append(f"Page {question.page_number}")
            article_title = quiz.source_metadata.get("article_title")
            if source_parts:
                st.caption("Source · " + " · ".join(source_parts))
            article_url = quiz.source_metadata.get("article_url")
            if article_title and article_url:
                st.caption(f"Source article · [{article_title}]({article_url})")
            elif article_title:
                st.caption(f"Source article · {article_title}")
            st.caption(f"Excerpt · {question.source_excerpt}")


def _topic_result_label(result: TopicSearchResult) -> str:
    """Format a compact, readable label without exposing API internals."""

    return f"{result.title} — {result.description}" if result.description else result.title


def render_topic_input() -> None:
    """Render explicit Wikipedia search, selection, and processing actions."""

    state = get_topic_state(st.session_state)
    query = st.text_input(
        "Topic",
        value=state.query,
        key="topic_query_input",
        placeholder="Enter a topic",
        label_visibility="collapsed",
    )
    st.caption("Search Wikipedia, then choose one article to use as the quiz source.")
    updated_state = update_topic_query(state, query)
    if updated_state != state:
        state = updated_state
        save_topic_state(st.session_state, state)
        clear_processed_content()
        st.session_state.pop("topic_article_selection", None)

    search_button_type = "secondary" if state.results else "primary"
    if st.button("SEARCH WIKIPEDIA", type=search_button_type, use_container_width=True):
        try:
            results = search_cached_wikipedia(query)
            state = store_search_results(state, query=query, results=results)
            save_topic_state(st.session_state, state)
            st.session_state.pop("topic_article_selection", None)
        except QuestLLMError as error:
            st.error(str(error))

    if not state.results:
        return
    result_by_id = {result.page_id: result for result in state.results}
    selected_id = state.selected_result.page_id if state.selected_result else None
    selected_index = list(result_by_id).index(selected_id) if selected_id in result_by_id else None
    selected_page_id = st.selectbox(
        "Search results",
        options=tuple(result_by_id),
        index=selected_index,
        format_func=lambda page_id: _topic_result_label(result_by_id[page_id]),
        placeholder="Select an article",
        key="topic_article_selection",
        label_visibility="collapsed",
    )
    if selected_page_id is not None and selected_page_id != selected_id:
        state = select_topic_result(state, selected_page_id)
        save_topic_state(st.session_state, state)

    if state.selected_result is None:
        return
    st.caption(f"Selected: {state.selected_result.title}")
    article_button_type = "secondary" if st.session_state.get("processed_content") else "primary"
    if st.button("PROCESS ARTICLE", type=article_button_type, use_container_width=True):
        try:
            document = retrieve_cached_wikipedia_article(
                state.selected_result.page_id,
                state.selected_result.title,
                state.selected_result.description,
                state.selected_result.article_url,
                state.query,
            )
            state = store_topic_document(state, document)
            save_topic_state(st.session_state, state)
            process_document(document)
        except QuestLLMError as error:
            st.error(str(error))


def render_create(workflow: QuizWorkflow) -> None:
    """Render source/configuration controls only while no active quiz exists."""

    render_brand()
    render_section_label("Source")
    source_choice = st.radio(
        "Source type",
        ("Paste Text", "Upload PDF", "Enter Topic"),
        horizontal=True,
        key="source_mode",
        label_visibility="collapsed",
    )
    previous_source = st.session_state.get("active_source_mode")
    if previous_source is not None and previous_source != source_choice:
        clear_processed_content()
        save_topic_state(st.session_state, reset_topic_state())
        st.session_state.pop("topic_query_input", None)
        st.session_state.pop("topic_article_selection", None)
    st.session_state["active_source_mode"] = source_choice
    source_is_ready = bool(st.session_state.get("processed_content"))

    if source_choice == "Paste Text":
        pasted_text = st.text_area(
            "Text source",
            placeholder="Paste your learning material here…",
            height=210,
            label_visibility="collapsed",
        )
        st.caption("Use clear, factual text for the strongest questions.")
        st.caption(f"Deployment limit: {MAXIMUM_SOURCE_TEXT_CHARACTERS:,} characters.")
        process_button_type = "secondary" if source_is_ready else "primary"
        if st.button("PROCESS TEXT", type=process_button_type, use_container_width=True):
            try:
                process_document(ingest_text(pasted_text))
            except QuestLLMError as error:
                st.error(str(error))
    elif source_choice == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "PDF source",
            type=["pdf"],
            label_visibility="collapsed",
        )
        maximum_pdf_megabytes = MAXIMUM_PDF_FILE_SIZE_BYTES // (1024 * 1024)
        st.caption(
            "Text-based PDFs work best. Scanned images may not contain extractable text. "
            f"Limit: {maximum_pdf_megabytes} MB, {MAXIMUM_PDF_PAGE_COUNT} pages."
        )
        process_button_type = "secondary" if source_is_ready else "primary"
        if st.button("PROCESS PDF", type=process_button_type, use_container_width=True):
            if uploaded_pdf is None:
                st.error("Upload a PDF before processing content.")
            else:
                try:
                    process_document(ingest_pdf(uploaded_pdf, filename=uploaded_pdf.name))
                except QuestLLMError as error:
                    st.error(str(error))
    else:
        render_topic_input()

    processed_content = st.session_state.get("processed_content")
    if not processed_content:
        return
    document, sentences, chunks, candidates = processed_content
    show_source_summary(
        document,
        sentence_count=len(sentences),
        chunk_count=len(chunks),
        candidate_count=len(candidates),
    )
    if not candidates:
        st.warning(
            "QuestLLM could not find enough clear concepts in this source. "
            "Try a longer or more factual passage."
        )
        return

    render_section_label("Quiz settings")
    selected_type_labels = st.multiselect(
        "Question types",
        options=[question_type.value for question_type in QuestionType],
        default=[question_type.value for question_type in QuestionType],
    )
    selected_types = tuple(QuestionType(label) for label in selected_type_labels)
    setting_columns = st.columns(2)
    difficulty_label = setting_columns[0].selectbox(
        "Difficulty",
        options=[level.value for level in Difficulty],
    )
    difficulty = Difficulty(difficulty_label)
    requested_count = setting_columns[1].number_input(
        "Number of questions",
        min_value=MINIMUM_QUIZ_QUESTION_COUNT,
        max_value=MAXIMUM_QUIZ_QUESTION_COUNT,
        value=min(6, MAXIMUM_QUIZ_QUESTION_COUNT),
        step=1,
    )
    with st.expander("Advanced settings"):
        generation_seed = st.number_input(
            "Generation seed",
            min_value=0,
            value=17,
            step=1,
            help="Use the same seed to reproduce a quiz from the same source and settings.",
        )

    if st.button("GENERATE QUIZ", type="primary", use_container_width=True):
        if not selected_types:
            st.warning("Select at least one question type.")
            return
        configuration = {
            "difficulty": difficulty.value,
            "requested_count": int(requested_count),
            "question_types": ", ".join(question_type.value for question_type in selected_types),
        }
        try:
            generating_workflow = begin_generation(
                workflow,
                document=document,
                configuration=configuration,
                seed=int(generation_seed),
            )
            save_workflow(st.session_state, generating_workflow)
            needs_t5_stems = any(
                question_type in {QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER}
                for question_type in selected_types
            )
            generator = None
            quiz_chunks = chunks
            with st.spinner(
                "Building your quiz… the first model download is about 900 MB, and CPU "
                "generation can take several minutes."
            ):
                if needs_t5_stems:
                    generator = load_cached_question_generator()
                    tokenizer_estimator = make_tokenizer_estimator(generator.tokenizer)
                    quiz_chunks = chunk_sentences(
                        sentences,
                        target_size=MODEL_CONTEXT_TOKEN_BUDGET,
                        overlap_size=CHUNK_OVERLAP_SIZE,
                        estimator=tokenizer_estimator,
                    )
                quiz = QuizGenerationPipeline(generator).build_from_processed(
                    document,
                    chunks=quiz_chunks,
                    candidates=candidates,
                    requested_count=int(requested_count),
                    question_types=selected_types,
                    difficulty=difficulty,
                    seed=int(generation_seed),
                )
            save_workflow(st.session_state, complete_generation(generating_workflow, quiz))
            st.rerun()
        except QuestLLMError as error:
            save_workflow(st.session_state, reset_workflow())
            st.error(str(error))
        except ValueError:
            save_workflow(st.session_state, reset_workflow())
            st.error(
                "Quiz generation could not be completed with these settings. "
                "Try fewer questions or a different source."
            )


try:
    with st.spinner("Preparing language resources…"):
        initialize_cached_nltk_resources()
except NltkResourceError:
    render_brand()
    st.error(
        "QuestLLM could not prepare its language resources. Refresh to retry. If the problem "
        "continues, ask the app owner to check the deployment network and logs."
    )
    st.stop()


workflow = get_workflow(st.session_state)
if workflow.stage is WorkflowStage.CREATE:
    render_create(workflow)
else:
    if workflow.stage is WorkflowStage.ATTEMPT:
        render_attempt(workflow)
    elif workflow.stage is WorkflowStage.REVIEW:
        render_review(workflow)
    else:
        st.info("Generating your quiz. Please wait for the current request to finish.")
    if st.button("CREATE NEW QUIZ", use_container_width=True):
        save_workflow(st.session_state, reset_workflow())
        clear_processed_content()
        save_topic_state(st.session_state, reset_topic_state())
        st.session_state.pop("topic_query_input", None)
        st.session_state.pop("topic_article_selection", None)
        st.rerun()
