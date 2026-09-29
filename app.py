"""Streamlit entry point for QuestLLM's create, attempt, and review workflow."""

import streamlit as st

from questllm.candidates import extract_and_rank_candidates
from questllm.chunking import chunk_sentences, make_tokenizer_estimator
from questllm.config import (
    APP_NAME,
    CHUNK_OVERLAP_SIZE,
    MAXIMUM_QUIZ_QUESTION_COUNT,
    MINIMUM_QUIZ_QUESTION_COUNT,
    MODEL_CONTEXT_TOKEN_BUDGET,
)
from questllm.document import Document
from questllm.exceptions import QuestLLMError
from questllm.generation import T5QuestionGenerator
from questllm.ingestion import (
    TopicSearchResult,
    WikipediaClient,
    ingest_pdf,
    ingest_text,
)
from questllm.model_loader import load_question_generation_model
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

st.title(APP_NAME)
st.caption("Turn your study material into a grounded practice quiz.")


def show_source_summary(
    document: Document,
    *,
    sentence_count: int,
    chunk_count: int,
    candidate_count: int,
) -> None:
    """Confirm a prepared source while keeping implementation details optional."""

    st.success("Your source is ready. Choose quiz settings below.")
    source_label = {
        "text": "Pasted text",
        "pdf": "PDF upload",
        "topic": "Wikipedia article",
    }[document.source_type.value]
    metrics = st.columns(3)
    metrics[0].metric("Source", source_label)
    metrics[1].metric("Characters", f"{document.character_count:,}")
    metrics[2].metric("Pages", document.page_count if document.page_count else "—")
    with st.expander("Processing details"):
        st.caption(
            f"Prepared {sentence_count} sentences across {chunk_count} sections and found "
            f"{candidate_count} quiz concepts."
        )
        st.text_area("Source preview", document.text[:800], height=160, disabled=True)


@st.cache_resource(show_spinner=False)
def load_cached_question_generator() -> T5QuestionGenerator:
    """Cache the heavyweight local model only after a user explicitly requests generation."""

    return T5QuestionGenerator(load_question_generation_model())


def process_document(document: Document) -> None:
    """Run downstream preparation once and retain it only in create state."""

    sentences = preprocess_document(document)
    chunks = chunk_sentences(sentences)
    candidates = extract_and_rank_candidates(sentences)
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
    st.header("Attempt Quiz")
    st.caption(f"{quiz.actual_question_count} questions · Difficulty: {quiz.difficulty.value}")
    st.info("Choose an answer for each question, then submit when you are ready.")
    for warning in quiz.warnings:
        st.warning(warning)

    with st.form(f"quiz-attempt-{quiz.id}"):
        answers: dict[str, AnswerValue] = {}
        for index, question in enumerate(quiz.questions, start=1):
            st.markdown(f"**{index}. {question.question_type.value}**")
            st.write(question.prompt)
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
                )
            elif question.question_type is QuestionType.FILL_IN_THE_BLANK:
                answers[question.id] = st.text_input(
                    "Your answer",
                    value=saved_answer if isinstance(saved_answer, str) else "",
                    key=answer_key,
                )
            else:
                answers[question.id] = st.text_area(
                    "Your answer",
                    value=saved_answer if isinstance(saved_answer, str) else "",
                    key=answer_key,
                    height=90,
                )
        submitted = st.form_submit_button("Submit Quiz", type="primary")

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
    st.header("Quiz Results")
    metrics = st.columns(4)
    metrics[0].metric("Score", f"{result.correct_count} / {result.total_questions}")
    metrics[1].metric("Percentage", f"{result.percentage}%")
    metrics[2].metric("Incorrect", result.incorrect_count)
    metrics[3].metric("Unanswered", result.unanswered_count)
    if quiz.source_metadata.get("source_provider") == "Wikipedia":
        article_title = quiz.source_metadata.get("article_title", "Wikipedia article")
        article_url = quiz.source_metadata.get("article_url")
        if article_url:
            st.markdown(f"Source article: [{article_title}]({article_url})")
        else:
            st.caption(f"Source provider: Wikipedia · Article: {article_title}")
    st.subheader("Breakdown by question type")
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
    st.subheader("Review")
    for index, question in enumerate(quiz.questions, start=1):
        question_result = results_by_id[question.id]
        st.markdown(f"**{index}. {question.question_type.value}**")
        st.write(question.prompt)
        if question_result.status is AnswerStatus.CORRECT:
            st.success("Correct")
        elif question_result.status is AnswerStatus.UNANSWERED:
            st.warning("Unanswered")
        else:
            st.error("Incorrect")
        if question.choices:
            for choice in question.choices:
                st.write(f"- {choice.text}")
        st.write(f"Your answer: `{_review_answer(question, question_result)}`")
        st.write(f"Correct answer: `{question.correct_answer}`")
        if question.question_type is QuestionType.TRUE_FALSE and question.boolean_answer is False:
            st.write(f"Original source fact: {question.source_sentence}")
        if question.page_number is not None:
            st.caption(f"Source page {question.page_number}")
        with st.expander("Source excerpt"):
            st.write(question.source_excerpt)


def _topic_result_label(result: TopicSearchResult) -> str:
    """Format a compact, readable label without exposing API internals."""

    return f"{result.title} — {result.description}" if result.description else result.title


def render_topic_input() -> None:
    """Render explicit Wikipedia search, selection, and processing actions."""

    state = get_topic_state(st.session_state)
    st.caption(
        "Topic mode sends your topic to Wikipedia and uses the selected article as quiz source."
    )
    query = st.text_input("Enter a topic", value=state.query, key="topic_query_input")
    updated_state = update_topic_query(state, query)
    if updated_state != state:
        state = updated_state
        save_topic_state(st.session_state, state)
        clear_processed_content()
        st.session_state.pop("topic_article_selection", None)

    if st.button("Search Wikipedia", type="primary"):
        try:
            results = search_cached_wikipedia(query)
            state = store_search_results(state, query=query, results=results)
            save_topic_state(st.session_state, state)
            st.session_state.pop("topic_article_selection", None)
        except QuestLLMError as error:
            st.error(str(error))

    if not state.results:
        return
    st.subheader("Choose a Wikipedia article")
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
    )
    if selected_page_id is not None and selected_page_id != selected_id:
        state = select_topic_result(state, selected_page_id)
        save_topic_state(st.session_state, state)

    if state.selected_result is None:
        return
    st.caption(f"Selected article: {state.selected_result.title}")
    st.link_button("Open selected article", state.selected_result.article_url)
    if st.button("Process Selected Article", type="primary"):
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

    st.header("Create a quiz")
    st.caption("1. Choose a source · 2. Prepare it · 3. Set quiz options · 4. Generate")
    source_choice = st.radio(
        "1. Choose your source",
        ("Paste Text", "Upload PDF", "Enter Topic"),
        horizontal=True,
        key="source_mode",
    )
    previous_source = st.session_state.get("active_source_mode")
    if previous_source is not None and previous_source != source_choice:
        clear_processed_content()
        save_topic_state(st.session_state, reset_topic_state())
        st.session_state.pop("topic_query_input", None)
        st.session_state.pop("topic_article_selection", None)
    st.session_state["active_source_mode"] = source_choice

    if source_choice == "Paste Text":
        pasted_text = st.text_area(
            "2. Paste your learning material",
            placeholder="Paste learning material here. QuestLLM will prepare it for a quiz.",
            height=220,
        )
        if st.button("Prepare Text", type="primary"):
            try:
                process_document(ingest_text(pasted_text))
            except QuestLLMError as error:
                st.error(str(error))
    elif source_choice == "Upload PDF":
        uploaded_pdf = st.file_uploader("2. Upload a text-based PDF", type=["pdf"])
        if st.button("Prepare PDF", type="primary"):
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

    st.subheader("3. Quiz settings")
    selected_type_labels = st.multiselect(
        "Question types",
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
    with st.expander("Advanced settings"):
        generation_seed = st.number_input(
            "Generation seed",
            min_value=0,
            value=17,
            step=1,
            help="Use the same seed to reproduce a quiz from the same source and settings.",
        )

    if st.button("Generate Quiz", type="primary"):
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
            with st.spinner("Building your quiz… this can take longer on CPU."):
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
        except (QuestLLMError, ValueError) as error:
            save_workflow(st.session_state, reset_workflow())
            st.error(str(error))


workflow = get_workflow(st.session_state)
if workflow.stage is WorkflowStage.CREATE:
    render_create(workflow)
else:
    if st.button("Create New Quiz"):
        save_workflow(st.session_state, reset_workflow())
        clear_processed_content()
        save_topic_state(st.session_state, reset_topic_state())
        st.session_state.pop("topic_query_input", None)
        st.session_state.pop("topic_article_selection", None)
        st.rerun()
    if workflow.stage is WorkflowStage.ATTEMPT:
        render_attempt(workflow)
    elif workflow.stage is WorkflowStage.REVIEW:
        render_review(workflow)
    else:
        st.info("Generating your quiz. Please wait for the current request to finish.")
