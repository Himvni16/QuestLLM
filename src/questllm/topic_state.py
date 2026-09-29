"""Small, testable Streamlit-session helpers for Wikipedia topic selection."""

from collections.abc import MutableMapping
from dataclasses import dataclass, field, replace

from questllm.document import Document, SourceType
from questllm.exceptions import WorkflowError
from questllm.ingestion.topic import TopicSearchResult

TOPIC_STATE_SESSION_KEY = "questllm_topic_state"


@dataclass(frozen=True, slots=True)
class TopicState:
    """Search, selection, and retrieved-document state for one topic input flow."""

    query: str = ""
    results: tuple[TopicSearchResult, ...] = field(default_factory=tuple)
    selected_result: TopicSearchResult | None = None
    document: Document | None = None


def _normalize_state_query(query: str) -> str:
    return " ".join(query.split())


def update_topic_query(state: TopicState, query: str) -> TopicState:
    """Invalidate stale results, selection, and content when the query changes."""

    normalized = _normalize_state_query(query)
    if normalized == state.query:
        return state
    return TopicState(query=normalized)


def store_search_results(
    state: TopicState,
    *,
    query: str,
    results: tuple[TopicSearchResult, ...],
) -> TopicState:
    """Store fresh results for the current query, without choosing an article implicitly."""

    normalized = _normalize_state_query(query)
    if not normalized:
        raise WorkflowError("Enter a topic before storing Wikipedia search results.")
    return TopicState(query=normalized, results=results)


def select_topic_result(state: TopicState, page_id: int) -> TopicState:
    """Persist an explicit user selection by MediaWiki's stable article page ID."""

    selected = next((result for result in state.results if result.page_id == page_id), None)
    if selected is None:
        raise WorkflowError("Choose an article from the current Wikipedia search results.")
    return replace(state, selected_result=selected, document=None)


def store_topic_document(state: TopicState, document: Document) -> TopicState:
    """Persist a retrieved topic document only for the currently selected article."""

    if state.selected_result is None:
        raise WorkflowError("Choose a Wikipedia article before processing it.")
    if document.source_type is not SourceType.TOPIC:
        raise WorkflowError("Only topic-derived documents can be stored in topic state.")
    return replace(state, document=document)


def reset_topic_state() -> TopicState:
    """Clear all topic-specific state without affecting other input modes."""

    return TopicState()


def get_topic_state(session_state: MutableMapping[str, object]) -> TopicState:
    """Read or initialize topic state from a generic session-state mapping."""

    state = session_state.get(TOPIC_STATE_SESSION_KEY)
    if isinstance(state, TopicState):
        return state
    initialized = reset_topic_state()
    session_state[TOPIC_STATE_SESSION_KEY] = initialized
    return initialized


def save_topic_state(session_state: MutableMapping[str, object], state: TopicState) -> None:
    """Persist all topic state together under one stable session key."""

    session_state[TOPIC_STATE_SESSION_KEY] = state
