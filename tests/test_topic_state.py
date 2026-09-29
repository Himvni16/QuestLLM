"""Offline tests for topic search, selection, and reset session state."""

from questllm.document import Document, SourceType
from questllm.ingestion.topic import TopicSearchResult
from questllm.topic_state import (
    TopicState,
    get_topic_state,
    reset_topic_state,
    save_topic_state,
    select_topic_result,
    store_search_results,
    store_topic_document,
    update_topic_query,
)


def _result(page_id: int = 42) -> TopicSearchResult:
    return TopicSearchResult(
        page_id=page_id,
        title="Photosynthesis",
        description="Plant process",
        article_url="https://en.wikipedia.org/wiki/Photosynthesis",
    )


def test_query_change_invalidates_stale_results_selection_and_document() -> None:
    state = store_search_results(TopicState(), query="photosynthesis", results=(_result(),))
    state = select_topic_result(state, 42)
    state = store_topic_document(
        state,
        Document(
            source_type=SourceType.TOPIC,
            text="Enough source material for topic state testing.",
        ),
    )

    changed = update_topic_query(state, "cellular respiration")

    assert changed.query == "cellular respiration"
    assert changed.results == ()
    assert changed.selected_result is None
    assert changed.document is None


def test_selected_article_is_retained_across_session_reruns_and_reset_clears_it() -> None:
    session: dict[str, object] = {}
    state = store_search_results(TopicState(), query="photosynthesis", results=(_result(),))
    selected = select_topic_result(state, 42)
    save_topic_state(session, selected)

    restored = get_topic_state(session)

    assert restored.selected_result == _result()
    assert reset_topic_state() == TopicState()
