"""Offline tests for MediaWiki-backed topic ingestion."""

from collections.abc import Mapping

import pytest
import requests

from questllm.document import SourceType
from questllm.exceptions import (
    AmbiguousTopicError,
    DisambiguationPageError,
    NoTopicSearchResultsError,
    NoUsableTopicTextError,
    TopicApiError,
    TopicNetworkError,
)
from questllm.ingestion.topic import TopicSearchResult, WikipediaClient, ingest_topic


class FakeResponse:
    """Minimal requests response double with controllable JSON and status behavior."""

    def __init__(self, payload: object, *, status_error: Exception | None = None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error:
            raise self._status_error

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Queue API responses and record requests without opening network connections."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _search_payload(*items: Mapping[str, object]) -> dict[str, object]:
    return {"query": {"search": list(items)}}


def _article_payload(
    extract: str,
    *,
    title: str = "Photosynthesis",
    pageprops: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "query": {
            "pages": [
                {
                    "pageid": 42,
                    "title": title,
                    "extract": extract,
                    "pageprops": dict(pageprops or {}),
                }
            ]
        }
    }


def _result() -> TopicSearchResult:
    return TopicSearchResult(
        page_id=42,
        title="Photosynthesis",
        description="Process used by plants.",
        article_url="https://en.wikipedia.org/wiki/Photosynthesis",
    )


def test_search_parses_valid_and_multiple_results() -> None:
    session = FakeSession(
        [
            FakeResponse(
                _search_payload(
                    {"pageid": 42, "title": "Photosynthesis", "snippet": "Plant <b>process</b>"},
                    {"pageid": 43, "title": "Photosystem", "snippet": "Related system"},
                )
            )
        ]
    )

    results = WikipediaClient(session=session).search("  photosynthesis ")

    assert [result.page_id for result in results] == [42, 43]
    assert results[0].description == "Plant process"
    assert results[0].article_url.endswith("/Photosynthesis")
    assert session.calls[0]["params"] == {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "list": "search",
        "srsearch": "photosynthesis",
        "srlimit": 5,
        "srprop": "snippet",
    }


def test_search_rejects_no_results_and_malformed_or_network_responses() -> None:
    with pytest.raises(NoTopicSearchResultsError):
        WikipediaClient(session=FakeSession([FakeResponse(_search_payload())])).search("photosynthesis")
    with pytest.raises(TopicApiError):
        WikipediaClient(session=FakeSession([FakeResponse({"query": {}})])).search("photosynthesis")
    with pytest.raises(TopicNetworkError):
        WikipediaClient(session=FakeSession([requests.Timeout()])).search("photosynthesis")
    with pytest.raises(TopicApiError):
        WikipediaClient(
            session=FakeSession([FakeResponse({}, status_error=requests.HTTPError())])
        ).search("photosynthesis")


def test_retrieve_article_preserves_paragraphs_and_provenance() -> None:
    extract = (
        "Photosynthesis converts light energy into chemical energy in plants.\n\n"
        "The process uses chlorophyll and releases oxygen.\n\n"
        "References\n\nCitation boilerplate"
    )
    client = WikipediaClient(session=FakeSession([FakeResponse(_article_payload(extract))]))

    document = client.retrieve_article(_result(), requested_topic="photosynthesis")

    assert document.source_type is SourceType.TOPIC
    assert document.pages == ()
    assert document.text == (
        "Photosynthesis converts light energy into chemical energy in plants.\n\n"
        "The process uses chlorophyll and releases oxygen."
    )
    assert document.metadata == {
        "requested_topic": "photosynthesis",
        "source_provider": "Wikipedia",
        "article_title": "Photosynthesis",
        "article_url": "https://en.wikipedia.org/wiki/Photosynthesis",
    }


def test_retrieve_article_rejects_disambiguation_and_insufficient_text() -> None:
    disambiguation_client = WikipediaClient(
        session=FakeSession(
            [
                FakeResponse(
                    _article_payload(
                        "This page lists meanings for a term.",
                        title="Mercury (disambiguation)",
                        pageprops={"disambiguation": ""},
                    )
                )
            ]
        )
    )
    with pytest.raises(DisambiguationPageError):
        disambiguation_client.retrieve_article(_result(), requested_topic="Mercury")

    short_client = WikipediaClient(
        session=FakeSession([FakeResponse(_article_payload("Too short."))])
    )
    with pytest.raises(NoUsableTopicTextError):
        short_client.retrieve_article(_result(), requested_topic="photosynthesis")


def test_retrieve_article_caps_oversized_content_at_paragraph_boundaries() -> None:
    first_paragraph = "A" * 70
    second_paragraph = "B" * 70
    client = WikipediaClient(
        session=FakeSession([FakeResponse(_article_payload(f"{first_paragraph}\n\n{second_paragraph}"))]),
        maximum_article_characters=80,
    )

    document = client.retrieve_article(_result(), requested_topic="photosynthesis")

    assert document.text == first_paragraph
    assert len(document.text) <= 80


def test_ingest_topic_returns_the_existing_document_shape_for_one_result() -> None:
    session = FakeSession(
        [
            FakeResponse(
                _search_payload(
                    {"pageid": 42, "title": "Photosynthesis", "snippet": "Plant process"}
                )
            ),
            FakeResponse(
                _article_payload(
                    "Photosynthesis converts light energy into chemical energy for plant growth."
                )
            ),
        ]
    )

    document = ingest_topic("Photosynthesis", client=WikipediaClient(session=session))

    assert document.source_type is SourceType.TOPIC
    assert document.page_count == 0
    assert document.metadata["source_provider"] == "Wikipedia"


def test_ingest_topic_requires_selection_when_search_is_ambiguous() -> None:
    client = WikipediaClient(
        session=FakeSession(
            [
                FakeResponse(
                    _search_payload(
                        {"pageid": 42, "title": "Mercury", "snippet": "Element"},
                        {"pageid": 43, "title": "Mercury (planet)", "snippet": "Planet"},
                    )
                )
            ]
        )
    )

    with pytest.raises(AmbiguousTopicError):
        ingest_topic("Mercury", client=client)
