"""Wikipedia-backed topic search and article ingestion for QuestLLM."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from html import unescape
from urllib.parse import quote

import requests

from questllm.config import (
    MAXIMUM_TOPIC_ARTICLE_CHARACTERS,
    MINIMUM_TEXT_CHARACTERS,
    MINIMUM_TOPIC_CHARACTERS,
    TOPIC_REQUEST_TIMEOUT_SECONDS,
    TOPIC_SEARCH_RESULT_LIMIT,
    WIKIPEDIA_API_URL,
    WIKIPEDIA_ARTICLE_BASE_URL,
    WIKIPEDIA_USER_AGENT,
)
from questllm.document import Document, SourceType
from questllm.exceptions import (
    AmbiguousTopicError,
    DisambiguationPageError,
    InvalidTopicError,
    NoTopicSearchResultsError,
    NoUsableTopicTextError,
    TopicApiError,
    TopicNetworkError,
)
from questllm.ingestion.text import normalize_text

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_TRAILING_SECTION = re.compile(
    r"(?im)^\s*(references|notes|bibliography|further reading|external links|see also)\s*$"
)
_DISAMBIGUATION_TITLE = re.compile(r"\(disambiguation\)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TopicSearchResult:
    """A user-selectable Wikipedia search result."""

    page_id: int
    title: str
    description: str
    article_url: str


def normalize_topic_query(topic: str) -> str:
    """Validate and normalize a concise topic query without altering its meaning."""

    normalized = _WHITESPACE.sub(" ", topic).strip()
    if not normalized:
        raise InvalidTopicError("Enter a topic to search Wikipedia.")
    if len(normalized) < MINIMUM_TOPIC_CHARACTERS:
        raise InvalidTopicError(
            f"Enter at least {MINIMUM_TOPIC_CHARACTERS} characters to search Wikipedia."
        )
    return normalized


def article_url_for_title(title: str) -> str:
    """Build the canonical English Wikipedia URL for an article title."""

    return f"{WIKIPEDIA_ARTICLE_BASE_URL}{quote(title.replace(' ', '_'))}"


def _plain_text(value: str) -> str:
    """Remove API-provided HTML markup from search snippets."""

    return _WHITESPACE.sub(" ", unescape(_HTML_TAG.sub("", value))).strip()


def _article_paragraphs(extract: str) -> tuple[str, ...]:
    """Remove common trailing reference sections and preserve useful paragraphs."""

    before_trailing_section = _TRAILING_SECTION.split(extract, maxsplit=1)[0]
    normalized = normalize_text(before_trailing_section, repair_hyphenation=True)
    return tuple(paragraph for paragraph in normalized.split("\n\n") if paragraph)


def normalize_article_extract(extract: str, *, maximum_characters: int) -> str:
    """Normalize an API article extract and cap it at complete paragraph boundaries."""

    paragraphs = _article_paragraphs(extract)
    selected: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        separator_length = 2 if selected else 0
        proposed_length = current_length + separator_length + len(paragraph)
        if proposed_length > maximum_characters:
            break
        selected.append(paragraph)
        current_length = proposed_length
    return "\n\n".join(selected)


class WikipediaClient:
    """Small MediaWiki API client that is independent of Streamlit and quiz generation."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        api_url: str = WIKIPEDIA_API_URL,
        timeout_seconds: int = TOPIC_REQUEST_TIMEOUT_SECONDS,
        search_limit: int = TOPIC_SEARCH_RESULT_LIMIT,
        maximum_article_characters: int = MAXIMUM_TOPIC_ARTICLE_CHARACTERS,
    ) -> None:
        self._session = session or requests.Session()
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds
        self._search_limit = search_limit
        self._maximum_article_characters = maximum_article_characters

    def _get_json(self, params: Mapping[str, str | int]) -> Mapping[str, object]:
        try:
            response = self._session.get(
                self._api_url,
                params=params,
                headers={"User-Agent": WIKIPEDIA_USER_AGENT},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as error:
            raise TopicNetworkError(
                "Wikipedia timed out. Check your connection and try again."
            ) from error
        except requests.HTTPError as error:
            raise TopicApiError(
                "Wikipedia could not provide that content. Try another topic."
            ) from error
        except requests.RequestException as error:
            raise TopicNetworkError(
                "Wikipedia could not be reached. Topic mode requires internet access."
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise TopicApiError(
                "Wikipedia returned an unreadable response. Try again shortly."
            ) from error
        if not isinstance(payload, Mapping):
            raise TopicApiError("Wikipedia returned an unexpected response. Try another topic.")
        if "error" in payload:
            raise TopicApiError("Wikipedia could not provide that content. Try another topic.")
        return payload

    def search(self, topic: str) -> tuple[TopicSearchResult, ...]:
        """Search Wikipedia and return a compact, user-selectable result list."""

        query = normalize_topic_query(topic)
        payload = self._get_json(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "search",
                "srsearch": query,
                "srlimit": self._search_limit,
                "srprop": "snippet",
            }
        )
        query_data = payload.get("query")
        if not isinstance(query_data, Mapping):
            raise TopicApiError("Wikipedia returned an unexpected search response. Try again.")
        raw_results = query_data.get("search")
        if not isinstance(raw_results, list):
            raise TopicApiError("Wikipedia returned an unexpected search response. Try again.")

        results: list[TopicSearchResult] = []
        for item in raw_results:
            if not isinstance(item, Mapping):
                raise TopicApiError("Wikipedia returned an unexpected search result. Try again.")
            page_id = item.get("pageid")
            title = item.get("title")
            snippet = item.get("snippet", "")
            if (
                not isinstance(page_id, int)
                or not isinstance(title, str)
                or not isinstance(snippet, str)
            ):
                raise TopicApiError("Wikipedia returned an incomplete search result. Try again.")
            results.append(
                TopicSearchResult(
                    page_id=page_id,
                    title=title,
                    description=_plain_text(snippet),
                    article_url=article_url_for_title(title),
                )
            )
        if not results:
            raise NoTopicSearchResultsError(
                "No Wikipedia articles matched that topic. Try a different query."
            )
        return tuple(results)

    def retrieve_article(self, result: TopicSearchResult, *, requested_topic: str) -> Document:
        """Retrieve one selected article as a normalized, provenance-rich document."""

        payload = self._get_json(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "extracts|pageprops",
                "pageids": result.page_id,
                "explaintext": 1,
                "exsectionformat": "plain",
                "redirects": 1,
            }
        )
        query_data = payload.get("query")
        if not isinstance(query_data, Mapping):
            raise TopicApiError("Wikipedia returned an unexpected article response. Try again.")
        raw_pages = query_data.get("pages")
        if isinstance(raw_pages, Mapping):
            pages = tuple(raw_pages.values())
        elif isinstance(raw_pages, list):
            pages = tuple(raw_pages)
        else:
            raise TopicApiError("Wikipedia returned an unexpected article response. Try again.")
        if len(pages) != 1 or not isinstance(pages[0], Mapping):
            raise TopicApiError(
                "Wikipedia could not resolve that article. Search again and choose another result."
            )
        page = pages[0]
        if page.get("missing") is not None:
            raise NoTopicSearchResultsError(
                "That Wikipedia article is no longer available. Search again."
            )

        title = page.get("title")
        extract = page.get("extract")
        page_properties = page.get("pageprops", {})
        if (
            not isinstance(title, str)
            or not isinstance(extract, str)
            or not isinstance(page_properties, Mapping)
        ):
            raise TopicApiError(
                "Wikipedia returned incomplete article content. Try another result."
            )
        if "disambiguation" in page_properties or _DISAMBIGUATION_TITLE.search(title):
            raise DisambiguationPageError(
                "That result is a disambiguation page. Search again and select a specific article."
            )

        text = normalize_article_extract(
            extract,
            maximum_characters=self._maximum_article_characters,
        )
        if len(text) < MINIMUM_TEXT_CHARACTERS:
            raise NoUsableTopicTextError(
                "That Wikipedia article has too little readable text to create a useful quiz."
            )
        return Document(
            source_type=SourceType.TOPIC,
            text=text,
            metadata={
                "requested_topic": normalize_topic_query(requested_topic),
                "source_provider": "Wikipedia",
                "article_title": title,
                "article_url": article_url_for_title(title),
            },
        )


def ingest_topic(topic: str, *, client: WikipediaClient | None = None) -> Document:
    """Ingest a topic only when it resolves to one unambiguous Wikipedia article."""

    wikipedia = client or WikipediaClient()
    results = wikipedia.search(topic)
    if len(results) != 1:
        raise AmbiguousTopicError(
            "Choose one of the Wikipedia results before processing this topic."
        )
    return wikipedia.retrieve_article(results[0], requested_topic=topic)
