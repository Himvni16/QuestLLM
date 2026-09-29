"""Adapters that convert user input into QuestLLM documents."""

from questllm.ingestion.pdf import ingest_pdf
from questllm.ingestion.text import ingest_text
from questllm.ingestion.topic import TopicSearchResult, WikipediaClient, ingest_topic

__all__ = ["TopicSearchResult", "WikipediaClient", "ingest_pdf", "ingest_text", "ingest_topic"]
