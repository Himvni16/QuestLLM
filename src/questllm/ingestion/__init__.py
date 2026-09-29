"""Adapters that convert user input into QuestLLM documents."""

from questllm.ingestion.pdf import ingest_pdf
from questllm.ingestion.text import ingest_text

__all__ = ["ingest_pdf", "ingest_text"]
