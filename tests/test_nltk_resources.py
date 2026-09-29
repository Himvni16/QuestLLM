"""Offline tests for explicit NLTK resource setup behavior."""

import pytest

from questllm import nltk_resources
from questllm.exceptions import NltkResourceError


def test_ensure_punkt_resources_raises_a_clear_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nltk_resources, "missing_punkt_resources", lambda: ("punkt_tab",))

    with pytest.raises(NltkResourceError, match="python -m questllm.nltk_resources --download"):
        nltk_resources.ensure_punkt_resources()


def test_missing_punkt_resources_returns_only_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find(resource_path: str) -> str:
        if resource_path.startswith("tokenizers/punkt_tab"):
            raise LookupError("missing")
        return resource_path

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)

    assert nltk_resources.missing_punkt_resources() == ("punkt_tab",)
