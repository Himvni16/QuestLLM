"""Offline tests for lazy local model-loader behaviour."""

import pytest

from questllm import model_loader
from questllm.exceptions import ModelLoadError


class _FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.evaluated = False

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self

    def eval(self) -> "_FakeModel":
        self.evaluated = True
        return self


def test_load_question_generation_model_uses_cpu_and_evaluation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr(model_loader, "select_device", lambda: "cpu")
    monkeypatch.setattr(
        model_loader.AutoTokenizer,
        "from_pretrained",
        lambda model_id: {"tokenizer": model_id},
    )
    monkeypatch.setattr(
        model_loader.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda _model_id: fake_model,
    )

    loaded = model_loader.load_question_generation_model("fake/t5")

    assert loaded.device == "cpu"
    assert loaded.model_id == "fake/t5"
    assert fake_model.device == "cpu"
    assert fake_model.evaluated is True


def test_model_loader_translates_download_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_network_error(_model_id: str) -> object:
        raise OSError("network connection failed")

    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", raise_network_error)

    with pytest.raises(ModelLoadError, match="could not download"):
        model_loader.load_question_generation_model("fake/t5")
