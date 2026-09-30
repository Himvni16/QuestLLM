"""Offline tests for explicit NLTK resource setup behavior."""

import pytest

from questllm import nltk_resources
from questllm.exceptions import NltkResourceError


def _configure_unwritable_environment_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    environment_root = tmp_path / "managed-python"
    environment_directory = environment_root / "nltk_data"
    fallback_directory = tmp_path / "runtime" / "questllm_nltk_data"
    real_ensure_writable = nltk_resources._ensure_directory_is_writable

    monkeypatch.setattr(nltk_resources.sys, "prefix", str(environment_root))
    monkeypatch.setattr(
        nltk_resources.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "runtime"),
    )

    def fake_ensure_writable(directory) -> None:
        if directory == environment_directory:
            raise PermissionError("managed Python installation is read-only")
        real_ensure_writable(directory)

    monkeypatch.setattr(nltk_resources, "_ensure_directory_is_writable", fake_ensure_writable)
    return fallback_directory


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


def test_ensure_pos_tagger_resources_raises_a_clear_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nltk_resources,
        "missing_pos_tagger_resources",
        lambda: ("averaged_perceptron_tagger_eng",),
    )

    with pytest.raises(NltkResourceError, match="python -m questllm.nltk_resources --download"):
        nltk_resources.ensure_pos_tagger_resources()


def test_missing_pos_tagger_resources_returns_only_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find(resource_path: str) -> str:
        if resource_path.startswith("taggers/averaged_perceptron_tagger_eng"):
            raise LookupError("missing")
        return resource_path

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)

    assert nltk_resources.missing_pos_tagger_resources() == ("averaged_perceptron_tagger_eng",)


def test_writable_environment_directory_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    environment_root = tmp_path / "writable-venv"
    expected = environment_root / "nltk_data"
    monkeypatch.setattr(nltk_resources.sys, "prefix", str(environment_root))
    monkeypatch.setattr(
        nltk_resources.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "unused-runtime"),
    )

    selected = nltk_resources.nltk_data_directory()

    assert selected == expected
    assert selected.is_dir()


def test_unwritable_environment_directory_uses_runtime_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fallback_directory = _configure_unwritable_environment_directory(monkeypatch, tmp_path)

    selected = nltk_resources.nltk_data_directory()

    assert selected == fallback_directory
    assert selected.is_dir()


def test_fallback_directory_is_first_in_nltk_search_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fallback_directory = _configure_unwritable_environment_directory(monkeypatch, tmp_path)
    monkeypatch.setattr(nltk_resources.nltk.data, "path", ["existing-nltk-path"])

    selected = nltk_resources.configure_nltk_data_path()

    assert selected == fallback_directory
    assert nltk_resources.nltk.data.path == [str(fallback_directory), "existing-nltk-path"]


def test_status_reports_the_selected_fallback_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fallback_directory = _configure_unwritable_environment_directory(monkeypatch, tmp_path)
    monkeypatch.setattr(nltk_resources.sys, "argv", ["nltk_resources", "--status"])
    monkeypatch.setattr(nltk_resources, "_missing_resources_in", lambda directory, resources: ())

    nltk_resources.main()

    output = capsys.readouterr().out
    assert f"QuestLLM NLTK directory: {fallback_directory}" in output
    assert "QuestLLM NLTK resources are ready." in output


def test_download_uses_the_active_environment_nltk_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    downloaded: list[tuple[str, str]] = []
    monkeypatch.setattr(nltk_resources.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        nltk_resources,
        "_missing_resources_in",
        lambda directory, resources: tuple(resources),
    )
    monkeypatch.setattr(nltk_resources, "ensure_punkt_resources", lambda: None)
    monkeypatch.setattr(nltk_resources, "ensure_pos_tagger_resources", lambda: None)

    def fake_download(package_name: str, *, download_dir: str) -> bool:
        downloaded.append((package_name, download_dir))
        return True

    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    nltk_resources.download_required_resources()

    expected_directory = str(tmp_path / "nltk_data")
    assert {package_name for package_name, _ in downloaded} == {
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    }
    assert {directory for _, directory in downloaded} == {expected_directory}


def test_missing_packages_download_to_selected_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fallback_directory = _configure_unwritable_environment_directory(monkeypatch, tmp_path)
    downloaded = []
    monkeypatch.setattr(
        nltk_resources,
        "_missing_resources_in",
        lambda directory, resources: ("punkt_tab",),
    )
    monkeypatch.setattr(
        nltk_resources,
        "_download_package",
        lambda package, directory: downloaded.append((package, directory)) or True,
    )
    monkeypatch.setattr(nltk_resources, "ensure_punkt_resources", lambda: None)
    monkeypatch.setattr(nltk_resources, "ensure_pos_tagger_resources", lambda: None)

    nltk_resources.download_required_resources()

    assert downloaded == [("punkt_tab", fallback_directory)]


def test_existing_fallback_resources_are_not_downloaded_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_unwritable_environment_directory(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nltk_resources,
        "_missing_resources_in",
        lambda directory, resources: (),
    )
    download_calls = 0

    def fake_download(package, directory) -> bool:
        nonlocal download_calls
        download_calls += 1
        return True

    monkeypatch.setattr(nltk_resources, "_download_package", fake_download)
    monkeypatch.setattr(nltk_resources, "ensure_punkt_resources", lambda: None)
    monkeypatch.setattr(nltk_resources, "ensure_pos_tagger_resources", lambda: None)

    nltk_resources.download_required_resources()

    assert download_calls == 0


def test_runtime_initialization_does_not_download_when_resources_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nltk_resources, "missing_required_resources", lambda: ())
    download_called = False

    def fake_download() -> None:
        nonlocal download_called
        download_called = True

    monkeypatch.setattr(nltk_resources, "download_required_resources", fake_download)

    nltk_resources.initialize_runtime_resources()

    assert download_called is False


def test_runtime_initialization_downloads_only_when_resources_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nltk_resources, "missing_required_resources", lambda: ("punkt",))
    download_calls = 0

    def fake_download() -> None:
        nonlocal download_calls
        download_calls += 1

    monkeypatch.setattr(nltk_resources, "download_required_resources", fake_download)

    nltk_resources.initialize_runtime_resources()

    assert download_calls == 1
