"""Explicit setup and verification for QuestLLM's NLTK data."""

import argparse
import socket
import sys
from pathlib import Path

import nltk

from questllm.config import NLTK_DOWNLOAD_TIMEOUT_SECONDS
from questllm.exceptions import NltkResourceError

_PUNKT_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab/english/",
}
_POS_TAGGER_RESOURCES = {
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
}


def nltk_data_directory() -> Path:
    """Return the NLTK data directory owned by the active Python environment."""

    return Path(sys.prefix) / "nltk_data"


def configure_nltk_data_path() -> Path:
    """Prioritize the active environment's NLTK directory without removing other paths."""

    data_directory = nltk_data_directory()
    directory_text = str(data_directory)
    nltk.data.path[:] = [path for path in nltk.data.path if path != directory_text]
    nltk.data.path.insert(0, directory_text)
    return data_directory


def missing_punkt_resources() -> tuple[str, ...]:
    """Return the Punkt packages unavailable in the local NLTK data directory."""

    configure_nltk_data_path()
    missing = []
    for package_name, resource_path in _PUNKT_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package_name)
    return tuple(missing)


def ensure_punkt_resources() -> None:
    """Raise a clear error instead of downloading tokenizer data during app execution."""

    missing = missing_punkt_resources()
    if missing:
        packages = ", ".join(missing)
        raise NltkResourceError(
            f"NLTK tokenizer data is missing: {packages}. "
            "Run `python -m questllm.nltk_resources --download` once, then try again."
        )


def missing_pos_tagger_resources() -> tuple[str, ...]:
    """Return the POS-tagger packages unavailable in the local NLTK data directory."""

    configure_nltk_data_path()
    missing = []
    for package_name, resource_path in _POS_TAGGER_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package_name)
    return tuple(missing)


def ensure_pos_tagger_resources() -> None:
    """Raise a clear error instead of downloading tagger data during app execution."""

    missing = missing_pos_tagger_resources()
    if missing:
        packages = ", ".join(missing)
        raise NltkResourceError(
            f"NLTK POS tagger data is missing: {packages}. "
            "Run `python -m questllm.nltk_resources --download` once, then try again."
        )


def missing_required_resources() -> tuple[str, ...]:
    """Return all tokenizer and POS-tagger packages needed by QuestLLM's current phases."""

    return missing_punkt_resources() + missing_pos_tagger_resources()


def _missing_resources_in(
    directory: Path,
    resources: dict[str, str],
) -> tuple[str, ...]:
    """Return packages absent from one explicit NLTK data directory."""

    missing = []
    for package_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path, paths=[str(directory)])
        except LookupError:
            missing.append(package_name)
    return tuple(missing)


def _download_package(package_name: str, data_directory: Path) -> bool:
    """Download one package with a bounded network timeout in the explicit CLI path only."""

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(NLTK_DOWNLOAD_TIMEOUT_SECONDS)
    try:
        return bool(nltk.download(package_name, download_dir=str(data_directory)))
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(previous_timeout)


def download_punkt_resources() -> None:
    """Download required Punkt packages when explicitly requested by a user or setup script."""

    data_directory = configure_nltk_data_path()
    data_directory.mkdir(parents=True, exist_ok=True)
    for package_name in _missing_resources_in(data_directory, _PUNKT_RESOURCES):
        if not _download_package(package_name, data_directory):
            raise NltkResourceError(
                f"NLTK could not download '{package_name}' into {data_directory}. "
                "Check your connection or proxy, then rerun the setup command."
            )

    ensure_punkt_resources()


def download_required_resources() -> None:
    """Download every missing QuestLLM NLTK package after explicit user approval."""

    data_directory = configure_nltk_data_path()
    data_directory.mkdir(parents=True, exist_ok=True)
    resources = {**_PUNKT_RESOURCES, **_POS_TAGGER_RESOURCES}
    for package_name in _missing_resources_in(data_directory, resources):
        if not _download_package(package_name, data_directory):
            raise NltkResourceError(
                f"NLTK could not download '{package_name}' into {data_directory}. "
                "Check your connection or proxy, then rerun the setup command."
            )

    ensure_punkt_resources()
    ensure_pos_tagger_resources()


def initialize_runtime_resources() -> None:
    """Ensure app resources once, downloading only packages that are genuinely missing."""

    if missing_required_resources():
        download_required_resources()


def main() -> None:
    """Provide an explicit command-line setup path for QuestLLM NLTK resources."""

    parser = argparse.ArgumentParser(description="Verify or download QuestLLM NLTK tokenizer data.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing tokenizer and POS-tagger resources instead of only verifying them.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the active environment's NLTK directory and resource status.",
    )
    args = parser.parse_args()

    try:
        if args.status:
            data_directory = configure_nltk_data_path()
            resources = {**_PUNKT_RESOURCES, **_POS_TAGGER_RESOURCES}
            missing = _missing_resources_in(data_directory, resources)
            print(f"QuestLLM NLTK directory: {data_directory}")
            print(f"NLTK search paths: {nltk.data.path}")
            if missing:
                print("Missing from the active environment: " + ", ".join(missing))
                print("Run `python -m questllm.nltk_resources --download` to install them.")
            else:
                print("QuestLLM NLTK resources are ready.")
        elif args.download:
            download_required_resources()
            print(f"QuestLLM NLTK resources are ready in: {configure_nltk_data_path()}")
        else:
            ensure_punkt_resources()
            ensure_pos_tagger_resources()
            print(f"QuestLLM NLTK resources are ready in: {configure_nltk_data_path()}")
    except NltkResourceError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
