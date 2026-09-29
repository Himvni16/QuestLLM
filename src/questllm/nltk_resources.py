"""Explicit setup and verification for NLTK sentence-tokenizer data."""

import argparse

import nltk

from questllm.exceptions import NltkResourceError

_PUNKT_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab/english/",
}


def missing_punkt_resources() -> tuple[str, ...]:
    """Return the Punkt packages unavailable in the local NLTK data directory."""

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


def download_punkt_resources() -> None:
    """Download required Punkt packages when explicitly requested by a user or setup script."""

    for package_name in missing_punkt_resources():
        if not nltk.download(package_name):
            raise NltkResourceError(
                f"NLTK could not download the '{package_name}' tokenizer resource."
            )

    ensure_punkt_resources()


def main() -> None:
    """Provide an explicit command-line setup path for tokenizer resources."""

    parser = argparse.ArgumentParser(description="Verify or download QuestLLM NLTK tokenizer data.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing Punkt resources instead of only verifying them.",
    )
    args = parser.parse_args()

    try:
        if args.download:
            download_punkt_resources()
            print("NLTK tokenizer resources are ready.")
        else:
            ensure_punkt_resources()
            print("NLTK tokenizer resources are already available.")
    except NltkResourceError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
