"""Basic package import tests."""

import questllm


def test_package_exposes_a_version() -> None:
    """The package is importable from the src layout."""

    assert questllm.__version__ == "0.1.0"
