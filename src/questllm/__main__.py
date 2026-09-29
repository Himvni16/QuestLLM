"""Provide a clear message when the package is run directly."""

from questllm.config import APP_NAME


def main() -> None:
    """Print the supported local launch command."""

    print(f"Run {APP_NAME} with: streamlit run app.py")


if __name__ == "__main__":
    main()
