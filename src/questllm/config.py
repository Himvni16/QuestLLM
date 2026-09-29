"""Small, dependency-free configuration values for QuestLLM."""

from dataclasses import dataclass

APP_NAME = "QuestLLM"
APP_DESCRIPTION = "Create practice quizzes from your learning material, locally."
MINIMUM_TEXT_CHARACTERS = 50


@dataclass(frozen=True)
class AppSettings:
    """Configuration that is useful before quiz-generation features are introduced."""

    app_name: str = APP_NAME
    language: str = "en"
    minimum_text_characters: int = MINIMUM_TEXT_CHARACTERS


def get_settings() -> AppSettings:
    """Return the current application settings."""

    return AppSettings()
