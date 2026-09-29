"""Small, dependency-free configuration values for QuestLLM."""

from dataclasses import dataclass

APP_NAME = "QuestLLM"
APP_DESCRIPTION = "Create practice quizzes from your learning material, locally."
MINIMUM_TEXT_CHARACTERS = 50
MINIMUM_SENTENCE_CHARACTERS = 15
TARGET_CHUNK_SIZE = 240
CHUNK_OVERLAP_SIZE = 40


@dataclass(frozen=True)
class AppSettings:
    """Configuration that is useful before quiz-generation features are introduced."""

    app_name: str = APP_NAME
    language: str = "en"
    minimum_text_characters: int = MINIMUM_TEXT_CHARACTERS
    minimum_sentence_characters: int = MINIMUM_SENTENCE_CHARACTERS
    target_chunk_size: int = TARGET_CHUNK_SIZE
    chunk_overlap_size: int = CHUNK_OVERLAP_SIZE


def get_settings() -> AppSettings:
    """Return the current application settings."""

    return AppSettings()
