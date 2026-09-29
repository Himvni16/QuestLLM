"""Small, dependency-free configuration values for QuestLLM."""

from dataclasses import dataclass

APP_NAME = "QuestLLM"
APP_DESCRIPTION = "Create practice quizzes from your learning material, locally."
MINIMUM_TEXT_CHARACTERS = 50
MINIMUM_SENTENCE_CHARACTERS = 15
TARGET_CHUNK_SIZE = 240
CHUNK_OVERLAP_SIZE = 40
MINIMUM_CANDIDATE_CHARACTERS = 2
MAXIMUM_CANDIDATE_TOKENS = 5
MAXIMUM_CANDIDATE_CHARACTERS = 80
DEFAULT_CANDIDATE_LIMIT = 12
CANDIDATE_COVERAGE_PENALTY = 0.08
CANDIDATE_FREQUENCY_BONUS = 0.05
CANDIDATE_SENTENCE_QUALITY_BONUS = 0.05
CANDIDATE_LENGTH_PENALTY = 0.02

# This intentionally small list only filters obvious non-answer phrases. It is not
# a replacement for NLTK's broader language resources or later NLP processing.
CANDIDATE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "he",
        "her",
        "him",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "we",
        "with",
        "you",
        "your",
    }
)
QUESTION_GENERATION_MODEL_ID = "valhalla/t5-base-qa-qg-hl"
MODEL_MAX_INPUT_TOKENS = 512
MODEL_INPUT_TOKEN_RESERVE = 96
MODEL_CONTEXT_TOKEN_BUDGET = MODEL_MAX_INPUT_TOKENS - MODEL_INPUT_TOKEN_RESERVE
QUESTION_PREVIEW_LIMIT = 3
QUESTION_GENERATION_NUM_BEAMS = 4
QUESTION_GENERATION_MAX_NEW_TOKENS = 64
MINIMUM_QUESTION_CHARACTERS = 12
MAXIMUM_QUESTION_CHARACTERS = 240
FILL_IN_THE_BLANK_TOKEN = "________"
QUESTION_TYPE_PREVIEW_LIMIT = 1
MCQ_CHOICE_ORDER_SEED = 17
MINIMUM_QUIZ_QUESTION_COUNT = 1
MAXIMUM_QUIZ_QUESTION_COUNT = 20
MAXIMUM_CANDIDATES_ATTEMPTED = 24
MAXIMUM_STEM_GENERATION_ATTEMPTS = 24
QUESTION_NEAR_DUPLICATE_THRESHOLD = 0.82
QUESTION_QUALITY_CUTOFF = 0.55
QUIZ_COVERAGE_BONUS = 0.08


@dataclass(frozen=True)
class AppSettings:
    """Configuration that is useful before quiz-generation features are introduced."""

    app_name: str = APP_NAME
    language: str = "en"
    minimum_text_characters: int = MINIMUM_TEXT_CHARACTERS
    minimum_sentence_characters: int = MINIMUM_SENTENCE_CHARACTERS
    target_chunk_size: int = TARGET_CHUNK_SIZE
    chunk_overlap_size: int = CHUNK_OVERLAP_SIZE
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    question_generation_model_id: str = QUESTION_GENERATION_MODEL_ID
    question_preview_limit: int = QUESTION_PREVIEW_LIMIT
    question_type_preview_limit: int = QUESTION_TYPE_PREVIEW_LIMIT
    minimum_quiz_question_count: int = MINIMUM_QUIZ_QUESTION_COUNT
    maximum_quiz_question_count: int = MAXIMUM_QUIZ_QUESTION_COUNT


def get_settings() -> AppSettings:
    """Return the current application settings."""

    return AppSettings()
