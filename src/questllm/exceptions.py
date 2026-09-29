"""Application-level errors surfaced by QuestLLM workflows."""


class QuestLLMError(Exception):
    """Base error for user-facing QuestLLM failures."""


class IngestionError(QuestLLMError):
    """Base error for source-content ingestion failures."""


class EmptyTextError(IngestionError):
    """Raised when pasted content has no readable text."""


class InsufficientTextError(IngestionError):
    """Raised when pasted content is too short to support a useful quiz."""


class InvalidPdfError(IngestionError):
    """Raised when uploaded bytes are not a readable PDF."""


class EncryptedPdfError(IngestionError):
    """Raised when a PDF requires a password."""


class NoUsablePdfTextError(IngestionError):
    """Raised when a PDF has too little extractable text for this OCR-free MVP."""


class TopicIngestionError(IngestionError):
    """Base error for Wikipedia topic search and article retrieval."""


class InvalidTopicError(TopicIngestionError):
    """Raised when a requested topic is empty or too short to search safely."""


class NoTopicSearchResultsError(TopicIngestionError):
    """Raised when Wikipedia cannot find a matching article."""


class AmbiguousTopicError(TopicIngestionError):
    """Raised when a topic requires an explicit article selection."""


class TopicNetworkError(TopicIngestionError):
    """Raised when Wikipedia cannot be reached or a request times out."""


class TopicApiError(TopicIngestionError):
    """Raised when Wikipedia returns an HTTP, API, or malformed response."""


class DisambiguationPageError(TopicIngestionError):
    """Raised when the selected page is a Wikipedia disambiguation page."""


class NoUsableTopicTextError(TopicIngestionError):
    """Raised when a Wikipedia article has too little readable material for a quiz."""


class NltkResourceError(QuestLLMError):
    """Raised when local NLTK tokenizer data has not been installed."""


class ModelLoadError(QuestLLMError):
    """Raised when QuestLLM cannot load its local question-generation model."""


class GenerationError(QuestLLMError):
    """Raised when a grounded question cannot be built or generated safely."""


class WorkflowError(QuestLLMError):
    """Raised when an invalid quiz workflow transition is requested."""
