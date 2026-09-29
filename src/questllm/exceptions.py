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
