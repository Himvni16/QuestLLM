"""PDF ingestion for text-based PDFs; OCR is intentionally out of scope."""

from io import BytesIO
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from questllm.config import MINIMUM_TEXT_CHARACTERS
from questllm.document import Document, DocumentPage, SourceType
from questllm.exceptions import EncryptedPdfError, InvalidPdfError, NoUsablePdfTextError
from questllm.ingestion.text import normalize_text


def _read_pdf_bytes(source: bytes | BinaryIO) -> bytes:
    """Return uploaded PDF content as bytes for safe, repeatable parsing."""

    if isinstance(source, bytes):
        return source

    try:
        source.seek(0)
        content = source.read()
    except (AttributeError, OSError, ValueError) as error:
        raise InvalidPdfError("The uploaded file could not be read as a PDF.") from error

    if not isinstance(content, bytes):
        raise InvalidPdfError("The uploaded file must provide binary PDF content.")
    return content


def ingest_pdf(source: bytes | BinaryIO, *, filename: str | None = None) -> Document:
    """Extract page-aware text from a readable, unencrypted PDF upload."""

    content = _read_pdf_bytes(source)
    if not content or b"%PDF-" not in content[:1024]:
        raise InvalidPdfError("The uploaded file is not a valid PDF.")

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise EncryptedPdfError("Password-protected PDFs are not supported yet.")

        pages = tuple(
            DocumentPage(
                page_number=page_number,
                text=normalize_text(page.extract_text() or "", repair_hyphenation=True),
            )
            for page_number, page in enumerate(reader.pages, start=1)
        )
    except EncryptedPdfError:
        raise
    except (OSError, PdfReadError, ValueError) as error:
        raise InvalidPdfError("QuestLLM could not read this PDF.") from error

    extracted_text = "\n\n".join(page.text for page in pages if page.text)
    if len(extracted_text) < MINIMUM_TEXT_CHARACTERS:
        raise NoUsablePdfTextError(
            "This PDF has little or no extractable text. It may be scanned or image-only; "
            "OCR is not available in this version of QuestLLM."
        )

    metadata = {"page_count": str(len(pages))}
    if filename:
        metadata["filename"] = filename

    return Document(
        source_type=SourceType.PDF,
        text=extracted_text,
        pages=pages,
        metadata=metadata,
    )
