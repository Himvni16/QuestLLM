"""Offline unit tests for PDF ingestion."""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from questllm.document import SourceType
from questllm.exceptions import (
    EncryptedPdfError,
    InvalidPdfError,
    NoUsablePdfTextError,
    SourceTooLargeError,
)
from questllm.ingestion import pdf as pdf_ingestion
from questllm.ingestion.pdf import ingest_pdf


def _escape_pdf_text(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")


def _make_text_pdf(page_texts: list[str | None]) -> bytes:
    """Build a tiny text PDF fixture without needing a PDF-generation dependency."""

    page_numbers = [3 + index * 2 for index in range(len(page_texts))]
    font_number = 3 + len(page_texts) * 2
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Kids ["
            + b" ".join(f"{number} 0 R".encode() for number in page_numbers)
            + f"] /Count {len(page_texts)} >>".encode()
        ),
        font_number: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, page_text in enumerate(page_texts):
        page_number = page_numbers[index]
        resources = f"/Resources << /Font << /F1 {font_number} 0 R >> >>".encode()
        content_number = page_number + 1
        stream = b"BT /F1 12 Tf 72 720 Td (" + _escape_pdf_text(page_text or "") + b") Tj ET"
        objects[content_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        objects[page_number] = (
            b"<< /Type /Page /Parent 2 0 R "
            + resources
            + f" /MediaBox [0 0 612 792] /Contents {content_number} 0 R >>".encode()
        )

    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number in range(1, max(objects) + 1):
        offsets.append(len(document))
        document.extend(f"{object_number} 0 obj\n".encode())
        document.extend(objects[object_number])
        document.extend(b"\nendobj\n")

    xref_offset = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    document.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    document.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(document)


def test_ingest_pdf_extracts_text_and_metadata() -> None:
    content = _make_text_pdf(
        ["This PDF has enough extracted text to support a later quiz generation step."]
    )

    document = ingest_pdf(content, filename="study-notes.pdf")

    assert document.source_type is SourceType.PDF
    assert document.page_count == 1
    assert document.pages[0].page_number == 1
    assert "enough extracted text" in document.text
    assert document.metadata == {"page_count": "1", "filename": "study-notes.pdf"}


def test_ingest_pdf_preserves_multiple_page_numbers() -> None:
    content = _make_text_pdf(
        [
            "The first page contains enough text to be a usable source for a later quiz.",
            "The second page adds another clear statement for processing and provenance.",
        ]
    )

    document = ingest_pdf(BytesIO(content))

    assert [page.page_number for page in document.pages] == [1, 2]
    assert "first page" in document.pages[0].text
    assert "second page" in document.pages[1].text
    assert "\n\n" in document.text


def test_ingest_pdf_keeps_empty_pages_without_including_them_in_full_text() -> None:
    content = _make_text_pdf(
        [
            "This first page has enough readable text for ingestion even if another page is blank.",
            None,
        ]
    )

    document = ingest_pdf(content)

    assert document.page_count == 2
    assert document.pages[1].text == ""
    assert "another page is blank" in document.text


def test_ingest_pdf_rejects_files_without_meaningful_extractable_text() -> None:
    with pytest.raises(NoUsablePdfTextError):
        ingest_pdf(_make_text_pdf([None]))


@pytest.mark.parametrize("content", [b"this is not a PDF", b"%PDF-1.4\nbroken document"])
def test_ingest_pdf_rejects_invalid_or_corrupt_content(content: bytes) -> None:
    with pytest.raises(InvalidPdfError):
        ingest_pdf(content)


def test_ingest_pdf_rejects_encrypted_content() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    encrypted_pdf = BytesIO()
    writer.write(encrypted_pdf)

    with pytest.raises(EncryptedPdfError):
        ingest_pdf(encrypted_pdf.getvalue())


def test_ingest_pdf_rejects_files_over_the_byte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    content = _make_text_pdf(
        ["This PDF has enough extracted text to support a later quiz generation step."]
    )
    monkeypatch.setattr(pdf_ingestion, "MAXIMUM_PDF_FILE_SIZE_BYTES", len(content) - 1)

    with pytest.raises(SourceTooLargeError, match="too large"):
        ingest_pdf(content)


def test_ingest_pdf_rejects_files_over_the_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_ingestion, "MAXIMUM_PDF_PAGE_COUNT", 1)

    with pytest.raises(SourceTooLargeError, match="too many pages"):
        ingest_pdf(
            _make_text_pdf(
                [
                    "The first page contains enough readable source text for a quiz.",
                    "The second page also contains enough readable source text for a quiz.",
                ]
            )
        )
