"""
Extract text from a PDF file.

Uses pdfplumber (accurate text layer extraction).
Scanned/image-only PDFs will return empty text — those require OCR.
"""
from __future__ import annotations

from pathlib import Path


def extract_pdf_text(pdf_path: str) -> str:
    """
    Return all text from the PDF concatenated page-by-page.
    Raises ImportError if pdfplumber is not installed.
    """
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                parts.append(text.strip())

    if not parts:
        raise ValueError(
            "No extractable text found in PDF. "
            "The file may be a scanned image — OCR is not currently supported."
        )

    return "\n\n".join(parts)


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Return basic metadata (page count, estimated word count)."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        sample = " ".join(
            (pdf.pages[i].extract_text() or "")
            for i in range(min(3, page_count))
        )
        words_per_page_est = len(sample.split()) / max(1, min(3, page_count))

    return {
        "page_count": page_count,
        "estimated_words": int(words_per_page_est * page_count),
    }
