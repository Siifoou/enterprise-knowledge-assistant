"""
Tests for extraction.py: reading Word/PDF files, keeping document order,
and rejecting files that contain no real text (e.g. scanned pages).
Word files are built on the fly; the two PDFs live in tests/fixtures/.
"""
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter

from extraction import EmptyDocumentError, extract_text

FIXTURES = Path(__file__).parent / "fixtures"


def make_docx_with_table(path):
    doc = Document()
    doc.add_paragraph("Vacation Policy. Employees accrue paid leave depending on their length of service.")
    table = doc.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([
        ("Years of Service", "Vacation Days"),
        ("0-2 years", "18 days"),
        ("5+ years", "27 days"),
    ]):
        table.cell(r, 0).text = a
        table.cell(r, 1).text = b
    doc.add_paragraph("Unused days may be carried over, up to a maximum of five days per year.")
    doc.save(path)


def test_docx_keeps_document_order(tmp_path):
    """Regression: tables used to be moved to the end of the text."""
    path = tmp_path / "policy.docx"
    make_docx_with_table(path)
    text = extract_text(str(path))
    assert (
        text.index("Vacation Policy")
        < text.index("[[TABLE_START]]")
        < text.index("[[TABLE_END]]")
        < text.index("Unused days")
    )


def test_docx_table_rows_are_readable(tmp_path):
    path = tmp_path / "policy.docx"
    make_docx_with_table(path)
    text = extract_text(str(path))
    assert "5+ years | 27 days" in text


def test_docx_with_almost_no_text_is_rejected(tmp_path):
    path = tmp_path / "tiny.docx"
    doc = Document()
    doc.add_paragraph("Hi")
    doc.save(path)
    with pytest.raises(EmptyDocumentError):
        extract_text(str(path))


def test_pdf_with_a_text_layer_is_extracted():
    text = extract_text(str(FIXTURES / "sample_text.pdf"))
    assert "22 vacation days" in text


def test_scanned_pdf_is_rejected():
    """An image-only PDF has no text layer: it must be refused, not silently indexed as empty."""
    with pytest.raises(EmptyDocumentError, match="scanned"):
        extract_text(str(FIXTURES / "scanned_image_only.pdf"))


def test_blank_pdf_is_rejected(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)
    with pytest.raises(EmptyDocumentError):
        extract_text(str(path))


def test_unsupported_file_type_is_refused():
    with pytest.raises(ValueError):
        extract_text("notes.txt")