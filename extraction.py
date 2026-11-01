from pypdf import PdfReader
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

MIN_TEXT_LENGTH = 50  # a real document should have at least this many characters


class EmptyDocumentError(Exception):
    """Raised when a file appears to have no meaningful extractable text —
    most commonly a scanned/image-only PDF, or a Word doc containing only
    an embedded image instead of real text."""
    pass


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    text_parts = []

    # Walk the document body in its real order, not paragraphs-then-tables
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            if para.text.strip():
                text_parts.append(para.text)

        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            table_lines = ["[[TABLE_START]]"]
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                table_lines.append(row_text)
            table_lines.append("[[TABLE_END]]")
            text_parts.append("\n".join(table_lines))

    return "\n".join(text_parts)


def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if len(text) < MIN_TEXT_LENGTH:
        raise EmptyDocumentError(
            f"'{file_path}' returned almost no extractable text "
            f"({len(text)} characters). This likely means it's a scanned/"
            f"image-only document, which this system cannot process."
        )

    return text


if __name__ == "__main__":
    text = extract_text("data/documents/vacation_policy.docx")
    print(text)
    print("\n---\n")
    print(f"Total characters extracted: {len(text)}")