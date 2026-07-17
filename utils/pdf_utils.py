"""
pdf_utils.py
------------
Extracts text from PDFs - and this version tracks WHICH PAGE each piece
of text came from as structured data (not just a "[Page 3]" text marker
buried inside the string). That structure is what lets us later tell the
user "this answer came from page 3 of file.pdf" instead of just the
filename.
"""

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when an uploaded file cannot be read as a usable PDF."""


def extract_pages_from_pdf(pdf_file):
    """
    Extracts text from a single PDF, page by page.

    Returns:
        list of dicts: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        (pages with no extractable text, e.g. pure-image pages, are skipped)
    """
    pages = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append({"page": page_number, "text": page_text})
    except Exception as error:
        raise PDFExtractionError(
            f"Could not read '{getattr(pdf_file, 'name', 'uploaded file')}' as a PDF."
        ) from error

    return pages


def extract_text_from_multiple_pdfs(pdf_files):
    """
    Takes a LIST of uploaded PDF files and returns a dictionary mapping
    each filename to its list of {"page": n, "text": ...} entries.

    Returns:
        dict: { "filename.pdf": [{"page": 1, "text": "..."}, ...], ... }
    """
    all_documents = {}

    for pdf_file in pdf_files:
        pages = extract_pages_from_pdf(pdf_file)
        if not pages:
            raise PDFExtractionError(
                f"'{pdf_file.name}' contains no extractable text. "
                "Use a text-based PDF or add OCR support for scanned documents."
            )
        all_documents[pdf_file.name] = pages

    return all_documents
