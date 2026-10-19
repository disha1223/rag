
import hashlib
import re
from pathlib import Path

import fitz


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_paper_id(pdf_path: str) -> str:
    """Generate a unique ID from the PDF contents."""
    content = Path(pdf_path).read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def parse_pdf(pdf_path: str):
    """Extract text from a PDF, preserving page numbers."""
    paragraphs = []

    with fitz.open(pdf_path) as document:
        title = (document.metadata or {}).get("title", "").strip()
        title = title or Path(pdf_path).stem

        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)

            for block in blocks:
                text = clean_text(block[4])

                if len(text) >= 25:
                    paragraphs.append({
                        "text": text,
                        "page": page_number
                    })

    return {
        "paper_id": get_paper_id(pdf_path),
        "title": title,
        "paragraphs": paragraphs
    }
