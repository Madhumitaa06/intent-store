# ================================================================
# INTENTSTORE - DOCUMENT PARSER
# ================================================================
#
# Handles:
#   - Text PDFs            -> extracted directly (free, fast)
#   - Scanned / handwritten PDF pages -> rendered to an image and
#                             transcribed with Claude vision
#   - Image files (jpg, png, webp) -> transcribed with Claude vision
#
# Transcriptions are cached in data/index/ocr_cache/ so re-running
# this script never pays for the same page twice.
#
# SETUP:
#   pip install anthropic pymupdf python-dotenv
#   ANTHROPIC_API_KEY in backend/.env or Codespaces secrets
# ================================================================

import base64
import json
import uuid
import os
import time
from pathlib import Path

import fitz  # PyMuPDF

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path("backend/.env"))
except Exception:
    pass

import anthropic


DOCUMENTS_FOLDER = Path("data/documents")
OUTPUT_FOLDER = Path("data/index")
OUTPUT_FILE = OUTPUT_FOLDER / "documents.json"
OCR_CACHE_FOLDER = OUTPUT_FOLDER / "ocr_cache"

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# A page with fewer characters than this is treated as "no text layer"
# (scanned or handwritten) and sent to vision transcription.
MIN_TEXT_CHARS = 30

# Model used for transcription. Override with CLAUDE_OCR_MODEL in .env.
OCR_MODEL = os.getenv("CLAUDE_OCR_MODEL", "claude-sonnet-5-5")

# Render zoom: 2.0 is about 144 DPI, a good balance for handwriting.
RENDER_ZOOM = 2.0

TRANSCRIPTION_PROMPT = (
    "Transcribe all the text on this page exactly as written. "
    "The page may be handwritten notes, so read carefully.\n\n"
    "Rules:\n"
    "- Preserve the structure: line breaks, bullet points, numbering, "
    "headings, and tables (use simple text for tables).\n"
    "- If a word is unreadable, write [illegible]. Do not guess wildly.\n"
    "- For diagrams, sketches or drawings, add a short description "
    "in square brackets, e.g. [diagram: flowchart from A to B].\n"
    "- Do not add commentary, explanations, or a preamble.\n"
    "- If the page is blank, reply with nothing."
)


_client = None


def get_client():
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY is not set (needed to read "
                "handwritten / scanned pages)."
            )
        _client = anthropic.Anthropic()
    return _client


# ================================================================
# VISION TRANSCRIPTION
# ================================================================

def transcribe_image_bytes(image_bytes, media_type="image/jpeg", retries=3):
    """Send one page image to Claude and return the transcribed text."""

    client = get_client()
    encoded = base64.standard_b64encode(image_bytes).decode("utf-8")

    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=OCR_MODEL,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    ],
                }],
            )

            return "".join(
                block.text for block in response.content
                if block.type == "text"
            ).strip()

        except Exception as error:
            print(f"    Transcription attempt {attempt} failed: {error}")
            if attempt == retries:
                return ""
            time.sleep(2 * attempt)

    return ""


def render_page_to_jpeg(page):
    """Render a PyMuPDF page to JPEG bytes."""
    matrix = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("jpeg", jpg_quality=85)


def ocr_page_with_cache(page, cache_key):
    """Transcribe a page, using the on-disk cache when available."""

    OCR_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    cache_file = OCR_CACHE_FOLDER / f"{cache_key}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    text = transcribe_image_bytes(render_page_to_jpeg(page))

    # Only cache successful transcriptions, so failures are retried
    # on the next run.
    if text:
        cache_file.write_text(text, encoding="utf-8")

    return text


# ================================================================
# FILE EXTRACTION (PDFs and images share one path)
# ================================================================

def extract_pages(file_path):
    """
    Works for PDFs and image files (PyMuPDF opens both; an image
    is simply a one-page document with no text layer).
    Returns (pages, page_count).
    """

    document = fitz.open(file_path)
    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text().strip()
        method = "text"

        if len(text) < MIN_TEXT_CHARS:
            print(f"  Page {page_number}: no text layer, "
                  f"transcribing with vision...")
            cache_key = f"{file_path.stem}_p{page_number}"
            transcribed = ocr_page_with_cache(page, cache_key)

            # Keep whichever result has more content
            if len(transcribed) > len(text):
                text = transcribed
                method = "vision"

        pages.append({
            "page_number": page_number,
            "text": text,
            "extraction_method": method,
        })

    page_count = len(document)
    document.close()

    return pages, page_count


# ================================================================
# MAIN
# ================================================================

def process_all_documents():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_documents = []

    supported = PDF_EXTENSIONS | IMAGE_EXTENSIONS
    files = sorted(
        path for path in DOCUMENTS_FOLDER.iterdir()
        if path.suffix.lower() in supported
    )

    print(f"Found {len(files)} documents.\n")

    for file_path in files:

        print(f"Processing: {file_path.name}")

        try:
            pages, page_count = extract_pages(file_path)
        except Exception as error:
            print(f"  FAILED to read {file_path.name}: {error}")
            continue

        file_type = (
            "PDF" if file_path.suffix.lower() in PDF_EXTENSIONS
            else "IMAGE"
        )

        vision_pages = sum(
            1 for p in pages if p["extraction_method"] == "vision"
        )

        all_documents.append({
            "filename": file_path.name,
            "document_id": str(uuid.uuid4()),
            "file_type": file_type,
            "page_count": page_count,
            "pages": pages,
        })

        print(f"  Pages extracted: {page_count} "
              f"({vision_pages} via vision)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_documents, file, ensure_ascii=False, indent=2)

    print("\n--------------------------------")
    print("DOCUMENT INGESTION COMPLETE")
    print("--------------------------------")
    print(f"Documents processed: {len(all_documents)}")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_documents()