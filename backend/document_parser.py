import fitz
from pathlib import Path
import json


DOCUMENTS_FOLDER = Path("data/documents")
OUTPUT_FOLDER = Path("data/index")
OUTPUT_FILE = OUTPUT_FOLDER / "documents.json"


def extract_pages_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        pages.append({
            "page_number": page_number,
            "text": text
        })

    page_count = len(document)

    document.close()

    return pages, page_count


def process_all_documents():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_documents = []

    pdf_files = sorted(DOCUMENTS_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF documents.\n")

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        pages, page_count = extract_pages_from_pdf(pdf_path)

        document_data = {
            "filename": pdf_path.name,
            "file_type": "PDF",
            "page_count": page_count,
            "pages": pages
        }

        all_documents.append(document_data)

        print(f"  Pages extracted: {page_count}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_documents, file, ensure_ascii=False, indent=2)

    print("\n--------------------------------")
    print("DOCUMENT INGESTION COMPLETE")
    print("--------------------------------")
    print(f"Documents processed: {len(all_documents)}")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_documents()