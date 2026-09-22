import json
from pathlib import Path


INPUT_FILE = Path("data/index/documents.json")
OUTPUT_FILE = Path("data/index/chunks.json")


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    all_chunks = []

    chunk_id = 0

    for document in documents:

        filename = document["filename"]

        for page in document["pages"]:

            page_number = page["page_number"]
            text = page["text"].strip()

            if not text:
                continue

            page_chunks = split_text(text)

            for chunk_number, chunk_text in enumerate(
                page_chunks, start=1
            ):

                chunk_id += 1

                chunk_data = {
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text
                }

                all_chunks.append(chunk_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("--------------------------------")
    print("TEXT CHUNKING COMPLETE")
    print("--------------------------------")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_chunks()