import json
from pathlib import Path
from collections import defaultdict
import re


CHUNKS_FILE = Path("data/index/chunks.json")
METADATA_FILE = Path("data/index/document_metadata.json")


def extract_keywords(text, limit=15):
    """
    Extract frequently occurring meaningful words from document text.
    """

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stop_words = {
        "this", "that", "with", "from", "have",
        "will", "which", "their", "there", "about",
        "into", "these", "those", "also", "than",
        "then", "they", "them", "were", "been",
        "being", "such", "using", "used", "where",
        "when", "what", "your", "more", "some",
        "other", "each", "only", "very", "here"
    }

    words = [
        word for word in words
        if word not in stop_words
    ]

    frequency = {}

    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    sorted_words = sorted(
        frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        word
        for word, count in sorted_words[:limit]
    ]


def generate_metadata():

    print("Loading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    documents = defaultdict(list)

    # Group chunks by document
    for chunk in chunks:
        documents[chunk["filename"]].append(chunk)

    metadata = {}

    for filename, document_chunks in documents.items():

        # Combine all text belonging to this document
        full_text = " ".join(
            chunk["text"]
            for chunk in document_chunks
        )

        # Determine page information
        pages = sorted(
            set(
                chunk["page_number"]
                for chunk in document_chunks
            )
        )

        metadata[filename] = {
            "filename": filename,
            "total_chunks": len(document_chunks),
            "total_pages": len(pages),
            "pages": pages,
            "keywords": extract_keywords(full_text),
            "text_length": len(full_text)
        }

    # Save metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nMetadata generation complete!")
    print(f"Documents processed: {len(metadata)}")
    print(f"Saved to: {METADATA_FILE}")


if __name__ == "__main__":
    generate_metadata()