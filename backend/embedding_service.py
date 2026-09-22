from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np


INPUT_FILE = Path("data/index/chunks.json")
OUTPUT_FILE = Path("data/index/embeddings.npy")

MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Load chunks
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Loaded {len(chunks)} text chunks.")

    # Extract only the text
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings)

    # Save embeddings
    np.save(OUTPUT_FILE, embeddings)

    print("\n-------------------------------")
    print("EMBEDDING GENERATION COMPLETE")
    print("-------------------------------")
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape[1])
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_embeddings()