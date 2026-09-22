import faiss
import numpy as np
import json
from pathlib import Path


EMBEDDINGS_FILE = Path("data/index/embeddings.npy")
CHUNKS_FILE = Path("data/index/chunks.json")
INDEX_FILE = Path("data/index/faiss.index")


def build_faiss_index():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embeddings shape:", embeddings.shape)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Add embeddings
    index.add(embeddings.astype("float32"))

    # Save index
    faiss.write_index(index, str(INDEX_FILE))

    # Load chunks to verify alignment
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("\n-------------------------------")
    print("FAISS INDEX CREATED")
    print("-------------------------------")
    print("Vectors in index:", index.ntotal)
    print("Chunks available:", len(chunks))
    print("Vector dimensions:", dimension)
    print("Index saved to:", INDEX_FILE)


if __name__ == "__main__":
    build_faiss_index()