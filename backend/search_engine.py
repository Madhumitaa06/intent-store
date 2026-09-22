# ================================================================
# INTENTSTORE - SEARCH ENGINE
# ================================================================

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from relevance_ranker import rank_results
except Exception:
    rank_results = None


# ================================================================
# PATHS
# ================================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MODEL_NAME = "all-MiniLM-L6-v2"


# ================================================================
# SEARCH ENGINE
# ================================================================

class IntentStoreSearch:

    def __init__(self):

        print("\n" + "=" * 70)
        print("INITIALIZING INTENTSTORE SEARCH ENGINE")
        print("=" * 70)

        # --------------------------------------------------------
        # Locate data
        # --------------------------------------------------------

        self.data_dir = self._find_data_directory()

        print("\nDATA DIRECTORY:")
        print(self.data_dir)

        self.index_dir = self._find_index_directory()

        print("\nINDEX DIRECTORY:")
        print(self.index_dir)

        # --------------------------------------------------------
        # Locate files
        # --------------------------------------------------------

        self.chunks_file = self._find_file(
            "chunks.json",
            required=True
        )

        self.metadata_file = self._find_file(
            "document_metadata.json",
            required=False
        )

        self.index_file = self._find_file(
            "faiss.index",
            required=False
        )

        print("\nFILES FOUND")

        print("chunks.json:")
        print(self.chunks_file)

        print("document_metadata.json:")
        print(self.metadata_file)

        print("faiss.index:")
        print(self.index_file)

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded.")

        # --------------------------------------------------------
        # Load chunks
        # --------------------------------------------------------

        print("\nLoading chunks...")

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        # Handle dictionary format
        if isinstance(self.chunks, dict):

            if "chunks" in self.chunks:

                self.chunks = self.chunks["chunks"]

            else:

                self.chunks = list(
                    self.chunks.values()
                )

        print(
            f"Loaded {len(self.chunks)} chunks."
        )

        if len(self.chunks) == 0:

            raise ValueError(
                "chunks.json exists but contains no chunks."
            )

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        self.document_metadata = {}

        if self.metadata_file:

            try:

                with open(
                    self.metadata_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.document_metadata = json.load(file)

                print(
                    f"Loaded {len(self.document_metadata)} metadata entries."
                )

            except Exception as error:

                print(
                    "WARNING: Could not load metadata:"
                )

                print(error)

        # --------------------------------------------------------
        # Load / build FAISS
        # --------------------------------------------------------

        if self.index_file and self.index_file.exists():

            try:

                print("\nLoading FAISS index...")

                self.index = faiss.read_index(
                    str(self.index_file)
                )

                print(
                    f"FAISS index loaded: "
                    f"{self.index.ntotal} vectors"
                )

            except Exception as error:

                print(
                    "\nCould not load FAISS index."
                )

                print(error)

                print(
                    "\nRebuilding index..."
                )

                self._build_index()

        else:

            print(
                "\nFAISS index not found."
            )

            print(
                "Building index automatically..."
            )

            self._build_index()

        # --------------------------------------------------------
        # Validate index
        # --------------------------------------------------------

        if self.index.ntotal != len(self.chunks):

            print("\nIndex/chunk count mismatch.")

            print(
                f"FAISS vectors : {self.index.ntotal}"
            )

            print(
                f"Chunks        : {len(self.chunks)}"
            )

            print(
                "\nRebuilding FAISS index..."
            )

            self._build_index()

        # --------------------------------------------------------
        # READY
        # --------------------------------------------------------

        print("\n" + "=" * 70)
        print("INTENTSTORE SEARCH ENGINE READY")
        print("=" * 70)

        print(
            "Vectors:",
            self.index.ntotal
        )

        print(
            "Chunks:",
            len(self.chunks)
        )

        print("=" * 70)

    # ============================================================
    # FIND DATA DIRECTORY
    # ============================================================

    def _find_data_directory(self):

        candidates = [

            BACKEND_DIR / "data",

            PROJECT_ROOT / "data",

        ]

        for directory in candidates:

            if directory.exists():

                return directory

        # Create default backend/data
        directory = BACKEND_DIR / "data"

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return directory

    # ============================================================
    # FIND INDEX DIRECTORY
    # ============================================================

    def _find_index_directory(self):

        candidates = [

            BACKEND_DIR / "data" / "index",

            PROJECT_ROOT / "data" / "index",

        ]

        # First priority: directory containing chunks
        for directory in candidates:

            if (
                directory.exists()
                and (
                    (directory / "chunks.json").exists()
                    or (directory / "faiss.index").exists()
                )
            ):

                return directory

        # Return existing directory
        for directory in candidates:

            if directory.exists():

                return directory

        # Create standard location
        directory = BACKEND_DIR / "data" / "index"

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return directory

    # ============================================================
    # FIND FILE
    # ============================================================

    def _find_file(
        self,
        filename,
        required=True
    ):

        candidates = [

            # Index directory
            self.index_dir / filename,

            # Backend data
            BACKEND_DIR / "data" / filename,

            # Project data
            PROJECT_ROOT / "data" / filename,

        ]

        # --------------------------------------------------------
        # Direct candidates
        # --------------------------------------------------------

        for path in candidates:

            if path.exists() and path.is_file():

                return path

        # --------------------------------------------------------
        # Recursive search
        # --------------------------------------------------------

        search_roots = [

            BACKEND_DIR,
            PROJECT_ROOT,

        ]

        for root in search_roots:

            try:

                matches = list(
                    root.rglob(filename)
                )

            except Exception:

                continue

            if matches:

                # Prefer index folder for index-related files
                for match in matches:

                    if match.parent.name.lower() == "index":

                        return match

                return matches[0]

        # --------------------------------------------------------
        # Missing
        # --------------------------------------------------------

        if required:

            raise FileNotFoundError(

                f"\nRequired file '{filename}' was not found.\n\n"

                f"Searched in:\n"

                f"{BACKEND_DIR}\n"

                f"{PROJECT_ROOT}\n\n"

                f"Please make sure the dataset files exist."

            )

        return None

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def _build_index(self):

        if not self.chunks:

            raise ValueError(
                "No document chunks available."
            )

        print(
            f"\nCreating embeddings for "
            f"{len(self.chunks)} chunks..."
        )

        texts = []

        for chunk in self.chunks:

            if isinstance(chunk, dict):

                text = str(
                    chunk.get(
                        "text",
                        ""
                    )
                )

            else:

                text = str(chunk)

            texts.append(text)

        embeddings = self.model.encode(

            texts,

            normalize_embeddings=True,

            show_progress_bar=True

        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            self.index_dir / "faiss.index"
        )

        faiss.write_index(
            self.index,
            str(output_file)
        )

        self.index_file = output_file

        print("\nFAISS INDEX CREATED")

        print(
            "Vectors:",
            self.index.ntotal
        )

        print(
            "Saved to:",
            output_file
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        top_k=5
    ):

        query = str(
            query
        ).strip()

        if not query:

            return []

        if self.index.ntotal == 0:

            return []

        # --------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------

        query_embedding = self.model.encode(

            [query],

            normalize_embeddings=True

        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # --------------------------------------------------------
        # FAISS
        # --------------------------------------------------------

        actual_k = min(
            int(top_k),
            self.index.ntotal
        )

        scores, indices = self.index.search(

            query_embedding,

            actual_k

        )

        # --------------------------------------------------------
        # Results
        # --------------------------------------------------------

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:

                continue

            if idx >= len(self.chunks):

                continue

            chunk = self.chunks[idx]

            if not isinstance(chunk, dict):

                chunk = {
                    "text": str(chunk)
                }

            result = {

                "filename": chunk.get(
                    "filename",
                    chunk.get(
                        "document",
                        "Unknown document"
                    )
                ),

                "page_number": chunk.get(
                    "page_number",
                    chunk.get(
                        "page",
                        "Unknown"
                    )
                ),

                "chunk_number": chunk.get(
                    "chunk_number",
                    chunk.get(
                        "chunk_id",
                        idx
                    )
                ),

                "similarity_score": round(
                    float(score),
                    4
                ),

                "text": chunk.get(
                    "text",
                    ""
                ),

            }

            results.append(result)

        # --------------------------------------------------------
        # Ranking
        # --------------------------------------------------------

        if (
            results
            and rank_results is not None
        ):

            try:

                results = rank_results(

                    query,

                    results,

                    self.document_metadata

                )

            except Exception as error:

                print(
                    "Ranking warning:",
                    error
                )

        # --------------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------------

        unique = []

        seen = set()

        for result in results:

            text = str(
                result.get(
                    "text",
                    ""
                )
            )

            key = " ".join(
                text.lower().split()
            )

            if key in seen:

                continue

            seen.add(key)

            unique.append(result)

            if len(unique) >= 5:

                break

        return unique


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    engine = IntentStoreSearch()

    print("\nSearch engine is working.")

    query = input(
        "\nEnter query: "
    )

    results = engine.search(
        query,
        top_k=5
    )

    print("\nRESULTS")

    for i, result in enumerate(
        results,
        1
    ):

        print(
            f"\n{i}. "
            f"{result['filename']}"
        )

        print(
            "Page:",
            result["page_number"]
        )

        print(
            "Score:",
            result["similarity_score"]
        )

        print(
            result["text"][:500]
        )