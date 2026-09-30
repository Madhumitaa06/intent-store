# ================================================================
# INTENTSTORE - SEARCH ENGINE
# ================================================================

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from auth import get_user_documents

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

        self.data_dir = self._find_data_directory()

        print("\nDATA DIRECTORY:")
        print(self.data_dir)

        self.index_dir = self._find_index_directory()

        print("\nINDEX DIRECTORY:")
        print(self.index_dir)

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

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded.")

        print("\nLoading chunks...")

        with open(
            self.chunks_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

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

        if (
            self.index_file
            and self.index_file.exists()
        ):

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

        for directory in candidates:

            if (
                directory.exists()
                and (
                    (directory / "chunks.json").exists()
                    or (directory / "faiss.index").exists()
                )
            ):

                return directory

        for directory in candidates:

            if directory.exists():
                return directory

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
            self.index_dir / filename,
            BACKEND_DIR / "data" / filename,
            PROJECT_ROOT / "data" / filename,
        ]

        for path in candidates:

            if (
                path.exists()
                and path.is_file()
            ):

                return path

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

                for match in matches:

                    if (
                        match.parent.name.lower()
                        == "index"
                    ):

                        return match

                return matches[0]

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
    # GET FILENAME FROM CHUNK
    # ============================================================

    def _get_chunk_filename(self, chunk):

        if not isinstance(chunk, dict):
            return None

        # --------------------------------------------------------
        # Direct filename fields
        # --------------------------------------------------------

        possible_fields = [
            "filename",
            "file_name",
            "document",
            "document_name",
            "source",
            "file",
            "path",
            "document_path"
        ]

        for field in possible_fields:

            value = chunk.get(field)

            if value:

                value = str(value).strip()

                if value:

                    return Path(value).name

        # --------------------------------------------------------
        # Nested metadata
        # --------------------------------------------------------

        metadata = chunk.get("metadata")

        if isinstance(metadata, dict):

            for field in possible_fields:

                value = metadata.get(field)

                if value:

                    value = str(value).strip()

                    if value:

                        return Path(value).name

        # --------------------------------------------------------
        # Nested document information
        # --------------------------------------------------------

        document_info = chunk.get("document_info")

        if isinstance(document_info, dict):

            for field in possible_fields:

                value = document_info.get(field)

                if value:

                    value = str(value).strip()

                    if value:

                        return Path(value).name

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
        top_k=5,
        user_id=None
    ):

        query = str(
            query
        ).strip()

        if not query:
            return []

        if self.index.ntotal == 0:
            return []

        # --------------------------------------------------------
        # SECURITY
        # --------------------------------------------------------

        if user_id is None:

            print(
                "SECURITY: Search rejected because "
                "no user_id was provided."
            )

            return []

        # --------------------------------------------------------
        # USER DOCUMENTS
        # --------------------------------------------------------

        user_documents = get_user_documents(
            user_id
        )

        print(
            f"USER {user_id} DOCUMENTS:",
            user_documents
        )

        if not user_documents:

            print(
                f"No documents assigned to user {user_id}."
            )

            return []

        # --------------------------------------------------------
        # QUERY EMBEDDING
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
        # FAISS SEARCH
        # --------------------------------------------------------

        actual_k = self.index.ntotal

        scores, indices = self.index.search(
            query_embedding,
            actual_k
        )

        # --------------------------------------------------------
        # RESULTS
        # --------------------------------------------------------

        results = []

        debug_filenames = set()

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

            # ----------------------------------------------------
            # ROBUST FILENAME DETECTION
            # ----------------------------------------------------

            filename = self._get_chunk_filename(
                chunk
            )

            if filename:
                debug_filenames.add(filename)

            if not filename:

                continue

            # ----------------------------------------------------
            # OWNERSHIP FILTER
            # ----------------------------------------------------

            if filename not in user_documents:
                continue

            result = {
                "filename": filename,

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

            if len(results) >= int(top_k):
                break

        print(
            "FILENAMES FOUND IN SEARCH RESULTS:",
            debug_filenames
        )

        # --------------------------------------------------------
        # RANKING
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
        # REMOVE DUPLICATES
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

            if len(unique) >= int(top_k):
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

    print(
        "\nFor this test, enter your user ID."
    )

    user_id = input(
        "User ID: "
    ).strip()

    try:

        user_id = int(user_id)

    except ValueError:

        print("Invalid user ID.")
        raise SystemExit

    results = engine.search(
        query,
        top_k=5,
        user_id=user_id
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
