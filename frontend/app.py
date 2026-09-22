# ================================================================
# INTENTSTORE FRONTEND + BACKEND CONNECTOR
# frontend/app.py
# ================================================================

import sys
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


# ================================================================
# PATHS
# ================================================================

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = BACKEND_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
DOCUMENTS_DIR = DATA_DIR / "documents"


# Add backend to Python path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ================================================================
# BACKEND IMPORTS
# ================================================================

RAGEngine = None
IntentStoreSearch = None

try:
    from rag_engine import RAGEngine
    print("RAGEngine imported successfully.")
except Exception as e:
    print("RAGEngine import failed:", e)

try:
    from search_engine import IntentStoreSearch
    print("IntentStoreSearch imported successfully.")
except Exception as e:
    print("SearchEngine import failed:", e)


# ================================================================
# FLASK
# ================================================================

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)


# ================================================================
# BACKEND INITIALIZATION
# ================================================================

rag_engine = None
search_engine = None


def initialize_backend():

    global rag_engine
    global search_engine

    print("\n" + "=" * 70)
    print("INTENTSTORE BACKEND INITIALIZATION")
    print("=" * 70)

    print("\nPROJECT ROOT:")
    print(PROJECT_ROOT)

    print("\nBACKEND:")
    print(BACKEND_DIR)

    print("\nDATA:")
    print(DATA_DIR)

    print("\nINDEX:")
    print(INDEX_DIR)

    print("\nDOCUMENTS:")
    print(DOCUMENTS_DIR)

    print("\nINDEX EXISTS:", INDEX_DIR.exists())
    print("DOCUMENTS EXISTS:", DOCUMENTS_DIR.exists())

    if INDEX_DIR.exists():
        print("\nINDEX FILES:")

        for f in INDEX_DIR.iterdir():
            print(
                " ",
                f.name,
                "exists:",
                f.exists()
            )

    # ------------------------------------------------------------
    # Search engine
    # ------------------------------------------------------------

    if IntentStoreSearch is not None:

        try:

            print("\nLoading search engine...")

            search_engine = IntentStoreSearch()

            print(
                "SEARCH ENGINE READY."
            )

        except Exception as e:

            print(
                "\nSEARCH ENGINE FAILED:"
            )

            print(
                repr(e)
            )

            search_engine = None

    # ------------------------------------------------------------
    # RAG engine
    # ------------------------------------------------------------

    if RAGEngine is not None:

        try:

            print("\nLoading RAG engine...")

            rag_engine = RAGEngine()

            print(
                "RAG ENGINE READY."
            )

            # If RAG engine contains search engine,
            # use that same engine.
            if hasattr(
                rag_engine,
                "search_engine"
            ):

                if rag_engine.search_engine is not None:

                    search_engine = (
                        rag_engine.search_engine
                    )

                    print(
                        "Using search engine from RAG engine."
                    )

        except Exception as e:

            print(
                "\nRAG ENGINE FAILED:"
            )

            print(
                repr(e)
            )

            rag_engine = None

    print("\n" + "=" * 70)

    print(
        "FINAL BACKEND STATUS"
    )

    print(
        "RAG:",
        rag_engine is not None
    )

    print(
        "SEARCH:",
        search_engine is not None
    )

    print(
        "DOCUMENT DIRECTORY:",
        DOCUMENTS_DIR.exists()
    )

    print("=" * 70)


initialize_backend()


# ================================================================
# HOME
# ================================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ================================================================
# STATIC FILES
# ================================================================

@app.route("/<path:filename>")
def static_files(filename):

    requested_file = FRONTEND_DIR / filename

    if (
        requested_file.exists()
        and requested_file.is_file()
    ):

        return send_from_directory(
            FRONTEND_DIR,
            filename
        )

    return jsonify({
        "success": False,
        "error": "File not found",
        "file": filename
    }), 404


# ================================================================
# HEALTH
# ================================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,

        "status": "ok",

        "rag_available":
            rag_engine is not None,

        "search_available":
            search_engine is not None,

        "documents_available":
            DOCUMENTS_DIR.exists(),

        "index_available":
            INDEX_DIR.exists(),

        "index_files":
            [
                f.name
                for f in INDEX_DIR.iterdir()
            ]
            if INDEX_DIR.exists()
            else []

    })


# ================================================================
# INFORMATION RETRIEVAL
# ================================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    if rag_engine is None:

        return jsonify({

            "success": False,

            "answer":
                "RAG engine is not available.",

            "results": []

        }), 503

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get(
            "question",
            ""
        )
    ).strip()

    if not question:

        return jsonify({

            "success": False,

            "answer":
                "Please enter a question.",

            "results": []

        }), 400

    try:

        print(
            "\nUSER QUESTION:",
            question
        )

        response = rag_engine.ask(
            question
        )

        if not isinstance(
            response,
            dict
        ):

            response = {
                "answer": str(response),
                "results": []
            }

        results = response.get(
            "results",
            []
        )

        answer = response.get(
            "answer",
            ""
        )

        formatted_results = []

        for result in results:

            formatted_results.append({

                "filename":
                    result.get(
                        "filename",
                        "Unknown"
                    ),

                "page_number":
                    result.get(
                        "page_number",
                        "Unknown"
                    ),

                "chunk_number":
                    result.get(
                        "chunk_number",
                        "Unknown"
                    ),

                "similarity_score":
                    result.get(
                        "similarity_score",
                        0
                    ),

                "keyword_score":
                    result.get(
                        "keyword_score",
                        0
                    ),

                "metadata_score":
                    result.get(
                        "metadata_score",
                        0
                    ),

                "relevance_score":
                    result.get(
                        "relevance_score",
                        0
                    ),

                "text":
                    result.get(
                        "text",
                        ""
                    )

            })

        return jsonify({

            "success": True,

            "question": question,

            "answer": answer,

            "results": formatted_results

        })

    except Exception as e:

        print(
            "\nRAG ERROR:"
        )

        print(
            repr(e)
        )

        return jsonify({

            "success": False,

            "answer":
                "Error while processing the question.",

            "error":
                str(e),

            "results": []

        }), 500


# ================================================================
# RAW SEMANTIC SEARCH
# ================================================================

@app.route("/api/search", methods=["POST"])
def search():

    if search_engine is None:

        return jsonify({

            "success": False,

            "error":
                "Search engine is not available.",

            "results": []

        }), 503

    data = request.get_json(
        silent=True
    ) or {}

    query = str(
        data.get(
            "query",
            ""
        )
    ).strip()

    if not query:

        return jsonify({

            "success": False,

            "error":
                "Please enter a search query.",

            "results": []

        }), 400

    try:

        results = search_engine.search(
            query,
            top_k=5
        )

        return jsonify({

            "success": True,

            "results": results

        })

    except Exception as e:

        print(
            "\nSEARCH ERROR:"
        )

        print(
            repr(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e),

            "results": []

        }), 500


# ================================================================
# DOCUMENT RETRIEVAL
# ================================================================
#
# IMPORTANT:
# This is a completely separate endpoint.
#
# It does NOT use the RAG answer endpoint.
# It searches the actual files inside:
#
# backend/data/documents
#
# ================================================================

@app.route("/api/document", methods=["POST"])
def document_retrieval():

    data = request.get_json(
        silent=True
    ) or {}

    query = str(
        data.get(
            "query",
            ""
        )
    ).strip()

    if not query:

        return jsonify({

            "success": False,

            "error":
                "Please enter a document name.",

            "document": None

        }), 400

    print(
        "\nDOCUMENT SEARCH:",
        query
    )

    if not DOCUMENTS_DIR.exists():

        return jsonify({

            "success": False,

            "error":
                "Document directory does not exist.",

            "document": None

        }), 404

    # ------------------------------------------------------------
    # Collect all files
    # ------------------------------------------------------------

    files = [
        f
        for f in DOCUMENTS_DIR.rglob("*")
        if f.is_file()
    ]

    if not files:

        return jsonify({

            "success": False,

            "error":
                "No documents found.",

            "document": None

        }), 404

    # ------------------------------------------------------------
    # Normalize query
    # ------------------------------------------------------------

    query_words = set(
        query.lower()
        .replace(
            "_",
            " "
        )
        .replace(
            "-",
            " "
        )
        .split()
    )

    # ------------------------------------------------------------
    # Score filenames
    # ------------------------------------------------------------

    scored_files = []

    for file in files:

        filename = file.name.lower()

        stem = file.stem.lower()

        filename_words = set(
            filename
            .replace(
                "_",
                " "
            )
            .replace(
                "-",
                " "
            )
            .split()
        )

        stem_words = set(
            stem
            .replace(
                "_",
                " "
            )
            .replace(
                "-",
                " "
            )
            .split()
        )

        score = 0

        # Exact filename match
        if query.lower() in filename:
            score += 100

        # Word matches
        score += len(
            query_words &
            filename_words
        ) * 20

        score += len(
            query_words &
            stem_words
        ) * 10

        scored_files.append(
            (
                score,
                file
            )
        )

    scored_files.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_file = (
        scored_files[0]
    )

    # ------------------------------------------------------------
    # If nothing matches filename
    # ------------------------------------------------------------

    if best_score <= 0:

        return jsonify({

            "success": False,

            "error":
                "No matching document was found.",

            "document": None,

            "available_documents":
                [
                    f.name
                    for f in files
                ]

        }), 404

    # ------------------------------------------------------------
    # Return document information
    # ------------------------------------------------------------

    relative_path = best_file.relative_to(
        DOCUMENTS_DIR
    )

    return jsonify({

        "success": True,

        "document": {

            "filename":
                best_file.name,

            "path":
                str(relative_path),

            "size":
                best_file.stat().st_size,

            "extension":
                best_file.suffix,

            "url":
                "/api/document/file/"
                + relative_path.as_posix()

        }

    })


# ================================================================
# SERVE RETRIEVED DOCUMENT
# ================================================================

@app.route(
    "/api/document/file/<path:filename>",
    methods=["GET"]
)
def serve_document(filename):

    requested = (
        DOCUMENTS_DIR / filename
    ).resolve()

    documents_root = (
        DOCUMENTS_DIR.resolve()
    )

    # Security check
    try:

        requested.relative_to(
            documents_root
        )

    except ValueError:

        return jsonify({

            "success": False,

            "error":
                "Invalid document path."

        }), 403

    if not requested.exists():

        return jsonify({

            "success": False,

            "error":
                "Document not found."

        }), 404

    return send_from_directory(
        requested.parent,
        requested.name
    )


# ================================================================
# EXAMPLE QUESTIONS
# ================================================================

EXAMPLE_QUESTIONS = [

    "What is the main motivation behind the proposed system?",

    "What are the key objectives mentioned in the document?",

    "What methodology is described in the document?",

    "What are the major findings discussed?",

    "What technologies or tools are mentioned?",

    "What problem does the proposed system solve?",

    "What are the advantages of the proposed approach?",

    "What are the limitations mentioned in the document?",

    "What conclusions are presented in the document?"

]


@app.route(
    "/api/examples",
    methods=["GET"]
)
def examples():

    return jsonify({

        "success": True,

        "questions":
            EXAMPLE_QUESTIONS

    })


# ================================================================
# START SERVER
# ================================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("INTENTSTORE")
    print("=" * 70)

    print(
        "\nOpen:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nRAG:",
        rag_engine is not None
    )

    print(
        "SEARCH:",
        search_engine is not None
    )

    print(
        "DOCUMENTS:",
        DOCUMENTS_DIR.exists()
    )

    print(
        "\n" + "=" * 70
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )