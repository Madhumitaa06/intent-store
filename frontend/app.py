# ================================================================
# INTENTSTORE FRONTEND + BACKEND CONNECTOR
# frontend/app.py
# ================================================================

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ================================================================
# PATHS
# ================================================================

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Use backend/data if it has the index, otherwise the project-level data folder
if (BACKEND_DIR / "data" / "index").exists():
    DATA_DIR = BACKEND_DIR / "data"
else:
    DATA_DIR = PROJECT_ROOT / "data"

INDEX_DIR = DATA_DIR / "index"
DOCUMENTS_DIR = DATA_DIR / "documents"

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

import auth

# ================================================================
# FLASK
# static_folder=None turns off Flask's built-in file serving,
# so only our checked static_files() route can serve files.
# ================================================================

app = Flask(__name__, static_folder=None)
CORS(app)

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
    print("DATA:", DATA_DIR)
    print("INDEX EXISTS:", INDEX_DIR.exists())
    print("DOCUMENTS EXISTS:", DOCUMENTS_DIR.exists())

    if IntentStoreSearch is not None:
        try:
            search_engine = IntentStoreSearch()
            print("SEARCH ENGINE READY.")
        except Exception as e:
            print("SEARCH ENGINE FAILED:", repr(e))
            search_engine = None

    if RAGEngine is not None:
        try:
            rag_engine = RAGEngine()
            print("RAG ENGINE READY.")

            if getattr(rag_engine, "search_engine", None) is not None:
                search_engine = rag_engine.search_engine
                print("Using search engine from RAG engine.")
        except Exception as e:
            print("RAG ENGINE FAILED:", repr(e))
            rag_engine = None

    print("RAG:", rag_engine is not None)
    print("SEARCH:", search_engine is not None)
    print("=" * 70)


initialize_backend()

# ================================================================
# AUTH: SIGNUP / LOGIN
# ================================================================


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", ""))

    success, message = auth.create_user(email, password)

    if not success:
        return jsonify({"success": False, "error": message}), 400

    return jsonify({"success": True, "message": message})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", ""))

    success, token_or_message = auth.check_login(email, password)

    if not success:
        return jsonify({"success": False, "error": token_or_message}), 401

    return jsonify({"success": True, "token": token_or_message})


def get_logged_in_user():
    """Returns {"user_id": ..., "email": ...} or None."""
    header = request.headers.get("Authorization", "")
    token = header.replace("Bearer ", "").strip()

    if not token:
        token = request.args.get("token", "").strip()

    return auth.get_user_from_token(token)


def login_required_response():
    return jsonify({"success": False, "error": "Please log in first."}), 401

# ================================================================
# HOME + STATIC FILES
# ================================================================

BLOCKED_EXTENSIONS = {
    ".py", ".db", ".sqlite", ".sqlite3", ".env", ".pyc", ".json", ".key"
}


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):

    if Path(filename).suffix.lower() in BLOCKED_EXTENSIONS:
        return jsonify({"success": False, "error": "File not found"}), 404

    requested_file = FRONTEND_DIR / filename

    if requested_file.exists() and requested_file.is_file():
        return send_from_directory(FRONTEND_DIR, filename)

    return jsonify({
        "success": False,
        "error": "File not found",
        "file": filename
    }), 404

# ================================================================
# HEALTH (no file names exposed)
# ================================================================


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "rag_available": rag_engine is not None,
        "search_available": search_engine is not None
    })

# ================================================================
# INFORMATION RETRIEVAL (AI ANSWER)
# ================================================================


@app.route("/api/ask", methods=["POST"])
def ask():

    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    if rag_engine is None:
        return jsonify({
            "success": False,
            "answer": "RAG engine is not available.",
            "results": []
        }), 503

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({
            "success": False,
            "answer": "Please enter a question.",
            "results": []
        }), 400

    try:
        print("\nUSER QUESTION:", question)

        response = rag_engine.ask(question)

        if not isinstance(response, dict):
            response = {"answer": str(response), "results": []}

        results = response.get("results", [])
        answer = response.get("answer", "")

        # Extra safety: only return results from files this user owns
        owned = auth.get_user_documents(user["user_id"])

        formatted_results = []

        for result in results:
            if result.get("filename") not in owned:
                continue

            formatted_results.append({
                "filename": result.get("filename", "Unknown"),
                "page_number": result.get("page_number", "Unknown"),
                "chunk_number": result.get("chunk_number", "Unknown"),
                "similarity_score": result.get("similarity_score", 0),
                "keyword_score": result.get("keyword_score", 0),
                "metadata_score": result.get("metadata_score", 0),
                "relevance_score": result.get("relevance_score", 0),
                "text": result.get("text", "")
            })

        # If nothing belongs to this user, do not show an answer built
        # from other people's documents
        if not formatted_results:
            answer = "No matching content found in your documents."

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "results": formatted_results
        })

    except Exception as e:
        print("RAG ERROR:", repr(e))
        return jsonify({
            "success": False,
            "answer": "Error while processing the question.",
            "error": str(e),
            "results": []
        }), 500

# ================================================================
# RAW SEMANTIC SEARCH
# ================================================================


@app.route("/api/search", methods=["POST"])
def search():

    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    if search_engine is None:
        return jsonify({
            "success": False,
            "error": "Search engine is not available.",
            "results": []
        }), 503

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({
            "success": False,
            "error": "Please enter a search query.",
            "results": []
        }), 400

    try:
        results = search_engine.search(
            query,
            top_k=5,
            user_id=user["user_id"]
        )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        print("SEARCH ERROR:", repr(e))
        return jsonify({
            "success": False,
            "error": str(e),
            "results": []
        }), 500

# ================================================================
# DOCUMENT RETRIEVAL (only the logged-in user's own files)
# ================================================================


@app.route("/api/document", methods=["POST"])
def document_retrieval():

    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({
            "success": False,
            "error": "Please enter a document name.",
            "document": None
        }), 400

    if not DOCUMENTS_DIR.exists():
        return jsonify({
            "success": False,
            "error": "Document directory does not exist.",
            "document": None
        }), 404

    owned = auth.get_user_documents(user["user_id"])

    files = [
        f for f in DOCUMENTS_DIR.rglob("*")
        if f.is_file() and f.name in owned
    ]

    if not files:
        return jsonify({
            "success": False,
            "error": "You have no documents yet.",
            "document": None
        }), 404

    def words(text):
        return set(
            text.lower().replace("_", " ").replace("-", " ").split()
        )

    query_words = words(query)
    scored_files = []

    for file in files:
        score = 0

        if query.lower() in file.name.lower():
            score += 100

        score += len(query_words & words(file.name)) * 20
        score += len(query_words & words(file.stem)) * 10

        scored_files.append((score, file))

    scored_files.sort(key=lambda x: x[0], reverse=True)
    best_score, best_file = scored_files[0]

    if best_score <= 0:
        return jsonify({
            "success": False,
            "error": "No matching document was found.",
            "document": None
        }), 404

    relative_path = best_file.relative_to(DOCUMENTS_DIR)

    return jsonify({
        "success": True,
        "document": {
            "filename": best_file.name,
            "path": str(relative_path),
            "size": best_file.stat().st_size,
            "extension": best_file.suffix,
            "url": "/api/document/file/" + relative_path.as_posix()
        }
    })

# ================================================================
# SERVE A DOCUMENT (login + ownership required)
# ================================================================


@app.route("/api/document/file/<path:filename>", methods=["GET"])
def serve_document(filename):

    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    requested = (DOCUMENTS_DIR / filename).resolve()
    documents_root = DOCUMENTS_DIR.resolve()

    try:
        requested.relative_to(documents_root)
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Invalid document path."
        }), 403

    if not requested.exists():
        return jsonify({
            "success": False,
            "error": "Document not found."
        }), 404

    if requested.name not in auth.get_user_documents(user["user_id"]):
        return jsonify({
            "success": False,
            "error": "Document not found."
        }), 404

    return send_from_directory(requested.parent, requested.name)

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


@app.route("/api/examples", methods=["GET"])
def examples():
    return jsonify({"success": True, "questions": EXAMPLE_QUESTIONS})

# ================================================================
# START SERVER
# ================================================================

if __name__ == "__main__":

    print("\nINTENTSTORE")
    print("Open: http://127.0.0.1:5000")
    print("RAG:", rag_engine is not None)
    print("SEARCH:", search_engine is not None)

    app.run(host="127.0.0.1", port=5000, debug=False)
