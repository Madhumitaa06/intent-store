# ================================================================
# INTENTSTORE FRONTEND + BACKEND CONNECTOR
# frontend/app.py
# ================================================================

import os
import sys
import json
import re
import uuid
from pathlib import Path

import io
import numpy as np
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import auth

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
# BACKEND IMPORTS (LAZY)
# Do not load PyTorch / transformer models while the web server starts.
# They are imported only when an AI feature is first used.
# ================================================================

WEB_ONLY = os.getenv("WEB_ONLY", "false").lower() == "true"  # legacy setting; no longer disables features
ENABLE_GENERATIVE_LLM = os.getenv("ENABLE_GENERATIVE_LLM", "false").lower() == "true"

RAGEngine = None
IntentStoreSearch = None
_backend_import_error = None

# ================================================================
# FLASK
# static_folder=None turns off Flask's built-in file serving,
# so only our checked static_files() route can serve files.
# ================================================================

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
CORS(app)

# ================================================================
# BACKEND INITIALIZATION (ON DEMAND)
# ================================================================

rag_engine = None
search_engine = None
_backend_initialized = False
_rag_initialized = False


def ensure_backend_initialized():
    """Load the semantic search backend only when a feature needs it."""
    global _backend_initialized, IntentStoreSearch, search_engine, _backend_import_error

    if _backend_initialized:
        return

    try:
        from search_engine import IntentStoreSearch as SearchClass
        IntentStoreSearch = SearchClass
        print("IntentStoreSearch imported on demand.")
    except Exception as error:
        _backend_import_error = error
        print("Search engine import failed:", repr(error))
        search_engine = None
        _backend_initialized = True
        return

    try:
        search_engine = IntentStoreSearch()
        print("SEARCH ENGINE READY.")
    except Exception as error:
        _backend_import_error = error
        print("SEARCH ENGINE FAILED:", repr(error))
        search_engine = None

    _backend_initialized = True
    print("SEARCH:", search_engine is not None)


def ensure_rag_initialized():
    """Optionally load generative RAG only when enabled and first requested."""
    global _rag_initialized, rag_engine, RAGEngine, search_engine

    if _rag_initialized or not ENABLE_GENERATIVE_LLM:
        return

    try:
        from rag_engine import RAGEngine as RagClass
        RAGEngine = RagClass
        rag_engine = RAGEngine()
        if getattr(rag_engine, "search_engine", None) is not None:
            search_engine = rag_engine.search_engine
        print("GENERATIVE RAG ENGINE READY.")
    except Exception as error:
        print("GENERATIVE RAG ENGINE FAILED:", repr(error))
        rag_engine = None
    finally:
        _rag_initialized = True


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
# PRIVATE USER DOCUMENT UPLOADS
# ================================================================

def _user_index_paths(user_id):
    folder = DATA_DIR / "user_indexes"
    return (
        folder,
        folder / f"{user_id}_chunks.json",
        folder / f"{user_id}_embeddings.npy"
    )


def _load_user_index(user_id, model):
    folder, chunks_path, embeddings_path = _user_index_paths(user_id)
    folder.mkdir(parents=True, exist_ok=True)

    chunks = []
    if chunks_path.exists():
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        except Exception:
            chunks = []

    vectors = None
    if embeddings_path.exists():
        try:
            vectors = np.load(embeddings_path)
        except Exception:
            vectors = None

    if vectors is None or len(vectors) != len(chunks):
        if chunks:
            texts = [str(c.get("text", "")) for c in chunks]
            vectors = np.asarray(
                model.encode(texts, normalize_embeddings=True),
                dtype="float32"
            )
            np.save(embeddings_path, vectors)
        else:
            vectors = np.empty(
                (0, model.get_sentence_embedding_dimension()),
                dtype="float32"
            )


    return chunks, vectors


def _update_user_organization(user_id, chunks, vectors):
    """Generate organization and version metadata for one user's files."""
    from collections import defaultdict
    from auto_organizer import classify, extract_tags, content_hash, shingles, jaccard

    folder, chunks_path, embeddings_path = _user_index_paths(user_id)
    folder.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    row_map = defaultdict(list)
    display_names = {}

    for i, chunk in enumerate(chunks):
        if int(chunk.get("user_id", -1)) != int(user_id):
            continue
        filename = chunk.get("filename")
        if filename:
            grouped[filename].append(str(chunk.get("text", "")))
            row_map[filename].append(i)
            display_names.setdefault(
                filename,
                chunk.get("display_filename") or Path(filename).name
            )

    names = list(grouped)
    full_text = {name: " ".join(grouped[name]) for name in names}
    hashes = {name: content_hash(full_text[name]) for name in names}
    word_sets = {name: shingles(full_text[name]) for name in names}

    exact_of = {}
    first_seen = {}
    for name in names:
        digest = hashes[name]
        if digest in first_seen:
            exact_of[name] = first_seen[digest]
        else:
            first_seen[digest] = name

    mean_vectors = {}
    if vectors is not None and len(vectors) == len(chunks):
        import numpy as np
        for name, rows in row_map.items():
            mean = vectors[rows].mean(axis=0)
            norm = np.linalg.norm(mean)
            mean_vectors[name] = mean / norm if norm else mean

    near = defaultdict(list)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if hashes[first] == hashes[second]:
                continue

            text_score = jaccard(word_sets[first], word_sets[second])
            meaning_score = 0.0
            if first in mean_vectors and second in mean_vectors:
                meaning_score = float(mean_vectors[first] @ mean_vectors[second])

            if (text_score >= 0.80 or
                    (text_score >= 0.40 and meaning_score >= 0.95)):
                near[first].append({
                    "filename": second,
                    "text_similarity": round(text_score, 3),
                    "meaning_similarity": round(meaning_score, 3)
                })
                near[second].append({
                    "filename": first,
                    "text_similarity": round(text_score, 3),
                    "meaning_similarity": round(meaning_score, 3)
                })

    organization = {}
    for name in names:
        category, confidence = classify(full_text[name])
        normalized_text = (display_names.get(name, name) + " " + full_text[name]).lower()

        education_markers = (
            "marksheet",
            "mark sheet",
            "marks memo",
            "grade sheet",
            "secondary school certificate",
            "ssc certificate",
            "hall ticket",
            "marks obtained"
        )

        if any(marker in normalized_text for marker in education_markers):
            category = "Education"
            confidence = 1.0

        tags = extract_tags(full_text[name])
        if category != "Uncategorized":
            tags = [category.lower()] + tags

        organization[name] = {
            "filename": name,
            "display_filename": display_names.get(name, Path(name).name),
            "category": category,
            "category_confidence": confidence,
            "tags": tags,
            "content_hash": hashes[name],
            "is_duplicate": name in exact_of,
            "exact_duplicate_of": exact_of.get(name),
            "near_duplicates": near[name]
        }

    org_path = folder / f"{user_id}_organization.json"
    org_path.write_text(
        json.dumps(organization, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Group exact and near duplicates into version chains.
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for name, info in organization.items():
        if info["exact_duplicate_of"]:
            union(name, info["exact_duplicate_of"])
        for item in info["near_duplicates"]:
            union(name, item["filename"])

    chains = defaultdict(list)
    for name in names:
        chains[find(name)].append(name)

    versions = {}
    chain_number = 0
    for members in chains.values():
        if len(members) < 2:
            continue
        chain_number += 1
        chain_id = f"chain_{chain_number}"
        members.sort(
            key=lambda name: (
                (DOCUMENTS_DIR / name).stat().st_mtime
                if (DOCUMENTS_DIR / name).exists() else 0
            )
        )
        for i, name in enumerate(members):
            versions[name] = {
                "chain_id": chain_id,
                "version": i + 1,
                "total_versions": len(members),
                "is_latest": i == len(members) - 1,
                "previous_version": members[i - 1] if i > 0 else None,
                "next_version": members[i + 1] if i + 1 < len(members) else None
            }

    (folder / f"{user_id}_versions.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


@app.route("/api/upload", methods=["POST"])
def upload_documents():
    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    # Load PDF/OCR dependencies only when a PDF upload is actually requested.
    global fitz, pytesseract, Image
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except Exception as error:
        return jsonify({
            "success": False,
            "error": "PDF processing dependencies are unavailable.",
            "detail": str(error)
        }), 503

    ensure_backend_initialized()

    files = request.files.getlist("documents")
    files = [f for f in files if f and f.filename]

    if not files:
        return jsonify({
            "success": False,
            "error": "Choose one or more PDF files to upload."
        }), 400

    user_id = user["user_id"]
    model = search_engine.model
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks, all_vectors = _load_user_index(user_id, model)
    uploaded = []
    errors = []

    # ============================================================
    # DUPLICATE DISPLAY FILENAME CHECK
    # ============================================================
    existing_display_names = {
        str(
            chunk.get("display_filename")
            or Path(str(chunk.get("filename", ""))).name
        ).casefold()
        for chunk in all_chunks
        if isinstance(chunk, dict)
    }

    # Also check registered files, including any without text chunks.
    for owned_name in auth.get_user_documents(user_id):
        stored = Path(owned_name).name
        prefix = f"user{user_id}_"

        if stored.startswith(prefix):
            tail = stored[len(prefix):]
            if len(tail) > 13 and tail[12] == "_":
                stored = tail[13:]

        existing_display_names.add(stored.casefold())

    for incoming in files:
        original_name = incoming.filename
        safe_name = secure_filename(original_name)

        if not safe_name or not safe_name.lower().endswith(".pdf"):
            errors.append(f"{original_name}: only PDF files are supported.")
            continue

        if safe_name.casefold() in existing_display_names:
            errors.append(
                f"{safe_name}: a file with this name already exists "
                "in your workspace. Rename the file and upload again."
            )
            continue

        stored_name = (
            f"user{user_id}_{uuid.uuid4().hex[:12]}_{safe_name}"
        )
        target = DOCUMENTS_DIR / stored_name
        incoming.save(target)

        new_chunks = []
        document_id = str(uuid.uuid4())

        try:
            with fitz.open(str(target)) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    # First extract any selectable PDF text.
                    text = page.get_text("text").strip()

                    # If this page is scanned or has very little text,
                    # render it and use local Tesseract OCR.
                    if len(text) < 30:
                        print(
                            f"  Page {page_number}: little selectable text; "
                            "trying local OCR..."
                        )
                        try:
                            pixmap = page.get_pixmap(
                                matrix=fitz.Matrix(2, 2),
                                alpha=False
                            )
                            image = Image.open(
                                io.BytesIO(pixmap.tobytes("png"))
                            )
                            ocr_text = pytesseract.image_to_string(
                                image,
                                lang="eng"
                            ).strip()

                            # Keep the more useful extraction.
                            if len(ocr_text) > len(text):
                                text = ocr_text
                                print(
                                    f"  Page {page_number}: OCR extracted "
                                    f"{len(text)} characters."
                                )
                        except Exception as ocr_error:
                            print(
                                f"  Page {page_number}: OCR failed "
                                f"({type(ocr_error).__name__})."
                            )

                    words = text.split()
                    start = 0
                    chunk_number = 0

                    while start < len(words):
                        piece = " ".join(words[start:start + 800]).strip()
                        if piece:
                            chunk_number += 1
                            new_chunks.append({
                                "document_id": document_id,
                                "user_id": user_id,
                                "filename": stored_name,
                                "display_filename": safe_name,
                                "page_number": page_number,
                                "chunk_number": chunk_number,
                                "text": piece
                            })
                        start += 650

        except Exception as error:
            target.unlink(missing_ok=True)
            errors.append(f"{original_name}: could not read PDF ({type(error).__name__}).")
            continue

        if not new_chunks:
            target.unlink(missing_ok=True)
            errors.append(
                f"{original_name}: no selectable text found. "
                "Scanned PDFs need OCR before they can be searched."
            )
            continue

        try:
            texts = [chunk["text"] for chunk in new_chunks]
            new_vectors = np.asarray(
                model.encode(texts, normalize_embeddings=True),
                dtype="float32"
            )

            if not auth.add_user_document(user_id, stored_name):
                target.unlink(missing_ok=True)
                errors.append(f"{original_name}: could not register document ownership.")
                continue

            all_chunks.extend(new_chunks)
            if len(all_vectors):
                all_vectors = np.vstack((all_vectors, new_vectors))
            else:
                all_vectors = new_vectors

            folder, chunks_path, embeddings_path = _user_index_paths(user_id)
            folder.mkdir(parents=True, exist_ok=True)

            temp_chunks = chunks_path.with_suffix(".tmp")
            temp_chunks.write_text(
                json.dumps(all_chunks, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            temp_chunks.replace(chunks_path)
            np.save(embeddings_path, all_vectors)

            try:
                _update_user_organization(user_id, all_chunks, all_vectors)
            except Exception as metadata_error:
                print("User organization/version update failed:",
                      type(metadata_error).__name__)

            # Add non-sensitive uploads to the user's private knowledge graph.
            graph_status = "skipped_sensitive"

            graph_text = " ".join(
                chunk.get("text", "") for chunk in new_chunks
            )
            lower_name = safe_name.lower()
            lower_text = graph_text[:12000].lower()

            sensitive_markers = (
                "aadhaar number", "aadhar number", "passport number",
                "pan number", "marksheet", "marks memo"
            )

            is_sensitive = (
                any(marker in lower_name for marker in (
                    "aadhaar", "aadhar", "passport", "pan_card",
                    "marksheet", "marks_memo", "bank_statement"
                ))
                or any(marker in lower_text for marker in sensitive_markers)
            )

            if not is_sensitive:
                try:
                    from entity_extractor import EntityExtractor
                    from knowledge_graph import KnowledgeGraph

                    extraction = EntityExtractor().extract(
                        stored_name,
                        graph_text[:12000],
                        document_id
                    )

                    if not isinstance(extraction, dict):
                        raise ValueError("Entity extraction returned an invalid result.")

                    extraction["filename"] = stored_name
                    extraction["document_id"] = document_id

                    graph = KnowledgeGraph()
                    try:
                        graph.add_user_document_knowledge(
                            extraction,
                            user_id
                        )
                    finally:
                        graph.close()

                    graph_status = "added"

                except Exception as graph_error:
                    print(
                        "Knowledge graph update failed:",
                        type(graph_error).__name__
                    )
                    graph_status = "failed"

            uploaded.append({
                "filename": safe_name,
                "chunks_created": len(new_chunks),
                "knowledge_graph": graph_status
            })
            existing_display_names.add(safe_name.casefold())

        except Exception as error:
            target.unlink(missing_ok=True)
            errors.append(
                f"{original_name}: indexing failed ({type(error).__name__})."
            )

    if not uploaded:
        return jsonify({
            "success": False,
            "uploaded": [],
            "errors": errors or ["No files were uploaded."]
        }), 400

    return jsonify({
        "success": True,
        "uploaded": uploaded,
        "errors": errors
    })

# ================================================================

@app.route("/api/organization", methods=["GET"])
def user_organization():
    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    user_id = user["user_id"]
    folder, _, _ = _user_index_paths(user_id)
    organization_path = folder / f"{user_id}_organization.json"
    versions_path = folder / f"{user_id}_versions.json"

    try:
        # Generate metadata for existing uploads if it does not exist yet.
        if not organization_path.exists():
            ensure_backend_initialized()
            
            if search_engine is None:
                return jsonify({
                    "success": False,
                    "error": "Search engine is unavailable."
                }), 503

            chunks, vectors = _load_user_index(
                user_id, search_engine.model
            )
            _update_user_organization(user_id, chunks, vectors)

        organization = json.loads(
            organization_path.read_text(encoding="utf-8")
        )

        versions = {}
        if versions_path.exists():
            versions = json.loads(
                versions_path.read_text(encoding="utf-8")
            )

        owned = auth.get_user_documents(user_id)
        documents = []

        for stored_name, item in organization.items():
            if stored_name not in owned:
                continue

            exact_name = item.get("exact_duplicate_of")
            exact_display = None
            if exact_name and exact_name in organization:
                exact_display = organization[exact_name].get(
                    "display_filename", Path(exact_name).name
                )

            near_duplicates = []
            for duplicate in item.get("near_duplicates", []):
                other_name = duplicate.get("filename")
                if other_name not in owned:
                    continue

                near_duplicates.append({
                    "filename": organization.get(
                        other_name, {}
                    ).get("display_filename", Path(other_name).name),
                    "text_similarity": duplicate.get("text_similarity", 0),
                    "meaning_similarity": duplicate.get("meaning_similarity", 0)
                })

            documents.append({
                "filename": item.get(
                    "display_filename", Path(stored_name).name
                ),
                "category": item.get("category", "Uncategorized"),
                "category_confidence": item.get("category_confidence", 0),
                "tags": item.get("tags", []),
                "is_duplicate": item.get("is_duplicate", False),
                "exact_duplicate_of": exact_display,
                "near_duplicates": near_duplicates,
                "version": versions.get(stored_name)
            })

        return jsonify({
            "success": True,
            "documents": documents,
            "total_documents": len(documents)
        })

    except Exception as error:
        print("Organization API error:", repr(error))
        return jsonify({
            "success": False,
            "error": "Could not load organization and version metadata."
        }), 500


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
        "search_available": search_engine is not None,
        "generative_llm_enabled": ENABLE_GENERATIVE_LLM
    })

# ================================================================
# INFORMATION RETRIEVAL (AI ANSWER)
# ================================================================



def _search_user_documents(user_id, query, top_k=5):
    """Retrieve evidence only from the authenticated user's index."""
    ensure_backend_initialized()
    
    if search_engine is None:
        return []

    user_id = int(user_id)
    chunks, vectors = _load_user_index(user_id, search_engine.model)
    owned = auth.get_user_documents(user_id)

    if not chunks or vectors is None or len(vectors) == 0:
        return []

    query_vector = np.asarray(
        search_engine.model.encode(
            [query],
            normalize_embeddings=True
        ),
        dtype="float32"
    )[0]

    if vectors.ndim != 2 or vectors.shape[1] != len(query_vector):
        raise ValueError("User embedding dimensions do not match.")

    scores = vectors @ query_vector

    # Combine semantic similarity with query-term and phrase matches.
    # This helps specific questions outrank loosely related documents.
    stop_words = {
        "what", "is", "the", "in", "a", "an", "for", "of",
        "to", "and", "are", "was", "were", "this", "that",
        "about", "how", "does", "do", "used", "with"
    }

    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    content_tokens = [
        token for token in query_tokens if token not in stop_words
    ]
    query_terms = set(content_tokens)
    query_phrases = [
        content_tokens[i] + " " + content_tokens[i + 1]
        for i in range(len(content_tokens) - 1)
    ]

    rank_scores = scores.astype("float32").copy()

    for chunk_index, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue

        text_lower = str(item.get("text", "")).lower()
        text_terms = set(re.findall(r"[a-z0-9]+", text_lower))

        overlap = len(query_terms & text_terms)
        keyword_score = overlap / max(len(query_terms), 1)

        rank_scores[chunk_index] += 0.08 * keyword_score

        for phrase in query_phrases:
            if phrase in text_lower:
                rank_scores[chunk_index] += 0.12

    ranked = np.argsort(rank_scores)[::-1]
    results = []

    for raw_index in ranked:
        index = int(raw_index)
        chunk = chunks[index]
        filename = chunk.get("filename")

        if int(chunk.get("user_id", -1)) != user_id:
            continue
        if filename not in owned:
            continue

        text = str(chunk.get("text", ""))
        upper_text = text.upper()
        dot_leaders = len(re.findall(r"\.{5,}", text))
        section_numbers = len(
            re.findall(r"\b\d+(?:\.\d+)+\b", text)
        )

        # Avoid ranking table-of-contents pages as answer evidence.
        if "TABLE OF CONTENTS" in upper_text:
            continue
        if dot_leaders >= 2 and section_numbers >= 3:
            continue

        results.append({
            "document_id": chunk.get("document_id", ""),
            "filename": chunk.get("display_filename") or filename,
            "page_number": chunk.get("page_number", "Unknown"),
            "chunk_number": chunk.get("chunk_number", "Unknown"),
            "similarity_score": round(float(scores[index]), 4),
            "keyword_score": 0,
            "metadata_score": 0,
            "relevance_score": round(float(rank_scores[index]), 4),
            "text": text
        })

        if len(results) >= max(1, int(top_k)):
            break

    return results



@app.route("/api/related", methods=["POST"])
def related_documents():
    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    data = request.get_json(silent=True) or {}
    requested_name = str(data.get("filename", "")).strip()

    if not requested_name:
        return jsonify({
            "success": False,
            "error": "Provide a document filename."
        }), 400

    user_id = int(user["user_id"])
    owned = auth.get_user_documents(user_id)
    folder, chunks_path, _ = _user_index_paths(user_id)

    if not chunks_path.exists():
        return jsonify({
            "success": True,
            "filename": requested_name,
            "related_documents": []
        })

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        stored_name = None
        document_id = None
        display_names = {}

        for chunk in chunks:
            stored = chunk.get("filename")
            if not stored or stored not in owned:
                continue

            display = chunk.get("display_filename") or Path(stored).name
            display_names[stored] = display

            if (
                requested_name == display
                or requested_name == stored
            ):
                stored_name = stored
                document_id = chunk.get("document_id")

        if not stored_name or not document_id:
            return jsonify({
                "success": False,
                "error": "That document was not found in your workspace."
            }), 404

        from knowledge_graph import KnowledgeGraph

        graph = KnowledgeGraph()
        try:
            related = graph.get_related_documents_for_user_document(
                document_id,
                user_id,
                limit=10
            )
        finally:
            graph.close()

        # Defence in depth: only return files registered to this user.
        results = []
        for item in related:
            related_stored = item.get("filename")
            if related_stored not in owned:
                continue

            results.append({
                "filename": display_names.get(
                    related_stored,
                    Path(related_stored).name
                ),
                "shared_count": item.get("shared_count", 0),
                "shared_entities": item.get("shared_entities", [])
            })

        return jsonify({
            "success": True,
            "filename": display_names.get(stored_name, requested_name),
            "related_documents": results
        })

    except Exception as error:
        print("Related-documents query failed:", type(error).__name__)
        return jsonify({
            "success": False,
            "error": "Could not load related documents."
        }), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({
            "success": False,
            "answer": "Please enter a question.",
            "results": []
        }), 400

    try:
        results = _search_user_documents(user["user_id"], question)

        if not results:
            return jsonify({
                "success": True,
                "question": question,
                "answer": (
                    "No matching content found in your uploaded documents. "
                    "Upload a text-based PDF first."
                ),
                "results": [],
                "generated_by_llm": False
            })

        generated_by_llm = False
        ensure_rag_initialized()

        if rag_engine is not None and getattr(rag_engine, "llm_enabled", False):
            try:
                answer = rag_engine._generate_answer_llm(question, results)
                generated_by_llm = True
            except Exception as error:
                print("AI generation failed:", repr(error))
                answer = rag_engine._generate_answer_extractive(results)
        elif rag_engine is not None:
            answer = rag_engine._generate_answer_extractive(results)
        else:
            answer = "\n\n".join(
                item["text"] for item in results if item.get("text")
            )

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "results": results,
            "generated_by_llm": generated_by_llm
        })

    except Exception as error:
        print("Private RAG error:", repr(error))
        return jsonify({
            "success": False,
            "answer": "Error while processing the question.",
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

    ensure_backend_initialized()

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
        results = _search_user_documents(
            user["user_id"],
            query,
            top_k=5
        )
        return jsonify({
            "success": True,
            "results": results
        })

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
            "document_id": best_file.stem,
            "path": str(relative_path),
            "size": best_file.stat().st_size,
            "extension": best_file.suffix,
            "url": "/api/document/file/" + relative_path.as_posix()
        }
    })

# STORAGE STATISTICS
# ================================================================


@app.route("/api/stats", methods=["GET"])
def storage_stats():
    user = get_logged_in_user()
    if user is None:
        return login_required_response()

    owned = auth.get_user_documents(user["user_id"])

    files = []
    if DOCUMENTS_DIR.exists():
        files = [
            f for f in DOCUMENTS_DIR.rglob("*")
            if f.is_file() and f.name in owned
        ]

    total_bytes = sum(f.stat().st_size for f in files)
    pdf_count = sum(1 for f in files if f.suffix.lower() == ".pdf")
    other_count = len(files) - pdf_count

    largest_file = None
    if files:
        largest = max(files, key=lambda f: f.stat().st_size)
        display_name = largest.name
        parts = largest.name.split("_", 2)
        if (
            len(parts) == 3
            and parts[0].startswith("user")
            and len(parts[1]) == 12
        ):
            display_name = parts[2]

        largest_file = {
            "filename": display_name,
            "size_bytes": largest.stat().st_size
        }

    return jsonify({
        "success": True,
        "total_documents": len(files),
        "total_storage_bytes": total_bytes,
        "pdf_count": pdf_count,
        "other_count": other_count,
        "largest_file": largest_file
    })


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
    app.run(host="127.0.0.1", port=5000, debug=False)
