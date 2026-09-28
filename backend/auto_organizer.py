"""
Objective 4: Auto-organize files
--------------------------------
Reads (never modifies) the files your teammate's scripts already create:
    data/index/chunks.json       (required)
    data/index/embeddings.npy    (optional - enables meaning-based duplicates)

Writes ONE new file:
    data/index/organization.json

Run from the project root (the intent-store-main folder):
    python3 backend/auto_organizer.py
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

CHUNKS_FILE = Path("data/index/chunks.json")
EMBEDDINGS_FILE = Path("data/index/embeddings.npy")
OUTPUT_FILE = Path("data/index/organization.json")

# ---------------- Settings (safe to tweak) ----------------
TEXT_NEAR_DUP_THRESHOLD = 0.80        # word-overlap similarity (0-1)
EMBEDDING_NEAR_DUP_THRESHOLD = 0.95   # meaning similarity (0-1)
MIN_TEXT_FOR_MEANING_MATCH = 0.40     # meaning match only counts if text also overlaps this much
SHINGLE_SIZE = 5
MIN_CATEGORY_HITS = 3
TAG_LIMIT = 8

CATEGORIES = {
    "Project Report": [
        "project", "report", "department", "guidance", "submitted",
        "acknowledgements", "certificate", "university", "guide",
        "coordinator", "partial", "fulfillment",
    ],
    "Research Paper": [
        "abstract", "methodology", "results", "conclusion", "references",
        "dataset", "experiment", "hypothesis", "literature", "analysis",
        "proposed", "evaluation", "model",
    ],
    "Finance": [
        "invoice", "payment", "amount", "tax", "budget", "revenue",
        "expense", "total", "bank", "price", "balance", "receipt",
    ],
    "Legal": [
        "agreement", "contract", "clause", "party", "liability", "terms",
        "law", "hereby", "court", "rights", "confidential", "obligations",
    ],
    "Education": [
        "syllabus", "lecture", "assignment", "course", "exam", "student",
        "semester", "chapter", "lesson", "curriculum", "professor", "unit",
    ],
    "Technical Documentation": [
        "function", "code", "api", "database", "server", "software",
        "algorithm", "system", "python", "implementation", "install",
        "configuration",
    ],
    "Resume": [
        "resume", "experience", "skills", "internship", "certification",
        "objective", "projects", "achievements", "profile",
    ],
    "Meeting Notes": [
        "agenda", "meeting", "minutes", "action", "attendees",
        "discussion", "decisions", "followup",
    ],
    "Business Report": [
        "summary", "findings", "recommendations", "overview", "performance",
        "quarterly", "strategy", "market", "growth", "stakeholders",
    ],
}

STOP_WORDS = {
    "this", "that", "with", "from", "have", "will", "which", "their",
    "there", "about", "into", "these", "those", "also", "than", "then",
    "they", "them", "were", "been", "being", "such", "using", "used",
    "where", "when", "what", "your", "more", "some", "other", "each",
    "only", "very", "here", "shall", "should", "would", "could",
}


# ---------------- Helpers ----------------

def extract_tags(text, limit=TAG_LIMIT):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return [w for w, _ in Counter(words).most_common(limit)]


def classify(text):
    counts = Counter(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
    scores = {
        cat: sum(counts[w] for w in words)
        for cat, words in CATEGORIES.items()
    }
    total = sum(scores.values())
    if total < MIN_CATEGORY_HITS:
        return "Uncategorized", 0.0
    best = max(scores, key=scores.get)
    return best, round(scores[best] / total, 2)


def content_hash(text):
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def shingles(text, size=SHINGLE_SIZE):
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) < size:
        return {" ".join(words)}
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_document_vectors(chunks, documents_order):
    """
    Average each document's chunk embeddings into one vector.
    Assumes embeddings.npy rows are in the same order as chunks.json
    (which is how your FAISS index is built). Returns None if unavailable.
    """
    if not EMBEDDINGS_FILE.exists():
        return None

    import numpy as np

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(embeddings) != len(chunks):
        print("Warning: embeddings and chunks don't line up - "
              "skipping meaning-based duplicate check.")
        return None

    rows = defaultdict(list)
    for i, chunk in enumerate(chunks):
        rows[chunk["filename"]].append(i)

    vectors = {}
    for filename in documents_order:
        mean = embeddings[rows[filename]].mean(axis=0)
        norm = np.linalg.norm(mean)
        vectors[filename] = mean / norm if norm else mean
    return vectors


# ---------------- Main ----------------

def organize():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk["filename"]].append(chunk)

    filenames = list(documents.keys())
    full_texts = {
        name: " ".join(c["text"] for c in documents[name])
        for name in filenames
    }

    vectors = load_document_vectors(chunks, filenames)
    if vectors is None:
        print("No usable embeddings found - using text overlap only.")

    hashes = {name: content_hash(text) for name, text in full_texts.items()}
    word_groups = {name: shingles(text) for name, text in full_texts.items()}

    # Exact duplicates
    first_seen = {}
    exact_of = {}
    for name in filenames:
        h = hashes[name]
        if h in first_seen:
            exact_of[name] = first_seen[h]
        else:
            first_seen[h] = name

    # Near duplicates
    near = defaultdict(list)
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            a, b = filenames[i], filenames[j]
            if hashes[a] == hashes[b]:
                continue

            text_sim = jaccard(word_groups[a], word_groups[b])
            meaning_sim = 0.0
            if vectors is not None:
                meaning_sim = float(vectors[a] @ vectors[b])

            if (text_sim >= TEXT_NEAR_DUP_THRESHOLD
                    or (text_sim >= MIN_TEXT_FOR_MEANING_MATCH
                        and meaning_sim >= EMBEDDING_NEAR_DUP_THRESHOLD)):
                near[a].append({
                    "filename": b,
                    "text_similarity": round(text_sim, 3),
                    "meaning_similarity": round(meaning_sim, 3),
                })
                near[b].append({
                    "filename": a,
                    "text_similarity": round(text_sim, 3),
                    "meaning_similarity": round(meaning_sim, 3),
                })

    result = {}
    for name in filenames:
        category, confidence = classify(full_texts[name])
        tags = extract_tags(full_texts[name])
        if category != "Uncategorized":
            tags = [category.lower()] + tags

        result[name] = {
            "filename": name,
            "category": category,
            "category_confidence": confidence,
            "tags": tags,
            "content_hash": hashes[name],
            "is_duplicate": name in exact_of,
            "exact_duplicate_of": exact_of.get(name),
            "near_duplicates": near.get(name, []),
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\nAuto-organization complete!")
    print(f"Documents processed: {len(result)}")
    print(f"Exact duplicates: {sum(1 for r in result.values() if r['is_duplicate'])}")
    print(f"Documents with near-duplicates: "
          f"{sum(1 for r in result.values() if r['near_duplicates'])}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    organize()
