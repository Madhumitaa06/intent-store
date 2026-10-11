import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
USER_ID = sys.argv[1] if len(sys.argv) > 1 else "2"

CHUNKS_FILE = ROOT / "data" / "user_indexes" / f"{USER_ID}_chunks.json"
EMBEDDINGS_FILE = ROOT / "data" / "user_indexes" / f"{USER_ID}_embeddings.npy"
OUTPUT_FILE = ROOT / "reports" / f"retrieval_benchmark_user{USER_ID}.csv"

# Manually labelled questions: each has an expected source document.
TEST_CASES = [
    ("What is the proposed system in IntentStore?", "intent_store.pdf"),
    ("How does semantic retrieval find files by meaning?", "intent_store.pdf"),
    ("What are the main objectives of IntentStore?", "intent_store.pdf"),
    ("Which model architecture is used for shelf-life prediction?", "shelf_life_pred.pdf"),
    ("How are freshness classes mapped to estimated shelf life?", "shelf_life_pred.pdf"),
    ("Which software libraries are used for image processing and model training?", "shelf_life_pred.pdf"),
]

STOP_WORDS = {
    "what", "is", "the", "a", "an", "in", "on", "of", "for",
    "to", "and", "are", "how", "does", "do", "which", "used",
    "with", "by", "from", "this", "that", "their", "main"
}


def tokens(text):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOP_WORDS
    ]


def top_unique(scores):
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return "", 0.0
    # Ties do not count as a confident top-1 prediction.
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        return "TIE", ranked[0][1]
    return ranked[0][0], ranked[0][1]


def main():
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise SystemExit(
            "User index files not found. Upload documents first, "
            "or pass the correct user ID as the script argument."
        )

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    vectors = np.asarray(np.load(EMBEDDINGS_FILE), dtype="float32")

    if len(chunks) != len(vectors):
        raise SystemExit(
            f"Index mismatch: {len(chunks)} chunks, {len(vectors)} vectors."
        )

    documents = defaultdict(list)
    document_rows = defaultdict(list)

    for i, chunk in enumerate(chunks):
        filename = chunk.get("display_filename") or chunk.get("filename", "Unknown")
        documents[filename].append(str(chunk.get("text", "")))
        document_rows[filename].append(i)

    available = set(documents)
    for _, expected in TEST_CASES:
        if expected not in available:
            raise SystemExit(
                f"Expected document {expected!r} not found. "
                f"Available: {sorted(available)}"
            )

    print(f"Evaluation user ID: {USER_ID}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for question, expected in TEST_CASES:
        start = time.perf_counter()
        query_vector = np.asarray(
            model.encode([question], normalize_embeddings=True),
            dtype="float32"
        )[0]
        semantic_chunk_scores = vectors @ query_vector

        semantic_scores = {
            name: float(max(semantic_chunk_scores[row] for row in rows))
            for name, rows in document_rows.items()
        }
        semantic_ms = (time.perf_counter() - start) * 1000
        semantic_top, semantic_score = top_unique(semantic_scores)

        start = time.perf_counter()
        query_terms = set(tokens(question))
        keyword_scores = {}

        for name, pieces in documents.items():
            doc_terms = Counter(tokens(" ".join(pieces)))
            keyword_scores[name] = float(
                sum(doc_terms.get(term, 0) for term in query_terms)
            )

        keyword_ms = (time.perf_counter() - start) * 1000
        keyword_top, keyword_score = top_unique(keyword_scores)

        semantic_correct = semantic_top == expected
        keyword_correct = keyword_top == expected

        results.append({
            "question": question,
            "expected_document": expected,
            "semantic_top1": semantic_top,
            "semantic_score": round(semantic_score, 4),
            "semantic_correct": semantic_correct,
            "semantic_time_ms": round(semantic_ms, 2),
            "keyword_top1": keyword_top,
            "keyword_score": round(keyword_score, 2),
            "keyword_correct": keyword_correct,
            "keyword_time_ms": round(keyword_ms, 2)
        })

        print(f"\nQuestion: {question}")
        print(f"  Expected: {expected}")
        print(f"  Semantic: {semantic_top} (score {semantic_score:.4f}) "
              f"- {'CORRECT' if semantic_correct else 'MISS/TIE'}")
        print(f"  Keyword:  {keyword_top} (score {keyword_score:.2f}) "
              f"- {'CORRECT' if keyword_correct else 'MISS/TIE'}")

    n = len(results)
    semantic_accuracy = sum(r["semantic_correct"] for r in results) / n * 100
    keyword_accuracy = sum(r["keyword_correct"] for r in results) / n * 100
    semantic_avg_ms = sum(r["semantic_time_ms"] for r in results) / n
    keyword_avg_ms = sum(r["keyword_time_ms"] for r in results) / n

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n========== RETRIEVAL BENCHMARK SUMMARY ==========")
    print(f"Labelled questions: {n}")
    print(f"Semantic top-1 accuracy: {semantic_accuracy:.1f}%")
    print(f"Keyword top-1 accuracy:  {keyword_accuracy:.1f}%")
    print(f"Average semantic retrieval time: {semantic_avg_ms:.2f} ms")
    print(f"Average keyword retrieval time:  {keyword_avg_ms:.2f} ms")
    print(f"Results saved to: {OUTPUT_FILE.relative_to(ROOT)}")
    print(
        "Note: timings are from this environment; semantic timing includes "
        "query embedding generation. This small two-document test is "
        "preliminary and does not measure RAG answer quality."
    )


if __name__ == "__main__":
    main()
