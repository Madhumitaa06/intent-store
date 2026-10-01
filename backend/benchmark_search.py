"""
Objective 8: Prove the system works better than keyword search
-----------------------------------------------------------------
Compares semantic search (sentence-transformers embeddings) against
plain keyword search, using an auto-generated test set built from
your own documents' real keywords.

Test set logic:
    For each document, a query is built from a few of its own
    top keywords (from document_metadata.json). The "correct answer"
    for that query is simply: this document. A good search system
    should be able to find a document from a handful of its own
    keywords, even if they're not phrased as a natural sentence.

Reads (never modifies):
    data/index/chunks.json
    data/index/document_metadata.json

Writes ONE new file:
    data/index/benchmark_results.json

Run from the project root (the intent-store-main folder):
    python3 backend/benchmark_search.py
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/index/chunks.json")
METADATA_FILE = Path("data/index/document_metadata.json")
OUTPUT_FILE = Path("data/index/benchmark_results.json")

MODEL_NAME = "all-MiniLM-L6-v2"
KEYWORDS_PER_QUERY = 4   # how many keywords form one test query
TOP_K = 5                # how deep to check for Accuracy@3 etc.


# ================================================================
# LOAD DATA
# ================================================================

def load_documents():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk["filename"]].append(chunk["text"])

    full_texts = {
        name: " ".join(texts)
        for name, texts in documents.items()
    }

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return full_texts, metadata


# ================================================================
# BUILD TEST SET
# ================================================================

def build_test_set(metadata):
    """
    One test case per document: a short query made of that
    document's own keywords, with the correct answer being
    that same document.
    """
    test_cases = []

    for filename, info in metadata.items():
        keywords = info.get("keywords", [])[:KEYWORDS_PER_QUERY]

        if len(keywords) < 2:
            continue  # not enough signal to make a fair query

        query = " ".join(keywords)

        test_cases.append({
            "query": query,
            "expected": filename
        })

    return test_cases


# ================================================================
# KEYWORD SEARCH (BASELINE)
# ================================================================

def tokenize(text):
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())


def keyword_search(query, full_texts):
    query_words = set(tokenize(query))

    scores = {}
    for filename, text in full_texts.items():
        doc_words = tokenize(text)
        if not doc_words:
            scores[filename] = 0.0
            continue
        doc_word_set = set(doc_words)
        overlap = len(query_words & doc_word_set)
        scores[filename] = overlap / len(query_words) if query_words else 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [filename for filename, _ in ranked]


# ================================================================
# SEMANTIC SEARCH
# ================================================================

def build_document_vectors(model, full_texts):
    filenames = list(full_texts.keys())
    texts = [full_texts[name] for name in filenames]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return filenames, np.asarray(embeddings, dtype="float32")


def semantic_search(model, query, filenames, doc_vectors):
    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    )
    query_vec = np.asarray(query_vec, dtype="float32")[0]

    scores = doc_vectors @ query_vec
    ranked_indices = np.argsort(-scores)

    return [filenames[i] for i in ranked_indices]


# ================================================================
# METRICS
# ================================================================

def evaluate(results_per_query, top_k=TOP_K):
    """
    results_per_query: list of (expected_filename, ranked_filenames, time_taken)
    """
    n = len(results_per_query)
    if n == 0:
        return {}

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    total_time = 0.0

    for expected, ranked, time_taken in results_per_query:
        total_time += time_taken

        if expected in ranked:
            rank = ranked.index(expected) + 1
        else:
            rank = None

        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1

        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "queries_tested": n,
        "accuracy_at_1": round(hits_at_1 / n, 4),
        "accuracy_at_3": round(hits_at_3 / n, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / n, 4),
        "avg_query_time_ms": round((total_time / n) * 1000, 2),
    }


# ================================================================
# MAIN
# ================================================================

def run_benchmark():
    print("Loading documents...")
    full_texts, metadata = load_documents()

    if not metadata:
        print(
            "No document_metadata.json found or it is empty. "
            "Run backend/metadata_generator.py first."
        )
        return

    test_cases = build_test_set(metadata)
    print(f"Built {len(test_cases)} test queries from document keywords.\n")

    if not test_cases:
        print("Not enough keyword data to build a test set.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building document vectors for semantic search...")
    filenames, doc_vectors = build_document_vectors(model, full_texts)

    semantic_results = []
    keyword_results = []

    print("\nRunning benchmark...\n")

    for i, case in enumerate(test_cases, start=1):
        query = case["query"]
        expected = case["expected"]

        print(f"[{i}/{len(test_cases)}] Query: \"{query}\"  (expecting {expected})")

        start = time.perf_counter()
        sem_ranked = semantic_search(model, query, filenames, doc_vectors)
        sem_time = time.perf_counter() - start

        start = time.perf_counter()
        kw_ranked = keyword_search(query, full_texts)
        kw_time = time.perf_counter() - start

        semantic_results.append((expected, sem_ranked, sem_time))
        keyword_results.append((expected, kw_ranked, kw_time))

    semantic_metrics = evaluate(semantic_results)
    keyword_metrics = evaluate(keyword_results)

    report = {
        "test_queries": len(test_cases),
        "semantic_search": semantic_metrics,
        "keyword_search": keyword_metrics,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<28}{'Semantic Search':<20}{'Keyword Search':<20}")
    print("-" * 68)

    for key, label in [
        ("accuracy_at_1", "Accuracy @1"),
        ("accuracy_at_3", "Accuracy @3"),
        ("mean_reciprocal_rank", "Mean Reciprocal Rank"),
        ("avg_query_time_ms", "Avg time per query (ms)"),
    ]:
        sem_val = semantic_metrics.get(key, "-")
        kw_val = keyword_metrics.get(key, "-")
        print(f"{label:<28}{str(sem_val):<20}{str(kw_val):<20}")

    print("\nSaved full report to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_benchmark()
