import re


def tokenize(text):
    """
    Convert text into meaningful lowercase words.
    """

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    return set(words)


def keyword_overlap(query, document_text):
    """
    Measure how many query terms appear in the text.
    """

    query_words = tokenize(query)
    document_words = tokenize(document_text)

    if not query_words:
        return 0.0

    common_words = query_words.intersection(document_words)

    return len(common_words) / len(query_words)


def metadata_relevance(query, metadata):
    """
    Measure how closely the query matches the
    document-level keywords.
    """

    if not metadata:
        return 0.0

    query_words = tokenize(query)

    document_keywords = set(
        metadata.get("keywords", [])
    )

    if not query_words or not document_keywords:
        return 0.0

    matching_words = query_words.intersection(
        document_keywords
    )

    return len(matching_words) / len(query_words)


def calculate_relevance(
    query,
    result,
    metadata=None
):
    """
    Calculate context-aware relevance.

    Current baseline:
    60% semantic similarity
    20% chunk keyword match
    20% document metadata relevance
    """

    semantic_score = result["similarity_score"]

    keyword_score = result["keyword_score"]

    metadata_score = metadata_relevance(
        query,
        metadata
    )

    relevance_score = (
        0.60 * semantic_score
        +
        0.20 * keyword_score
        +
        0.20 * metadata_score
    )

    return relevance_score


def rank_results(
    query,
    results,
    document_metadata=None
):

    if document_metadata is None:
        document_metadata = {}

    for result in results:

        # Chunk-level keyword relevance
        result["keyword_score"] = round(
            keyword_overlap(
                query,
                result["text"]
            ),
            4
        )

        # Document-level contextual relevance
        metadata = document_metadata.get(
            result["filename"],
            {}
        )

        result["metadata_score"] = round(
            metadata_relevance(
                query,
                metadata
            ),
            4
        )

        # Final relevance
        result["relevance_score"] = round(
            calculate_relevance(
                query,
                result,
                metadata
            ),
            4
        )

    # Highest relevance first
    results.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return results