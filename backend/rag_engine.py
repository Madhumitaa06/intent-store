# ================================================================
# INTENTSTORE - RAG ENGINE
# ================================================================

from pathlib import Path

from search_engine import IntentStoreSearch


# ================================================================
# RAG ENGINE
# ================================================================

class RAGEngine:

    def __init__(self):

        print("\n" + "=" * 70)
        print("INITIALIZING INTENTSTORE RAG ENGINE")
        print("=" * 70)

        # --------------------------------------------------------
        # Connect search engine
        # --------------------------------------------------------

        self.search_engine = IntentStoreSearch()

        print(
            "\nSearch engine connected to RAG engine."
        )

        print("\n" + "=" * 70)
        print("RAG ENGINE READY")
        print("=" * 70)

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question,
        top_k=5
    ):

        question = str(
            question
        ).strip()

        if not question:

            return {

                "answer":
                    "Please enter a question.",

                "results": []

            }

        print("\n" + "=" * 70)

        print(
            "RAG QUESTION:"
        )

        print(question)

        print("=" * 70)

        # --------------------------------------------------------
        # Retrieve evidence
        # --------------------------------------------------------

        results = self.search_engine.search(

            question,

            top_k=top_k

        )

        # --------------------------------------------------------
        # No evidence
        # --------------------------------------------------------

        if not results:

            return {

                "answer":
                    (
                        "No relevant information was found "
                        "in the stored documents."
                    ),

                "results": []

            }

        # --------------------------------------------------------
        # Build answer
        # --------------------------------------------------------

        answer = self._generate_answer(

            question,

            results

        )

        return {

            "answer": answer,

            "results": results

        }

    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    def _generate_answer(
        self,
        question,
        results
    ):

        # --------------------------------------------------------
        # Take best evidence
        # --------------------------------------------------------

        best_results = results[:3]

        evidence_parts = []

        for result in best_results:

            text = str(
                result.get(
                    "text",
                    ""
                )
            ).strip()

            if text:

                evidence_parts.append(
                    text
                )

        if not evidence_parts:

            return (
                "Relevant documents were found, "
                "but they do not contain readable text."
            )

        # --------------------------------------------------------
        # Combine evidence
        # --------------------------------------------------------

        combined = " ".join(
            evidence_parts
        )

        # Clean whitespace
        combined = " ".join(
            combined.split()
        )

        # --------------------------------------------------------
        # Limit response size
        # --------------------------------------------------------

        max_length = 1500

        if len(combined) > max_length:

            combined = (
                combined[:max_length]
                + "..."
            )

        # --------------------------------------------------------
        # Answer
        # --------------------------------------------------------

        answer = (

            "Based on the retrieved documents:\n\n"

            + combined

        )

        return answer


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    rag = RAGEngine()

    question = input(
        "\nAsk a question: "
    )

    response = rag.ask(
        question
    )

    print("\nANSWER")
    print(
        response["answer"]
    )

    print("\nSOURCES")

    for result in response["results"]:

        print(
            result.get(
                "filename"
            ),
            "- page",
            result.get(
                "page_number"
            )
        )