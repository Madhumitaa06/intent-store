# ================================================================
# INTENTSTORE - RAG ENGINE (with real LLM generation)
# ================================================================
#
# CHANGE FROM PREVIOUS VERSION:
# The old _generate_answer() just concatenated raw retrieved chunk
# text. This version sends the retrieved chunks as CONTEXT to an
# LLM and asks it to compose a grounded natural-language answer,
# which is what "RAG" actually means.
#
# SETUP:
#   pip install openai python-dotenv
#   Create a .env file (or set an environment variable) with:
#       OPENAI_API_KEY=your_key_here
#       LLM_MODEL=gpt-4o-mini          (optional, this is the default)
#
# If no API key is found, the engine automatically falls back to
# the old extractive behavior so the app never crashes during a demo.
# ================================================================

import os
from pathlib import Path

from search_engine import IntentStoreSearch

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


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

        print("\nSearch engine connected to RAG engine.")

        # --------------------------------------------------------
        # Connect LLM client (optional)
        # --------------------------------------------------------

        self.llm_client = None
        self.llm_enabled = False

        api_key = os.getenv("OPENAI_API_KEY")

        if _OPENAI_AVAILABLE and api_key:
            try:
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_enabled = True
                print(f"LLM generation ENABLED (model: {LLM_MODEL}).")
            except Exception as error:
                print("Could not initialize LLM client:", error)
                self.llm_enabled = False
        else:
            print(
                "LLM generation DISABLED "
                "(no OPENAI_API_KEY found). "
                "Falling back to extractive answers."
            )

        print("\n" + "=" * 70)
        print("RAG ENGINE READY")
        print("=" * 70)

    # ============================================================
    # ASK
    # ============================================================

    def ask(self, question, top_k=5):

        question = str(question).strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "results": [],
                "generated_by_llm": False
            }

        print("\n" + "=" * 70)
        print("RAG QUESTION:")
        print(question)
        print("=" * 70)

        # --------------------------------------------------------
        # Retrieve evidence
        # --------------------------------------------------------

        results = self.search_engine.search(question, top_k=top_k)

        if not results:
            return {
                "answer": (
                    "No relevant information was found "
                    "in the stored documents."
                ),
                "results": [],
                "generated_by_llm": False
            }

        # --------------------------------------------------------
        # Generate answer (LLM if available, else extractive)
        # --------------------------------------------------------

        if self.llm_enabled:
            try:
                answer = self._generate_answer_llm(question, results)
                return {
                    "answer": answer,
                    "results": results,
                    "generated_by_llm": True
                }
            except Exception as error:
                print("LLM generation failed, falling back:", error)

        answer = self._generate_answer_extractive(results)

        return {
            "answer": answer,
            "results": results,
            "generated_by_llm": False
        }

    # ============================================================
    # LLM-BASED GENERATION (real RAG step)
    # ============================================================

    def _generate_answer_llm(self, question, results, top_n=5):

        context_blocks = []

        for i, result in enumerate(results[:top_n], start=1):
            filename = result.get("filename", "Unknown document")
            page = result.get("page_number", "Unknown")
            text = str(result.get("text", "")).strip()

            if not text:
                continue

            context_blocks.append(
                f"[Source {i}: {filename}, page {page}]\n{text}"
            )

        context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are IntentStore's document assistant. Answer the "
            "user's question using ONLY the information in the "
            "provided context. If the context does not contain "
            "enough information to answer, say so clearly instead "
            "of guessing. Keep the answer concise and cite which "
            "source(s) (by number) support each part of your answer."
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer the question using only the context above."
        )

        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

        return answer

    # ============================================================
    # EXTRACTIVE FALLBACK (old behavior, kept as a safety net)
    # ============================================================

    def _generate_answer_extractive(self, results, top_n=3):

        best_results = results[:top_n]

        evidence_parts = []

        for result in best_results:
            text = str(result.get("text", "")).strip()
            if text:
                evidence_parts.append(text)

        if not evidence_parts:
            return (
                "Relevant documents were found, "
                "but they do not contain readable text."
            )

        combined = " ".join(evidence_parts)
        combined = " ".join(combined.split())

        max_length = 1500
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."

        return (
            "[LLM unavailable — showing retrieved excerpts instead]\n\n"
            + combined
        )


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    rag = RAGEngine()

    question = input("\nAsk a question: ")

    response = rag.ask(question)

    print("\nGENERATED BY LLM:", response["generated_by_llm"])

    print("\nANSWER")
    print(response["answer"])

    print("\nSOURCES")

    for result in response["results"]:
        print(
            result.get("filename"),
            "- page",
            result.get("page_number")
        )
