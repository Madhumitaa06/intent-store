# ================================================================
# INTENTSTORE - BUILD KNOWLEDGE GRAPH
# ================================================================
#
# PURPOSE:
# Runs the full objective-3 pipeline:
#   1. Reads the documents parsed by document_parser.py
#      (data/index/documents.json)
#   2. For each document, asks the LLM (via EntityExtractor) to pull
#      out entities, projects, and relationships
#   3. Normalizes entity names so spelling/case variants merge
#   4. Writes everything into the Neo4j knowledge graph
#   5. Links related documents (shared entities -> RELATED_DOC)
#
# USAGE:
#   python backend/build_knowledge_graph.py
# ================================================================

import json
import re
from pathlib import Path

from entity_extractor import EntityExtractor
from knowledge_graph import KnowledgeGraph


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "index" / "documents.json"

# Two documents are linked if they share at least this many entities.
MIN_SHARED_ENTITIES = 2


def combine_document_text(document, max_pages=3):
    """
    Combine the first few pages of a document into one block of
    text. Titles, abstracts and intros (where entity/project names
    usually live) tend to be in the first pages, so this keeps LLM
    calls fast and cheap without losing the important context.
    """

    pages = document.get("pages", [])[:max_pages]
    text_parts = [page.get("text", "") for page in pages]

    return "\n\n".join(text_parts)


def clean_name(name):
    """Collapse whitespace and strip stray punctuation."""

    name = re.sub(r"\s+", " ", str(name)).strip()
    return name.strip(" .,:;\"'()[]")


class NameCanonicalizer:
    """
    Makes 'FAISS', 'faiss' and ' Faiss ' resolve to ONE spelling
    (the first one seen), so they become a single graph node.
    """

    def __init__(self):
        self.seen = {}

    def canonical(self, name):
        name = clean_name(name)
        if not name:
            return ""
        return self.seen.setdefault(name.lower(), name)


def normalize_result(result, canon):
    """Apply canonical names to entities and relationships."""

    entities = []
    seen_names = set()

    for entity in result.get("entities", []):
        name = canon.canonical(entity.get("name", ""))
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        entities.append({"name": name, "type": entity.get("type", "TOPIC")})

    relationships = []

    for rel in result.get("relationships", []):
        source = canon.canonical(rel.get("source", ""))
        target = canon.canonical(rel.get("target", ""))
        if not source or not target or source.lower() == target.lower():
            continue
        relationships.append({
            "source": source,
            "relation": rel.get("relation", "RELATED_TO"),
            "target": target,
        })

    return {
        "filename": result["filename"],
        "document_id": result["document_id"],
        "entities": entities,
        "relationships": relationships,
    }


def build_graph():

    if not DOCUMENTS_FILE.exists():
        print(
            f"Could not find {DOCUMENTS_FILE}. "
            "Run document_parser.py first."
        )
        return

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Found {len(documents)} documents to process.\n")

    extractor = EntityExtractor()
    graph = KnowledgeGraph()
    canon = NameCanonicalizer()

    try:

        for i, document in enumerate(documents, start=1):

            filename = document["filename"]
            text = combine_document_text(document)

            if not text.strip():
                print(f"[{i}/{len(documents)}] {filename}: no text, skipping")
                continue

            print(f"[{i}/{len(documents)}] Extracting entities from {filename}...")

            result = extractor.extract(filename, text, document["document_id"])
            result = normalize_result(result, canon)

            n_entities = len(result["entities"])
            n_relationships = len(result["relationships"])

            print(f"    Found {n_entities} entities, {n_relationships} relationships")

            graph.add_document_knowledge(result)

        # ----- Objective 3: link related documents -----
        links = graph.link_related_documents(min_shared=MIN_SHARED_ENTITIES)

        summary = graph.get_graph_summary()

        print("\n--------------------------------")
        print("KNOWLEDGE GRAPH BUILD COMPLETE")
        print("--------------------------------")
        print(f"Documents in graph:      {summary['documents']}")
        print(f"Entities in graph:       {summary['entities']}")
        print(f"Relationships in graph:  {summary['relationships']}")
        print(f"Related-document links:  {links}")

    finally:
        graph.close()


if __name__ == "__main__":
    build_graph()
    