# ================================================================
# INTENTSTORE - KNOWLEDGE GRAPH (Neo4j AuraDB)
# ================================================================
#
# PURPOSE:
# Stores documents, the entities/projects extracted from them, and
# the relationships between everything, in a real graph database.
#
# GRAPH MODEL:
#   (:Document {filename})
#   (:Entity {name, type})
#   (:Document)-[:MENTIONS]->(:Entity)
#   (:Entity)-[:<RELATION>]->(:Entity)   e.g. USES, PART_OF
#   (:Document)-[:RELATED_DOC {score, shared}]->(:Document)
#
# SETUP:
#   pip install neo4j python-dotenv
#   Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env
#   or as Codespaces secrets.
# ================================================================

import os

try:
    from dotenv import load_dotenv
    load_dotenv("backend/.env", override=True)
except Exception:
    pass

from neo4j import GraphDatabase


class KnowledgeGraph:

    def __init__(self):

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            raise ValueError(
                "NEO4J_URI and NEO4J_PASSWORD must be set "
                "(in .env or as Codespaces secrets)."
            )

        self.driver = GraphDatabase.driver(
            uri, auth=(username, password)
        )

        # Uniqueness constraints keep re-running the build
        # script from creating duplicate nodes.
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    # ============================================================
    # SETUP
    # ============================================================

    def _ensure_constraints(self):

        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",

            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]

        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    # ============================================================
    # ADD A DOCUMENT'S EXTRACTED KNOWLEDGE TO THE GRAPH
    # ============================================================

    def add_document_knowledge(self, extraction_result):
        """
        extraction_result is the dict returned by
        EntityExtractor.extract(), i.e.
        {"filename": ..., "entities": [...], "relationships": [...]}
        """

        filename = extraction_result["filename"]
        document_id = extraction_result["document_id"]
        entities = extraction_result.get("entities", [])
        relationships = extraction_result.get("relationships", [])

        with self.driver.session() as session:

            # --- Document node ---
            session.run(
                "MERGE (d:Document {document_id: $document_id})",
                document_id=document_id
            )

            # --- Entity nodes + Document -> Entity edges ---
            for entity in entities:
                name = str(entity.get("name", "")).strip()
                etype = str(entity.get("type", "TOPIC")).strip().upper()

                if not name:
                    continue

                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $etype
                    WITH e
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (d)-[:MENTIONS]->(e)
                    """,
                    name=name,
                    etype=etype,
                    document_id=document_id
                )

            # --- Entity <-> Entity relationships ---
            for rel in relationships:
                source = str(rel.get("source", "")).strip()
                target = str(rel.get("target", "")).strip()
                relation = str(rel.get("relation", "RELATED_TO"))
                relation = relation.strip().upper().replace(" ", "_")

                if not source or not target:
                    continue

                # Neo4j doesn't support parameterized relationship
                # types, so we sanitize to letters/digits/underscores
                # before interpolating.
                safe_relation = "".join(
                    ch for ch in relation if ch.isalnum() or ch == "_"
                ) or "RELATED_TO"

                query = f"""
                    MERGE (a:Entity {{name: $source}})
                    MERGE (b:Entity {{name: $target}})
                    MERGE (a)-[:{safe_relation}]->(b)
                """

                session.run(query, source=source, target=target)

        print(f"Added knowledge from {filename} to the graph.")

    # ============================================================
    # QUERIES: ENTITY-BASED
    # ============================================================

    def get_related_documents(self, entity_name, limit=10):
        """Documents that mention an entity (partial name match)."""

        query = """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($entity_name)
            RETURN DISTINCT d.filename AS filename
            LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query, entity_name=entity_name, limit=limit
            )
            return [record["filename"] for record in result]

    def get_document_entities(self, filename):
        """List all entities mentioned in a specific document."""

        query = """
            MATCH (d:Document {document_id: $document_id})-[:MENTIONS]->(e:Entity)
            RETURN e.name AS name, e.type AS type
        """

        with self.driver.session() as session:
            result = session.run(query, filename=filename)
            return [
                {"name": r["name"], "type": r["type"]} for r in result
            ]

    def get_related_entities(self, entity_name, limit=10):
        """Entities connected to a given entity via any relationship."""

        query = """
            MATCH (a:Entity)-[r]-(b:Entity)
            WHERE toLower(a.name) CONTAINS toLower($entity_name)
            RETURN DISTINCT b.name AS name, type(r) AS relation
            LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query, entity_name=entity_name, limit=limit
            )
            return [
                {"name": r["name"], "relation": r["relation"]}
                for r in result
            ]

    def get_graph_summary(self):
        """Quick stats, useful for a demo / dashboard."""

        query = """
            MATCH (d:Document) WITH count(d) AS documents
            MATCH (e:Entity) WITH documents, count(e) AS entities
            MATCH ()-[r]->() WITH documents, entities,
                count(r) AS relationships
            RETURN documents, entities, relationships
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return dict(record)
            return {"documents": 0, "entities": 0, "relationships": 0}

    # ============================================================
    # QUERIES: DOCUMENT-TO-DOCUMENT (objective 3)
    # ============================================================

    def get_related_documents_for_document(self, document_id, limit=10):
        """
        Documents related to a given document, ranked by how many
        entities they share. Also returns WHICH entities are shared,
        so results can be explained (objective 6).
        """

        query = """
            MATCH (d:Document {document_id: $document_id})-[:MENTIONS]->(e:Entity)
                  <-[:MENTIONS]-(other:Document)
            WHERE other.document_id <> $document_id
            WITH other, collect(DISTINCT {name: e.name, type: e.type}) AS shared
            RETURN other.document_id AS document_id,
                   other.filename AS filename,
                   size(shared) AS shared_count,
                   shared AS shared_entities
            ORDER BY shared_count DESC
            LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, document_id=document_id, limit=limit)
            return [
                {
                    "filename": r["filename"],
                    "document_id": r["document_id"],
                    "shared_count": r["shared_count"],
                    "shared_entities": r["shared_entities"],
                }
                for r in result
            ]

    def link_related_documents(self, min_shared=2):
        """
        Store Document-[:RELATED_DOC]->Document edges for every pair
        sharing at least `min_shared` entities. Run after the build.
        """

        query = """
            MATCH (a:Document)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(b:Document)
            WHERE a.document_id < b.document_id
            WITH a, b, collect(DISTINCT e.name) AS shared
            WHERE size(shared) >= $min_shared
            MERGE (a)-[r:RELATED_DOC]->(b)
            SET r.score = size(shared), r.shared = shared
            RETURN count(r) AS links
        """

        with self.driver.session() as session:
            record = session.run(query, min_shared=min_shared).single()
            return record["links"] if record else 0


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    kg = KnowledgeGraph()

    print("\nGraph summary:", kg.get_graph_summary())

    print("Links created:", kg.link_related_documents(min_shared=2))

    fn = input("\nDocument filename to find related docs for (e.g. 1.pdf)? ")
    for r in kg.get_related_documents_for_document(fn):
        names = [e["name"] for e in r["shared_entities"]]
        print(f" - {r['filename']} ({r['shared_count']} shared): {names}")

    entity = input("\nSearch for documents about which entity? ")

    docs = kg.get_related_documents(entity)
    print(f"\nDocuments mentioning '{entity}':")
    for doc in docs:
        print(" -", doc)

    related = kg.get_related_entities(entity)
    print(f"\nEntities related to '{entity}':")
    for r in related:
        print(f" - {r['name']}  ({r['relation']})")

    kg.close()