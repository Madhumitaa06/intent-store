import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv("backend/.env", override=True)

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")


class EntityExtractor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            print("Entity extraction DISABLED (no GROQ_API_KEY found).")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            print(f"Entity extraction ENABLED (model: {LLM_MODEL}).")

    def extract(self, filename, text, document_id=None):
        if not self.client:
            return {
                "filename": filename,
                "document_id": document_id,
                "entities": [],
                "relationships": []
            }

        prompt = f"""
Extract important entities and relationships from this document.

Return ONLY valid JSON.
Do not use markdown.
Do not add comments.
Use exactly this structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "PERSON|ORGANIZATION|TECHNOLOGY|CONCEPT|LOCATION|TOPIC|OTHER"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "relation": "RELATION",
      "target": "entity name"
    }}
  ]
}}

Document filename:
{filename}

Document text:
{text}
"""

        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract entities and relationships and always return valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,
                    max_tokens=3000
                )

                raw = response.choices[0].message.content.strip()

                # Remove accidental markdown fences.
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw).strip()

                # Extract the JSON object if extra text was returned.
                start = raw.find("{")
                end = raw.rfind("}")

                if start != -1 and end != -1:
                    raw = raw[start:end + 1]

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # Ask the model to repair malformed JSON.
                    repair_response = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "Repair malformed JSON. Return ONLY valid JSON."
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Repair this into valid JSON while preserving "
                                    "the entities and relationships:\n\n" + raw
                                )
                            }
                        ],
                        temperature=0,
                        max_tokens=3000
                    )

                    repaired = repair_response.choices[0].message.content.strip()
                    repaired = re.sub(
                        r"^```(?:json)?\s*", "", repaired
                    )
                    repaired = re.sub(
                        r"\s*```$", "", repaired
                    ).strip()

                    start = repaired.find("{")
                    end = repaired.rfind("}")

                    if start != -1 and end != -1:
                        repaired = repaired[start:end + 1]

                    data = json.loads(repaired)

                entities = []
                for entity in data.get("entities", []):
                    if isinstance(entity, dict) and entity.get("name"):
                        entities.append({
                            "name": str(entity["name"]).strip(),
                            "type": str(entity.get("type", "OTHER")).strip()
                        })

                relationships = []
                for relationship in data.get("relationships", []):
                    if (
                        isinstance(relationship, dict)
                        and relationship.get("source")
                        and relationship.get("relation")
                        and relationship.get("target")
                    ):
                        relationships.append({
                            "source": str(relationship["source"]).strip(),
                            "relation": str(relationship["relation"]).strip(),
                            "target": str(relationship["target"]).strip()
                        })

                return {
                    "filename": filename,
                "document_id": document_id,
                    "entities": entities,
                    "relationships": relationships
                }

            except Exception as e:
                error_text = str(e)

                if "rate_limit" in error_text.lower() or "429" in error_text:
                    wait_time = 15 * (attempt + 1)
                    print(
                        f"Rate limit reached for {filename}. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue

                print(f"Entity extraction failed for {filename}: {e}")
                break

        return {
            "filename": filename,
                "document_id": document_id,
            "entities": [],
            "relationships": []
        }
