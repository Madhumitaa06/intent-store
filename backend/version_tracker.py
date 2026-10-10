"""
Objective 5: Track file versions
---------------------------------
Reads (never modifies):
    data/index/organization.json   (near/exact duplicate links from auto_organizer.py)
    data/documents/                (original files, for modified-date ordering)

Writes ONE new file:
    data/index/versions.json

Run from the project root (the intent-store-main folder):
    python3 backend/version_tracker.py
"""

import json
import os
from pathlib import Path

ORGANIZATION_FILE = Path("data/index/organization.json")
DOCUMENTS_DIR = Path("data/documents")
OUTPUT_FILE = Path("data/index/versions.json")


def find_chains(organization):
    """Group files into version chains using their duplicate links."""
    parent = {name: name for name in organization}

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
        if info.get("exact_duplicate_of"):
            union(name, info["exact_duplicate_of"])
        for near in info.get("near_duplicates", []):
            union(name, near["filename"])

    chains = {}
    for name in organization:
        root = find(name)
        chains.setdefault(root, []).append(name)

    return chains


def file_modified_time(filename):
    path = DOCUMENTS_DIR / filename
    if path.exists():
        return os.path.getmtime(path)
    return 0  # unknown files sort first


def track_versions():
    print("Loading organization data...")
    with open(ORGANIZATION_FILE, "r", encoding="utf-8") as f:
        organization = json.load(f)

    chains = find_chains(organization)

    result = {}
    chain_count = 0
    for members in chains.values():
        if len(members) < 2:
            continue  # not a version chain, just a single unrelated file
        chain_count += 1
        chain_id = f"chain_{chain_count}"

        ordered = sorted(members, key=file_modified_time)

        for i, name in enumerate(ordered, start=1):
            result[name] = {
                "chain_id": chain_id,
                "version": i,
                "total_versions": len(ordered),
                "is_latest": i == len(ordered),
                "previous_version": ordered[i - 2] if i > 1 else None,
                "next_version": ordered[i] if i < len(ordered) else None,
            }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("\nVersion tracking complete!")
    print(f"Version chains found: {chain_count}")
    print(f"Files in a chain: {len(result)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    track_versions()