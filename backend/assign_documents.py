"""
Assign existing documents to a user
-------------------------------------
One-time helper: gives a chosen account ownership of every file
currently in data/documents, so you have something to search
and demo with.

Run from the project root (the intent-store-main folder):
    python3 backend/assign_documents.py
"""

import sqlite3
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import auth  # noqa: E402

DOCUMENTS_DIR = BACKEND_DIR.parent / "data" / "documents"


def get_user_by_email(email):
    conn = auth.get_connection()
    row = conn.execute(
        "SELECT id, email FROM users WHERE email = ?",
        (email.strip().lower(),)
    ).fetchone()
    conn.close()
    return row


def list_users():
    conn = auth.get_connection()
    rows = conn.execute(
        "SELECT id, email FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def main():
    print("Users currently registered:")
    users = list_users()

    if not users:
        print("No users found. Sign up an account in the app first.")
        return

    for user_id, email in users:
        print(f"  id={user_id}  email={email}")

    email = input("\nEnter the email to assign documents to: ").strip()

    match = get_user_by_email(email)
    if not match:
        print("No user found with that email.")
        return

    user_id, matched_email = match

    if not DOCUMENTS_DIR.exists():
        print(f"Documents folder not found: {DOCUMENTS_DIR}")
        return

    files = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file()]

    if not files:
        print("No files found in data/documents.")
        return

    added = 0
    skipped = 0

    for f in files:
        ok = auth.add_user_document(user_id, f.name)
        if ok:
            added += 1
        else:
            skipped += 1

    print(f"\nDone. {added} documents assigned to {matched_email}.")
    print(f"{skipped} were already assigned and skipped.")


if __name__ == "__main__":
    main()
