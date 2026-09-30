"""
Objective 7: Secure login and personal workspace

Provides:
- User signup
- User login with JWT
- Token verification
- Document ownership per user

Storage:
backend/data/users.db
"""

import sqlite3
import bcrypt
import jwt
import datetime
from pathlib import Path


# ================================================================
# DATABASE
# ================================================================

DB_FILE = Path(__file__).resolve().parent / "data" / "users.db"

SECRET_KEY = "change-this-to-something-random-and-long"

TOKEN_EXPIRY_HOURS = 24


def get_connection():
    DB_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(DB_FILE)


# ================================================================
# INITIALIZE DATABASE
# ================================================================

def init_db():
    """
    Creates the users and documents tables if they don't exist.
    Safe to call every time the app starts.
    """

    conn = get_connection()

    # ------------------------------------------------------------
    # USERS TABLE
    # ------------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # ------------------------------------------------------------
    # DOCUMENT OWNERSHIP TABLE
    # ------------------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(filename, user_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ================================================================
# CREATE USER
# ================================================================

def create_user(email, password):
    """
    Registers a new user.

    Returns:
        (True, "Account created.")
        or
        (False, error message)
    """

    email = email.strip().lower()

    if not email or "@" not in email:
        return False, "Please enter a valid email."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO users
            (email, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (
                email,
                password_hash,
                datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
            )
        )

        conn.commit()

        return True, "Account created."

    except sqlite3.IntegrityError:
        return False, (
            "An account with this email already exists."
        )

    finally:
        conn.close()


# ================================================================
# LOGIN
# ================================================================

def check_login(email, password):
    """
    Verifies email and password.

    Returns:
        (True, JWT token)
        or
        (False, error message)
    """

    email = email.strip().lower()

    conn = get_connection()

    row = conn.execute(
        """
        SELECT id, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    conn.close()

    if row is None:
        return False, "No account with this email."

    user_id, password_hash = row

    if not bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    ):
        return False, "Incorrect password."

    token = jwt.encode(
        {
            "user_id": user_id,
            "email": email,
            "exp": (
                datetime.datetime.now(
                    datetime.timezone.utc
                )
                + datetime.timedelta(
                    hours=TOKEN_EXPIRY_HOURS
                )
            )
        },
        SECRET_KEY,
        algorithm="HS256"
    )

    return True, token


# ================================================================
# GET USER FROM TOKEN
# ================================================================

def get_user_from_token(token):
    """
    Decodes a JWT token.

    Returns:
        {"user_id": ..., "email": ...}

    or None if the token is missing, invalid,
    or expired.
    """

    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"]
        )

        return {
            "user_id": payload["user_id"],
            "email": payload["email"]
        }

    except jwt.ExpiredSignatureError:
        return None

    except jwt.InvalidTokenError:
        return None


# ================================================================
# GET USER'S DOCUMENTS
# ================================================================

def get_user_documents(user_id):
    """
    Returns the filenames belonging to a specific user.

    Example:
        get_user_documents(1)

    returns:
        {"AI.pdf", "ML.pdf"}
    """

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT filename
        FROM documents
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    return {
        row[0]
        for row in rows
    }


# ================================================================
# ADD DOCUMENT FOR USER
# ================================================================

def add_user_document(user_id, filename):
    """
    Associates a document with a user.

    Returns:
        True if added successfully.
        False if the document is already assigned.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO documents
            (filename, user_id)
            VALUES (?, ?)
            """,
            (
                filename,
                user_id
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


# ================================================================
# INITIALIZE DATABASE ON IMPORT
# ================================================================

init_db()


# ================================================================
# SIMPLE TEST
# ================================================================

if __name__ == "__main__":

    print("Testing auth.py...")

    ok, msg = create_user(
        "test@example.com",
        "password123"
    )

    print(
        "create_user:",
        ok,
        msg
    )

    ok, token_or_msg = check_login(
        "test@example.com",
        "password123"
    )

    print(
        "check_login:",
        ok,
        token_or_msg
    )

    if ok:
        user = get_user_from_token(
            token_or_msg
        )

        print(
            "get_user_from_token:",
            user
        )

        if user:
            print(
                "User documents:",
                get_user_documents(
                    user["user_id"]
                )
            )

