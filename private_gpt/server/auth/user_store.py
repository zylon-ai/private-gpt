"""User, group, and collection management backed by SQLite."""

import hashlib
import logging
import secrets
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from injector import inject, singleton

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 310_000  # OWASP 2023 recommendation for PBKDF2-SHA256


@dataclass
class UserRecord:
    username: str
    is_admin: bool
    groups: list[str] = field(default_factory=list)


@dataclass
class GroupRecord:
    group_name: str
    collections: list[str] = field(default_factory=list)


@dataclass
class CollectionRecord:
    collection_name: str
    display_name: str


_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS groups (
    group_name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS user_groups (
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    group_name TEXT NOT NULL REFERENCES groups(group_name) ON DELETE CASCADE,
    PRIMARY KEY (username, group_name)
);
CREATE TABLE IF NOT EXISTS collections (
    collection_name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS group_collections (
    group_name TEXT NOT NULL REFERENCES groups(group_name) ON DELETE CASCADE,
    collection_name TEXT NOT NULL REFERENCES collections(collection_name) ON DELETE CASCADE,
    PRIMARY KEY (group_name, collection_name)
);
"""


@singleton
class UserStore:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._db_path: Path = local_data_path / "auth.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db(settings)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self, settings: Settings) -> None:
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLES_SQL)
            conn.commit()
            # Create default admin if no users exist
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count == 0:
                admin_password = settings.user_auth.default_admin_password
                logger.info(
                    "No users found — creating default admin user. "
                    "Change the password via the admin panel or settings."
                )
                self.create_user("admin", admin_password, is_admin=True)
                # Also create a default collection
                default_col = settings.user_auth.default_collection_name
                self.create_collection(default_col, display_name=default_col.capitalize())

    # ── Password helpers ─────────────────────────────────────────────────────

    @staticmethod
    def hash_password(plain: str) -> str:
        salt = secrets.token_hex(32)
        dk = hashlib.pbkdf2_hmac(
            "sha256", plain.encode(), salt.encode(), _PBKDF2_ITERATIONS
        )
        return f"{salt}:{dk.hex()}"

    @staticmethod
    def _verify_hash(password_hash: str, plain: str) -> bool:
        try:
            salt, stored_hex = password_hash.split(":", 1)
        except ValueError:
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", plain.encode(), salt.encode(), _PBKDF2_ITERATIONS
        )
        return secrets.compare_digest(dk.hex(), stored_hex)

    def verify_password(self, username: str, plain: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None:
            return False
        return self._verify_hash(row["password_hash"], plain)

    # ── User CRUD ────────────────────────────────────────────────────────────

    def create_user(
        self, username: str, plain_password: str, is_admin: bool = False
    ) -> UserRecord:
        pw_hash = self.hash_password(plain_password)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                (username, pw_hash, int(is_admin)),
            )
            conn.commit()
        return UserRecord(username=username, is_admin=is_admin)

    def delete_user(self, username: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()

    def change_password(self, username: str, new_plain_password: str) -> None:
        pw_hash = self.hash_password(new_plain_password)
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (pw_hash, username),
            )
            conn.commit()

    def list_users(self) -> list[UserRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username, is_admin FROM users ORDER BY username"
            ).fetchall()
            users = []
            for row in rows:
                groups = [
                    r["group_name"]
                    for r in conn.execute(
                        "SELECT group_name FROM user_groups WHERE username = ?",
                        (row["username"],),
                    ).fetchall()
                ]
                users.append(
                    UserRecord(
                        username=row["username"],
                        is_admin=bool(row["is_admin"]),
                        groups=groups,
                    )
                )
        return users

    def get_user(self, username: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, is_admin FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                return None
            groups = [
                r["group_name"]
                for r in conn.execute(
                    "SELECT group_name FROM user_groups WHERE username = ?", (username,)
                ).fetchall()
            ]
        return UserRecord(
            username=row["username"], is_admin=bool(row["is_admin"]), groups=groups
        )

    def is_admin(self, username: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_admin FROM users WHERE username = ?", (username,)
            ).fetchone()
        return bool(row["is_admin"]) if row else False

    # ── Group CRUD ───────────────────────────────────────────────────────────

    def create_group(self, group_name: str) -> GroupRecord:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO groups (group_name) VALUES (?)", (group_name,)
            )
            conn.commit()
        return GroupRecord(group_name=group_name)

    def delete_group(self, group_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM groups WHERE group_name = ?", (group_name,))
            conn.commit()

    def list_groups(self) -> list[GroupRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT group_name FROM groups ORDER BY group_name"
            ).fetchall()
            groups = []
            for row in rows:
                cols = [
                    r["collection_name"]
                    for r in conn.execute(
                        "SELECT collection_name FROM group_collections WHERE group_name = ?",
                        (row["group_name"],),
                    ).fetchall()
                ]
                groups.append(GroupRecord(group_name=row["group_name"], collections=cols))
        return groups

    def assign_user_to_group(self, username: str, group_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO user_groups (username, group_name) VALUES (?, ?)",
                (username, group_name),
            )
            conn.commit()

    def remove_user_from_group(self, username: str, group_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM user_groups WHERE username = ? AND group_name = ?",
                (username, group_name),
            )
            conn.commit()

    # ── Collection CRUD ──────────────────────────────────────────────────────

    def create_collection(
        self, collection_name: str, display_name: str = ""
    ) -> CollectionRecord:
        dn = display_name or collection_name
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO collections (collection_name, display_name) VALUES (?, ?)",
                (collection_name, dn),
            )
            conn.commit()
        return CollectionRecord(collection_name=collection_name, display_name=dn)

    def delete_collection(self, collection_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM collections WHERE collection_name = ?", (collection_name,)
            )
            conn.commit()

    def list_collections(self) -> list[CollectionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT collection_name, display_name FROM collections ORDER BY collection_name"
            ).fetchall()
        return [
            CollectionRecord(
                collection_name=r["collection_name"], display_name=r["display_name"]
            )
            for r in rows
        ]

    def assign_collection_to_group(
        self, group_name: str, collection_name: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO group_collections (group_name, collection_name) VALUES (?, ?)",
                (group_name, collection_name),
            )
            conn.commit()

    def remove_collection_from_group(
        self, group_name: str, collection_name: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM group_collections WHERE group_name = ? AND collection_name = ?",
                (group_name, collection_name),
            )
            conn.commit()

    # ── Access resolution ────────────────────────────────────────────────────

    def get_user_collections(self, username: str) -> list[str]:
        """Return all collection names accessible to the user.

        Admins can access every collection. Regular users inherit access from
        their group memberships.
        """
        with self._connect() as conn:
            is_admin_row = conn.execute(
                "SELECT is_admin FROM users WHERE username = ?", (username,)
            ).fetchone()
            if is_admin_row is None:
                return []
            if bool(is_admin_row["is_admin"]):
                # Admins see everything
                rows = conn.execute(
                    "SELECT collection_name FROM collections"
                ).fetchall()
                return [r["collection_name"] for r in rows]

            rows = conn.execute(
                """
                SELECT DISTINCT gc.collection_name
                FROM group_collections gc
                JOIN user_groups ug ON ug.group_name = gc.group_name
                WHERE ug.username = ?
                """,
                (username,),
            ).fetchall()
        return [r["collection_name"] for r in rows]
