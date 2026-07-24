import enum
import functools
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from sqlalchemy import Engine, create_engine

from private_gpt.utils.dependencies import format_missing_dependency_message


class DatabaseDialect(enum.StrEnum):
    """Database family, independent of the exact SQLAlchemy driver name.

    Extend this (and _DIALECT_NAME_MAP below) when adding support for a new
    database family -- consumers should switch on this enum rather than
    comparing raw dialect-name strings, so a new family is a single place to
    wire up instead of a set of ad hoc string-list checks scattered around.
    """

    POSTGRES = "postgres"
    MYSQL = "mysql"
    MSSQL = "mssql"
    DB2 = "db2"
    UNKNOWN = "unknown"


_DIALECT_NAME_MAP: dict[str, DatabaseDialect] = {
    "postgresql": DatabaseDialect.POSTGRES,
    "postgres": DatabaseDialect.POSTGRES,
    "mysql": DatabaseDialect.MYSQL,
    "mssql": DatabaseDialect.MSSQL,
    "microsoft": DatabaseDialect.MSSQL,
    "db2": DatabaseDialect.DB2,
    "ibm_db_sa": DatabaseDialect.DB2,
}

_URL_PASSWORD_RE = re.compile(r"(://[^:/?#@]+:)([^@/?#]+)(@)")
_DB2_BRACED_PWD_RE = re.compile(r"(PWD=)\{((?:[^}]|}})*)\}(;)", re.IGNORECASE)
_DB2_PLAIN_PWD_RE = re.compile(r"(PWD=)(?!\{)([^;]*)(;)", re.IGNORECASE)
_DB2_VALUE_NEEDS_ESCAPING_RE = re.compile(r"[;{}]|^\s|\s$")


@functools.cache
def _load_ibm_db_dbi() -> Any:
    try:
        import ibm_db_dbi  # type: ignore[import-not-found,import-untyped]
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "DB2 database query",
                extras=("database-db2", "database"),
            )
        ) from e

    return ibm_db_dbi


def classify_dialect(dialect_name: str | None) -> DatabaseDialect:
    """Classify a dialect/scheme identifier into a DatabaseDialect."""
    return _DIALECT_NAME_MAP.get((dialect_name or "").lower(), DatabaseDialect.UNKNOWN)


def is_db2_connection_string(connection_string: str) -> bool:
    """Whether a SQLAlchemy-style connection string targets DB2."""
    scheme = connection_string.split("://", 1)[0].split("+", 1)[0]
    return classify_dialect(scheme) is DatabaseDialect.DB2


def escape_db2_cli_value(value: str) -> str:
    """Escape a value for a DB2 CLI/db2dsdriver keyword=value; connection string."""
    if not _DB2_VALUE_NEEDS_ESCAPING_RE.search(value):
        return value
    return "{" + value.replace("}", "}}") + "}"


def build_db2_native_connection_string(connection_string: str) -> str:
    """Build a native DB2 CLI connection string from a SQLAlchemy-style URL."""
    remainder = connection_string.split("://", 1)[-1]
    parsed = urlparse(f"//{remainder}")
    params = parse_qs(parsed.query, keep_blank_values=True)

    # urlparse does not percent-decode username/password/path, unlike parse_qs
    database = unquote(parsed.path.lstrip("/"))
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")

    parts = [f"DATABASE={database}", f"HOSTNAME={parsed.hostname or ''}"]
    if parsed.port:
        parts.append(f"PORT={parsed.port}")
    parts.append("PROTOCOL=TCPIP")
    parts.append(f"UID={escape_db2_cli_value(username)}")
    parts.append(f"PWD={escape_db2_cli_value(password)}")

    for key, values in params.items():
        parts.append(f"{key.upper()}={escape_db2_cli_value(values[-1])}")

    return ";".join(parts) + ";"


def mask_connection_secrets(text: str) -> str:
    """Mask any password embedded in a connection string or error message."""
    masked = _URL_PASSWORD_RE.sub(r"\1***\3", text)
    masked = _DB2_BRACED_PWD_RE.sub(r"\1{***}\3", masked)
    masked = _DB2_PLAIN_PWD_RE.sub(r"\1***\3", masked)
    return masked


def create_engine_for_connection_string(connection_string: str) -> Engine:
    """Create a SQLAlchemy engine for a connection string."""
    if is_db2_connection_string(connection_string):
        return create_engine(
            "db2+ibm_db://",
            creator=lambda: _load_ibm_db_dbi().connect(
                build_db2_native_connection_string(connection_string), "", ""
            ),
        )
    return create_engine(connection_string)
