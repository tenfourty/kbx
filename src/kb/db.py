"""Database layer — SQLite (metadata + FTS5) and LanceDB (vectors).

LanceDB and PyArrow are imported lazily so that commands which only need
SQLite (entity queries, FTS search, context, etc.) work without the heavy
ML/vector dependencies installed.  This lets ``kb`` run on minimal Python
environments (e.g. Linux VMs, CI) for read-only operations.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    pass

EMBEDDING_DIM = 1024


def normalize_path(path: str) -> str:
    """Normalize a path string to NFC Unicode form.

    macOS stores filenames as NFD (decomposed), but CLI input and most
    other systems use NFC (precomposed). Normalizing to NFC ensures
    consistent lookups.
    """
    return unicodedata.normalize("NFC", path)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    title TEXT,
    doc_date TEXT,
    doc_type TEXT,
    source_system TEXT,
    source_id TEXT,
    tags TEXT,
    content_hash TEXT NOT NULL,
    file_mtime TEXT,
    chunk_count INTEGER DEFAULT 0,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases TEXT,
    metadata TEXT,
    source_path TEXT
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    mention_type TEXT NOT NULL,
    UNIQUE(entity_id, document_id, mention_type)
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_date ON documents(doc_date);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_document ON entity_mentions(document_id);

CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

# Each migration is (name, sql). SQL is executed idempotently (try/except).
# sql=None means a Python-based migration handled in _apply_migrations.
MIGRATIONS: list[tuple[str, str | None]] = [
    ("001_add_file_mtime", "ALTER TABLE documents ADD COLUMN file_mtime TEXT"),
    (
        "002_create_facts_table",
        """CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER REFERENCES entities(id),
            fact_text TEXT NOT NULL,
            fact_date TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
    ),
    ("003_add_entity_pinned", "ALTER TABLE entities ADD COLUMN pinned INTEGER DEFAULT 0"),
    (
        "004_create_document_attendees",
        """CREATE TABLE IF NOT EXISTS document_attendees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
            name TEXT NOT NULL,
            email TEXT,
            matched_by TEXT,
            UNIQUE(document_id, email)
        )""",
    ),
    ("005_normalize_paths_nfc", None),  # Python-based migration, handled in _apply_migrations
    (
        "006_unique_entity_name_type",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type ON entities(name, entity_type)",
    ),
    ("007_add_document_pinned", "ALTER TABLE documents ADD COLUMN pinned INTEGER DEFAULT 0"),
]


def _migrate_005_normalize_paths(conn: sqlite3.Connection) -> None:
    """Normalize all stored paths to NFC Unicode form."""
    # Normalize documents.path
    rows = conn.execute("SELECT id, path FROM documents").fetchall()
    for row in rows:
        nfc = normalize_path(row["path"] if isinstance(row, sqlite3.Row) else row[1])
        old = row["path"] if isinstance(row, sqlite3.Row) else row[1]
        if nfc != old:
            rid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
            conn.execute("UPDATE documents SET path = ? WHERE id = ?", (nfc, rid))

    # Normalize entities.source_path
    rows = conn.execute(
        "SELECT id, source_path FROM entities WHERE source_path IS NOT NULL"
    ).fetchall()
    for row in rows:
        old = row["source_path"] if isinstance(row, sqlite3.Row) else row[1]
        nfc = normalize_path(old)
        if nfc != old:
            rid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
            conn.execute("UPDATE entities SET source_path = ? WHERE id = ?", (nfc, rid))


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations. Idempotent."""
    applied = set()
    try:
        rows = conn.execute("SELECT name FROM migrations").fetchall()
        applied = {r[0] if isinstance(r, tuple) else r["name"] for r in rows}
    except Exception:
        return  # migrations table doesn't exist yet

    for name, sql in MIGRATIONS:
        if name in applied:
            continue
        if sql is None:
            # Python-based migration
            if name == "005_normalize_paths_nfc":
                _migrate_005_normalize_paths(conn)
        else:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                err_msg = str(e).lower()
                if "already exists" not in err_msg and "duplicate column" not in err_msg:
                    import sys

                    print(f"Migration {name} failed: {e}", file=sys.stderr)
                    raise
        conn.execute("INSERT OR IGNORE INTO migrations (name) VALUES (?)", (name,))
    conn.commit()


# FTS5 and triggers must be created separately (no IF NOT EXISTS for virtual tables)
FTS_SQL = """
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content, heading,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'
);
"""

TRIGGERS_SQL = [
    """CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content, heading) VALUES (new.id, new.content, new.heading);
    END;""",
    """CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, heading) VALUES ('delete', old.id, old.content, old.heading);
    END;""",
]

# LanceDB schema — built lazily to avoid top-level pyarrow import
_lance_schema = None


def get_lance_schema() -> Any:
    """Get the LanceDB PyArrow schema, building it on first call."""
    global _lance_schema
    if _lance_schema is None:
        import pyarrow as pa

        _lance_schema = pa.schema(
            [
                pa.field("chunk_id", pa.int64()),
                pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
                pa.field("doc_type", pa.utf8()),
                pa.field("doc_date", pa.utf8()),
                pa.field("tags", pa.utf8()),
                pa.field("document_id", pa.int64()),
                pa.field("entity_ids", pa.utf8()),
            ]
        )
    return _lance_schema


# Backward compat alias — existing code references LANCE_SCHEMA directly.
# This is a module-level property trick: on first access, it builds the schema.
def __getattr__(name: str) -> Any:
    if name == "LANCE_SCHEMA":
        return get_lance_schema()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Database:
    """Manages both SQLite and LanceDB connections."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._sqlite_path = data_dir / "metadata.db"
        self._lance_path = data_dir / "vectors"
        self._lance_path.mkdir(parents=True, exist_ok=True)

        self._sqlite_conn: sqlite3.Connection | None = None
        self._lance_db: Any | None = None  # lancedb.DBConnection (lazy)
        self._lance_table: Any | None = None  # lancedb.table.Table (lazy)

        # Initialize SQLite schema
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        conn = self.get_sqlite_conn()
        conn.executescript(SCHEMA_SQL)

        # Check if FTS table exists before creating
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if not existing:
            conn.executescript(FTS_SQL)
            for trigger_sql in TRIGGERS_SQL:
                conn.execute(trigger_sql)
            conn.commit()

        # Apply schema migrations
        _apply_migrations(conn)

    def get_sqlite_conn(self) -> sqlite3.Connection:
        if self._sqlite_conn is None:
            self._sqlite_conn = sqlite3.connect(str(self._sqlite_path))
            self._sqlite_conn.row_factory = sqlite3.Row
            self._sqlite_conn.execute("PRAGMA foreign_keys=ON")
            self._sqlite_conn.execute("PRAGMA synchronous=NORMAL")
            self._sqlite_conn.execute("PRAGMA cache_size=-64000")  # 64MB
            self._sqlite_conn.execute("PRAGMA busy_timeout=5000")
        return self._sqlite_conn

    def get_lance_db(self) -> Any:
        if self._lance_db is None:
            import lancedb

            self._lance_db = lancedb.connect(str(self._lance_path))
        return self._lance_db

    def get_lance_table(self) -> Any:
        """Get the chunks vector table, or None if it doesn't exist yet."""
        if self._lance_table is None:
            db = self.get_lance_db()
            if "chunks" in db.list_tables().tables:
                self._lance_table = db.open_table("chunks")
        return self._lance_table

    def ensure_lance_table(self, data: list[dict[str, Any]] | None = None) -> Any:
        """Create or open the chunks vector table. Creates with sample data if needed."""
        if self._lance_table is not None:
            return self._lance_table

        db = self.get_lance_db()
        if "chunks" in db.list_tables().tables:
            self._lance_table = db.open_table("chunks")
        elif data:
            self._lance_table = db.create_table("chunks", data=data, schema=get_lance_schema())
        else:
            # Create empty table with schema
            self._lance_table = db.create_table("chunks", schema=get_lance_schema())
        return self._lance_table

    def close(self) -> None:
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        self._lance_table = None
        self._lance_db = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
