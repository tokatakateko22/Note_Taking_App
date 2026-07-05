import sqlite3
import json
import uuid
from datetime import datetime
import numpy as np
from src.models import Note

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'default',
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    tags        TEXT NOT NULL,          -- JSON array
    category    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    embedding   BLOB                   -- serialised np.float32 array
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, body, tags,
    content=notes, content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, body, tags) VALUES (new.rowid, new.title, new.body, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, body, tags) VALUES('delete', old.rowid, old.title, old.body, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, body, tags) VALUES('delete', old.rowid, old.title, old.body, old.tags);
  INSERT INTO notes_fts(rowid, title, body, tags) VALUES (new.rowid, new.title, new.body, new.tags);
END;

CREATE INDEX IF NOT EXISTS idx_notes_user    ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);
"""

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def close(self):
        if self.conn:
            self.conn.close()

    def _init_db(self):
        with self.conn as conn:
            conn.executescript(SCHEMA)

    def _row_to_note(self, row) -> Note:
        embedding = None
        if row[8]:
            embedding = np.frombuffer(row[8], dtype=np.float32)

        return Note(
            id=row[0],
            user_id=row[1],
            title=row[2],
            body=row[3],
            tags=json.loads(row[4]),
            category=row[5],
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            embedding=embedding
        )

    def insert(self, note: Note) -> Note:
        with self.conn as conn:
            embedding_blob = note.embedding.tobytes() if note.embedding is not None else None
            conn.execute(
                """
                INSERT INTO notes (id, user_id, title, body, tags, category, created_at, updated_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id, note.user_id, note.title, note.body, json.dumps(note.tags),
                    note.category, note.created_at.isoformat(), note.updated_at.isoformat(),
                    embedding_blob
                )
            )
        return note

    def get_by_id(self, note_id: str, user_id: str) -> Note | None:
        with self.conn as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
            row = cursor.fetchone()
            if row:
                return self._row_to_note(row)
        return None

    def fts_search(self, query: str, user_id: str, limit: int = 10) -> list[Note]:
        with self.conn as conn:
            sql = """
                SELECT n.* 
                FROM notes n
                JOIN notes_fts f ON n.rowid = f.rowid
                WHERE f.notes_fts MATCH ? AND n.user_id = ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = conn.execute(sql, (query, user_id, limit))
            results = [self._row_to_note(row) for row in cursor.fetchall()]

            # Case-insensitive fallback when FTS finds nothing (e.g. casing mismatches)
            if not results and query.strip():
                like = f"%{query.strip()}%"
                cursor = conn.execute(
                    """
                    SELECT * FROM notes
                    WHERE user_id = ?
                      AND (LOWER(title) LIKE LOWER(?) OR LOWER(body) LIKE LOWER(?))
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, like, like, limit),
                )
                results = [self._row_to_note(row) for row in cursor.fetchall()]

            return results

    def keyword_search(self, query: str, user_id: str, limit: int = 10) -> list[Note]:
        """Case-insensitive search across title and body."""
        with self.conn as conn:
            like = f"%{query.strip()}%"
            cursor = conn.execute(
                """
                SELECT * FROM notes
                WHERE user_id = ?
                  AND (LOWER(title) LIKE LOWER(?) OR LOWER(body) LIKE LOWER(?))
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, like, like, limit),
            )
            return [self._row_to_note(row) for row in cursor.fetchall()]

    def filter_by_tags(self, tags: list[str], user_id: str) -> list[Note]:
        if not tags:
            return self.get_all(user_id)
        with self.conn as conn:
            placeholders = ",".join("?" * len(tags))
            sql = f"""
                SELECT DISTINCT n.*
                FROM notes n, json_each(n.tags) j
                WHERE n.user_id = ? AND LOWER(j.value) IN ({placeholders})
            """
            params = [user_id] + [t.casefold() for t in tags]
            cursor = conn.execute(sql, params)
            return [self._row_to_note(row) for row in cursor.fetchall()]

    def filter_by_date_range(self, from_dt: str, to_dt: str, user_id: str) -> list[Note]:
        with self.conn as conn:
            sql = "SELECT * FROM notes WHERE user_id = ?"
            params = [user_id]
            if from_dt:
                sql += " AND created_at >= ?"
                params.append(from_dt)
            if to_dt:
                sql += " AND created_at <= ?"
                params.append(to_dt)
            sql += " ORDER BY created_at DESC"
            cursor = conn.execute(sql, params)
            return [self._row_to_note(row) for row in cursor.fetchall()]

    def update(self, note_id: str, patches: dict, user_id: str) -> Note | None:
        with self.conn as conn:
            # First ensure it exists
            cursor = conn.execute("SELECT * FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
            if not cursor.fetchone():
                return None
            
            patches['updated_at'] = datetime.now().isoformat()
            
            if 'tags' in patches and isinstance(patches['tags'], list):
                patches['tags'] = json.dumps(patches['tags'])
                
            if 'embedding' in patches and patches['embedding'] is not None:
                patches['embedding'] = patches['embedding'].tobytes()
            
            set_clause = ", ".join([f"{k} = ?" for k in patches.keys()])
            values = list(patches.values())
            
            sql = f"UPDATE notes SET {set_clause} WHERE id = ? AND user_id = ?"
            conn.execute(sql, values + [note_id, user_id])
            
        return self.get_by_id(note_id, user_id)

    def delete(self, note_id: str, user_id: str) -> bool:
        with self.conn as conn:
            cursor = conn.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id))
            return cursor.rowcount > 0

    def get_all(self, user_id: str, limit: int = 100) -> list[Note]:
        with self.conn as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?", (user_id, limit))
            return [self._row_to_note(row) for row in cursor.fetchall()]
