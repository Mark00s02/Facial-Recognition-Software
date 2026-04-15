import sqlite3
import os
import pickle


class Database:
    def __init__(self, db_path="data/faces.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                encoding    BLOB    NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recognition_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                confidence  REAL,
                timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # ── Face CRUD ─────────────────────────────────────────────────────────────

    def add_face(self, name: str, encoding_blob: bytes):
        self.conn.execute(
            "INSERT INTO faces (name, encoding) VALUES (?, ?)",
            (name, encoding_blob),
        )
        self.conn.commit()

    def get_all_faces(self):
        """Returns list of (name, encoding_blob)."""
        cur = self.conn.execute("SELECT name, encoding FROM faces")
        return cur.fetchall()

    def get_face_names(self):
        """Returns one row per person: (name, sample_count, earliest_created_at)."""
        cur = self.conn.execute(
            "SELECT name, COUNT(*) as cnt, MIN(created_at) as first_seen "
            "FROM faces GROUP BY name ORDER BY name"
        )
        return cur.fetchall()

    def delete_person(self, name: str):
        """Delete all DB samples for a person by name."""
        self.conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        self.conn.commit()

    def delete_face(self, face_id: int):
        self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()

    def face_exists(self, name: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM faces WHERE LOWER(name) = LOWER(?) LIMIT 1", (name,)
        )
        return cur.fetchone() is not None

    # ── Recognition log ───────────────────────────────────────────────────────

    def log_recognition(self, name: str, confidence: float):
        self.conn.execute(
            "INSERT INTO recognition_log (name, confidence) VALUES (?, ?)",
            (name, confidence),
        )
        self.conn.commit()

    def get_recognition_log(self, limit: int = 100):
        cur = self.conn.execute(
            "SELECT name, confidence, timestamp FROM recognition_log "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def clear_recognition_log(self):
        self.conn.execute("DELETE FROM recognition_log")
        self.conn.commit()

    def close(self):
        self.conn.close()
