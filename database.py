import sqlite3
import os


class Database:
    def __init__(self, db_path="data/faces.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS faces (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                encoding   BLOB    NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS faces_dl (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                embedding  BLOB    NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS recognition_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL,
                confidence REAL,
                timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── LBPH face CRUD ────────────────────────────────────────────────────────

    def add_face(self, name: str, encoding_blob: bytes):
        self.conn.execute(
            "INSERT INTO faces (name, encoding) VALUES (?, ?)",
            (name, encoding_blob))
        self.conn.commit()

    def get_all_faces(self):
        """Returns [(name, encoding_blob), ...] for the LBPH recogniser."""
        return self.conn.execute(
            "SELECT name, encoding FROM faces").fetchall()

    # ── DL embedding CRUD ─────────────────────────────────────────────────────

    def add_dl_embedding(self, name: str, embedding_blob: bytes):
        self.conn.execute(
            "INSERT INTO faces_dl (name, embedding) VALUES (?, ?)",
            (name, embedding_blob))
        self.conn.commit()

    def get_all_dl_embeddings(self):
        """Returns [(name, embedding_blob), ...] for the DL recogniser."""
        return self.conn.execute(
            "SELECT name, embedding FROM faces_dl").fetchall()

    # ── Shared helpers ────────────────────────────────────────────────────────

    def get_face_names(self):
        """
        Returns one row per person: (name, sample_count, first_seen).
        Prefers the DL table; falls back to the LBPH table if DL is empty.
        """
        for tbl in ("faces_dl", "faces"):
            rows = self.conn.execute(
                f"SELECT name, COUNT(*) AS cnt, MIN(created_at) AS first_seen "
                f"FROM {tbl} GROUP BY name ORDER BY name"
            ).fetchall()
            if rows:
                return rows
        return []

    def delete_person(self, name: str):
        """Remove all LBPH and DL records for a person."""
        self.conn.execute("DELETE FROM faces    WHERE name = ?", (name,))
        self.conn.execute("DELETE FROM faces_dl WHERE name = ?", (name,))
        self.conn.commit()

    def delete_face(self, face_id: int):
        self.conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()

    def face_exists(self, name: str) -> bool:
        for tbl in ("faces", "faces_dl"):
            cur = self.conn.execute(
                f"SELECT 1 FROM {tbl} WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (name,))
            if cur.fetchone():
                return True
        return False

    # ── Recognition log ───────────────────────────────────────────────────────

    def log_recognition(self, name: str, confidence: float):
        self.conn.execute(
            "INSERT INTO recognition_log (name, confidence) VALUES (?, ?)",
            (name, confidence))
        self.conn.commit()

    def get_recognition_log(self, limit: int = 100):
        return self.conn.execute(
            "SELECT name, confidence, timestamp FROM recognition_log "
            "ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()

    def clear_recognition_log(self):
        self.conn.execute("DELETE FROM recognition_log")
        self.conn.commit()

    def close(self):
        self.conn.close()
