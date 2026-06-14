import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .config import SQLITE_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    subject TEXT,
    doc_type TEXT,
    year INTEGER,
    tags TEXT,
    uploaded_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER,
    chunk_id TEXT,
    content TEXT,
    page INTEGER,
    FOREIGN KEY(doc_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    doc_id INTEGER,
    score REAL,
    FOREIGN KEY(doc_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT,
    topic TEXT,
    total_questions INTEGER DEFAULT 0,
    solved INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    try:
        conn.executescript(SCHEMA_SQL)
    finally:
        conn.close()


def upsert_document(meta: Dict) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents(path,name,ext,subject,doc_type,year,tags) VALUES(?,?,?,?,?,?,?)",
            (
                meta.get("path"),
                meta.get("name"),
                meta.get("ext"),
                meta.get("subject"),
                meta.get("doc_type"),
                meta.get("year"),
                meta.get("tags"),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def insert_chunks(doc_id: int, chunks: List[Tuple[str, str, Optional[int]]]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT INTO chunks(doc_id,chunk_id,content,page) VALUES(?,?,?,?)",
            [(doc_id, cid, content, page) for cid, content, page in chunks],
        )
        conn.commit()
    finally:
        conn.close()


def record_topics(doc_id: int, topics: List[Tuple[str, float]]):
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT INTO topics(name,doc_id,score) VALUES(?,?,?)",
            [(name, doc_id, score) for name, score in topics],
        )
        conn.commit()
    finally:
        conn.close()


def update_progress(subject: str, topic: str, delta_total: int = 0, delta_solved: int = 0):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, total_questions, solved FROM progress WHERE subject=? AND topic=?",
            (subject, topic),
        )
        row = cur.fetchone()
        if row:
            pid, total, solved = row
            cur.execute(
                "UPDATE progress SET total_questions=?, solved=?, last_updated=datetime('now') WHERE id=?",
                (total + delta_total, solved + delta_solved, pid),
            )
        else:
            cur.execute(
                "INSERT INTO progress(subject, topic, total_questions, solved) VALUES(?,?,?,?)",
                (subject, topic, max(delta_total, 0), max(delta_solved, 0)),
            )
        conn.commit()
    finally:
        conn.close()


def list_documents() -> List[Dict]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, ext, subject, doc_type, year, tags, path, uploaded_at FROM documents ORDER BY uploaded_at DESC"
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "ext": r[2],
                "subject": r[3],
                "doc_type": r[4],
                "year": r[5],
                "tags": r[6],
                "path": r[7],
                "uploaded_at": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()


def list_topics(subject: Optional[str] = None) -> List[Dict]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        if subject:
            cur.execute(
                "SELECT t.name, t.score, d.subject FROM topics t JOIN documents d ON t.doc_id = d.id WHERE d.subject=? ORDER BY t.score DESC",
                (subject,),
            )
        else:
            cur.execute(
                "SELECT t.name, t.score, d.subject FROM topics t JOIN documents d ON t.doc_id = d.id ORDER BY t.score DESC"
            )
        rows = cur.fetchall()
        return [{"name": r[0], "score": r[1], "subject": r[2]} for r in rows]
    finally:
        conn.close()
