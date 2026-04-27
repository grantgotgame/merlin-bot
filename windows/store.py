"""SQLite-backed transcript and event log.

A daemon thread drains a queue of (kind, payload) tuples into the DB so
the audio loop is never blocked by disk I/O. The schema is intentionally
small: messages, events, sessions. FTS5 over messages.text enables
search across past sessions.
"""

import json
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Iterable

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merlin.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    intent TEXT,
    latency_ms INTEGER,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
"""

# Events that aren't worth persisting — high-frequency telemetry.
_SKIP_EVENT_TYPES = {"audio_rms", "tts_chunk", "face_box"}


class Store:
    def __init__(self, bus=None):
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.session_id: int | None = None
        self._init_db()
        self._open_session()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        if bus is not None:
            self.attach(bus)

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because the writer thread reuses one conn
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _open_session(self):
        conn = self._connect()
        try:
            cur = conn.execute("INSERT INTO sessions(started_at) VALUES (?)", (time.time(),))
            self.session_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Bus integration
    # ------------------------------------------------------------------

    def attach(self, bus):
        bus.on("*", self._on_event)

    def _on_event(self, event: str, **payload):
        # Persist user/merlin messages as structured rows; everything else
        # goes into the events table for replay/debugging.
        if event == "stt_complete":
            self.log_message(
                role="user",
                text=payload.get("text", ""),
                latency_ms=payload.get("latency_ms"),
            )
        elif event == "thinking_complete":
            self.log_message(
                role="assistant",
                text=payload.get("text", ""),
                intent=payload.get("intent"),
                latency_ms=payload.get("latency_ms"),
            )
        elif event in _SKIP_EVENT_TYPES:
            return
        else:
            self.log_event(event, payload)

    # ------------------------------------------------------------------
    # Public writers (queue-backed)
    # ------------------------------------------------------------------

    def log_message(self, role: str, text: str, intent: str | None = None, latency_ms: int | None = None):
        if not text:
            return
        self._queue.put(("message", {
            "session_id": self.session_id,
            "ts": time.time(),
            "role": role,
            "text": text,
            "intent": intent,
            "latency_ms": latency_ms,
        }))

    def log_event(self, etype: str, payload: dict[str, Any] | None = None):
        self._queue.put(("event", {
            "session_id": self.session_id,
            "ts": time.time(),
            "type": etype,
            "payload": json.dumps(payload or {}, default=str),
        }))

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self):
        conn = self._connect()
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self._write(conn, item)
                except Exception as e:
                    print(f"[store] write error: {e}")
            # Drain remaining items on shutdown
            while not self._queue.empty():
                try:
                    self._write(conn, self._queue.get_nowait())
                except Exception:
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _write(self, conn, item):
        kind, row = item
        if kind == "message":
            conn.execute(
                "INSERT INTO messages(session_id, ts, role, text, intent, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (row["session_id"], row["ts"], row["role"], row["text"], row["intent"], row["latency_ms"]),
            )
        elif kind == "event":
            conn.execute(
                "INSERT INTO events(session_id, ts, type, payload) VALUES (?, ?, ?, ?)",
                (row["session_id"], row["ts"], row["type"], row["payload"]),
            )
        conn.commit()

    # ------------------------------------------------------------------
    # Read APIs (called from the FastAPI thread)
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT s.id, s.started_at, s.ended_at, COUNT(m.id) AS msg_count
                FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def history(self, session_id: int | None = None, limit: int = 200) -> list[dict]:
        conn = self._connect()
        try:
            if session_id is None:
                session_id = self.session_id
            cur = conn.execute(
                "SELECT ts, role, text, intent, latency_ms FROM messages WHERE session_id = ? ORDER BY ts ASC LIMIT ?",
                (session_id, limit),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def search(self, q: str, limit: int = 50) -> list[dict]:
        if not q.strip():
            return []
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT m.id, m.session_id, m.ts, m.role, m.text
                FROM messages_fts f JOIN messages m ON m.id = f.rowid
                WHERE messages_fts MATCH ?
                ORDER BY m.ts DESC LIMIT ?
                """,
                (q, limit),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def recent_events(self, since: float | None = None, limit: int = 200) -> list[dict]:
        conn = self._connect()
        try:
            if since is not None:
                cur = conn.execute(
                    "SELECT ts, type, payload FROM events WHERE ts > ? ORDER BY ts ASC LIMIT ?",
                    (since, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT ts, type, payload FROM events ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            for r in rows:
                try:
                    r["payload"] = json.loads(r["payload"]) if r["payload"] else {}
                except Exception:
                    r["payload"] = {}
            return rows
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        if self.session_id is not None:
            try:
                conn = self._connect()
                conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), self.session_id))
                conn.commit()
                conn.close()
            except Exception:
                pass
