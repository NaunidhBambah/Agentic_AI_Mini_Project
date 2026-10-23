"""SQLite store for doubts, their status, and guardrail events (used by all three views)."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS doubts (
    id TEXT PRIMARY KEY,
    student TEXT,
    question TEXT,          -- already PII-masked
    category TEXT,          -- concept | assignment | admin | off_topic | blocked
    topic TEXT,
    status TEXT,            -- answered | verified | hint | escalated | ta_answered | refused | blocked
    answer TEXT,
    draft TEXT,             -- agent draft shown to the TA when escalated
    sources TEXT,           -- JSON list
    confidence REAL,
    guardrails TEXT,        -- JSON list of guardrail events
    trace TEXT,             -- JSON list of node steps
    answered_by TEXT,       -- agent | ta:<name> | memory:<ta name>
    created_at TEXT,
    resolved_at TEXT,
    is_seed INTEGER DEFAULT 0,
    standalone TEXT,        -- question rewritten with conversation context (used for retrieval/memory)
    chat_id TEXT,           -- conversation thread this doubt belongs to
    scope TEXT,             -- course file the student focused on ('' = all materials)
    subject TEXT,           -- which subject's library the doubt was asked in
    web_answer TEXT,        -- optional extra explanation from a web search the student approved
    web_sources TEXT        -- JSON list of {title, url} for that web answer
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with _conn() as con:
        con.executescript(SCHEMA)
        for col in ("standalone", "chat_id", "scope", "subject", "web_answer", "web_sources"):   # upgrade older databases in place
            try:
                con.execute(f"ALTER TABLE doubts ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass


def now() -> str:
    """Timestamp with milliseconds so doubts asked in the same second keep their order."""
    t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def new_id() -> str:
    return "D" + uuid.uuid4().hex[:8].upper()


def save(record: dict):
    rec = dict(record)
    if "web_answer" not in rec:                 # graph re-saves must not wipe an approved web answer
        old = get(rec["id"]) if rec.get("id") else None
        if old:
            rec["web_answer"], rec["web_sources"] = old.get("web_answer"), old.get("web_sources")
    for key in ("sources", "guardrails", "trace", "web_sources"):
        if not isinstance(rec.get(key), str):
            rec[key] = json.dumps(rec.get(key) or [])
    cols = ["id", "student", "question", "category", "topic", "status", "answer", "draft",
            "sources", "confidence", "guardrails", "trace", "answered_by", "created_at",
            "resolved_at", "is_seed", "standalone", "chat_id", "scope", "subject", "web_answer", "web_sources"]
    rec.setdefault("is_seed", 0)
    with _conn() as con:
        con.execute(f"INSERT OR REPLACE INTO doubts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [rec.get(c) for c in cols])


def update(doubt_id: str, **fields):
    for key in ("sources", "guardrails", "trace", "web_sources"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    sets = ", ".join(f"{k}=?" for k in fields)
    with _conn() as con:
        con.execute(f"UPDATE doubts SET {sets} WHERE id=?", [*fields.values(), doubt_id])


def _row(r) -> dict:
    d = dict(r)
    for key in ("sources", "guardrails", "trace", "web_sources"):
        try:
            d[key] = json.loads(d.get(key) or "[]")
        except json.JSONDecodeError:
            d[key] = []
    return d


def get(doubt_id: str):
    with _conn() as con:
        r = con.execute("SELECT * FROM doubts WHERE id=?", (doubt_id,)).fetchone()
    return _row(r) if r else None


_SUBJ = "COALESCE(NULLIF(subject, ''), ?)"


def list_doubts(status=None, student=None, limit=500, chat_id=None, subject=None):
    q, args = "SELECT * FROM doubts WHERE 1=1", []
    if subject:
        q += f" AND {_SUBJ} = ?"
        args += [config.COURSE_NAME, subject]
    if chat_id:
        q += " AND COALESCE(chat_id, 'earlier') = ?"
        args.append(chat_id)
    if status:
        q += f" AND status IN ({','.join('?' * len(status))})"
        args += list(status)
    if student:
        q += " AND student=?"
        args.append(student)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _conn() as con:
        return [_row(r) for r in con.execute(q, args).fetchall()]


def new_chat_id() -> str:
    return "C" + uuid.uuid4().hex[:8].upper()


def list_chats(student: str, subject: str = None) -> list:
    """A student's conversations (in one subject, if given), newest first."""
    sql = f"""SELECT COALESCE(chat_id, 'earlier') AS cid, COUNT(*) AS n,
                    MIN(created_at) AS started, MAX(created_at) AS last
             FROM doubts WHERE student = ? {f"AND {_SUBJ} = ?" if subject else ""}
             GROUP BY cid ORDER BY last DESC"""
    args = (student, config.COURSE_NAME, subject) if subject else (student,)
    with _conn() as con:
        rows = [dict(r) for r in con.execute(sql, args).fetchall()]
        for r in rows:
            first = con.execute(
                "SELECT question FROM doubts WHERE student=? AND COALESCE(chat_id,'earlier')=? "
                "ORDER BY created_at LIMIT 1", (student, r["cid"])).fetchone()
            r["title"] = (first[0] if first else "New chat")[:48]
    return rows


def delete_chat(student: str, chat_id: str):
    with _conn() as con:
        con.execute("DELETE FROM doubts WHERE student=? AND COALESCE(chat_id,'earlier')=?", (student, chat_id))


def count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM doubts").fetchone()[0]


def clear_all():
    with _conn() as con:
        con.execute("DELETE FROM doubts")