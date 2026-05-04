import sqlite3, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "youtube.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id      TEXT PRIMARY KEY,
            url           TEXT,
            title         TEXT,
            channel       TEXT,
            description   TEXT,
            published     TEXT,
            fetched_at    TEXT,
            view_count    INTEGER DEFAULT 0,
            like_count    INTEGER DEFAULT 0,
            dislike_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS comments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    TEXT,
            source      TEXT,
            author      TEXT,
            text        TEXT,
            sentiment   TEXT,
            score       REAL,
            likes       INTEGER DEFAULT 0,
            fetched_at  TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(video_id)
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id   TEXT,
            role       TEXT,
            content    TEXT,
            created_at TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(video_id)
        );
        CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_chat_history_video_id ON chat_history(video_id);
    """)
    # Migrate: add stats columns to existing databases
    for col, typ in [("view_count","INTEGER"),("like_count","INTEGER"),("dislike_count","INTEGER"),("comment_count","INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {typ} DEFAULT 0")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()

def upsert_video(video_id, url, title, channel, description, published,
                 view_count=0, like_count=0, dislike_count=0, comment_count=0):
    conn = get_conn()
    conn.execute("""
        INSERT INTO videos (video_id, url, title, channel, description, published, fetched_at,
                            view_count, like_count, dislike_count, comment_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            title=excluded.title, channel=excluded.channel,
            description=excluded.description, fetched_at=excluded.fetched_at,
            view_count=excluded.view_count, like_count=excluded.like_count,
            dislike_count=excluded.dislike_count, comment_count=excluded.comment_count
    """, (video_id, url, title, channel, description, published, datetime.utcnow().isoformat(),
          view_count, like_count, dislike_count, comment_count))
    conn.commit()
    conn.close()

def save_comments(video_id, comments):
    conn = get_conn()
    conn.execute("DELETE FROM comments WHERE video_id=?", (video_id,))
    now = datetime.utcnow().isoformat()
    conn.executemany("""
        INSERT INTO comments (video_id, source, author, text, sentiment, score, likes, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [(video_id, c["source"], c["author"], c["text"],
           c["sentiment"], c["score"], c["likes"], now) for c in comments])
    conn.commit()
    conn.close()

def get_video(video_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_comments(video_id, source=None):
    conn = get_conn()
    if source:
        rows = conn.execute("SELECT * FROM comments WHERE video_id=? AND source=? ORDER BY score DESC", (video_id, source)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM comments WHERE video_id=? ORDER BY score DESC", (video_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_videos():
    conn = get_conn()
    rows = conn.execute("SELECT video_id, title, channel, url, fetched_at FROM videos ORDER BY fetched_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_chat_message(video_id, role, content):
    conn = get_conn()
    conn.execute("INSERT INTO chat_history (video_id, role, content, created_at) VALUES (?,?,?,?)",
                 (video_id, role, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_chat_history(video_id):
    conn = get_conn()
    rows = conn.execute("SELECT role, content FROM chat_history WHERE video_id=? ORDER BY id", (video_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def clear_chat_history(video_id):
    conn = get_conn()
    conn.execute("DELETE FROM chat_history WHERE video_id=?", (video_id,))
    conn.commit()
    conn.close()

init_db()
