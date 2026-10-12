import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).with_name("reputation_cache.db")

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT UNIQUE NOT NULL,
            company TEXT NOT NULL,
            industry TEXT,
            region TEXT NOT NULL,
            days INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            article_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            normalized_url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published_at TEXT,
            snippet TEXT,
            article_text TEXT,
            extraction_method TEXT,
            relevance_score INTEGER,
            sentiment_label TEXT,
            positive REAL,
            neutral REAL,
            negative REAL,
            signed_score REAL,
            UNIQUE(run_id, normalized_url),
            FOREIGN KEY(run_id) REFERENCES search_runs(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_runs_key ON search_runs(cache_key);
        CREATE INDEX IF NOT EXISTS idx_articles_run ON articles(run_id);
        """)

def get_cached_run(cache_key, max_age_hours=24):
    with connect() as conn:
        row = conn.execute("""
            SELECT * FROM search_runs
            WHERE cache_key = ?
              AND datetime(created_at) >= datetime('now', ?)
        """, (cache_key, f"-{int(max_age_hours)} hours")).fetchone()
        return dict(row) if row else None

def get_articles(run_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE run_id = ? ORDER BY published_at DESC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

def replace_run(cache_key, company, industry, region, days, created_at, articles):
    with connect() as conn:
        old = conn.execute("SELECT id FROM search_runs WHERE cache_key = ?", (cache_key,)).fetchone()
        if old:
            conn.execute("DELETE FROM articles WHERE run_id = ?", (old["id"],))
            conn.execute("DELETE FROM search_runs WHERE id = ?", (old["id"],))
        cur = conn.execute("""
            INSERT INTO search_runs(cache_key, company, industry, region, days, created_at, article_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cache_key, company, industry, region, days, created_at, len(articles)))
        run_id = cur.lastrowid
        conn.executemany("""
            INSERT OR IGNORE INTO articles(
                run_id, url, normalized_url, title, source, published_at, snippet,
                article_text, extraction_method, relevance_score, sentiment_label,
                positive, neutral, negative, signed_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            run_id, a["url"], a["normalized_url"], a["title"], a.get("source"),
            a.get("published_at"), a.get("snippet"), a.get("article_text"),
            a.get("extraction_method"), a.get("relevance_score"),
            a.get("sentiment_label"), a.get("positive"), a.get("neutral"),
            a.get("negative"), a.get("signed_score")
        ) for a in articles])
        return run_id
