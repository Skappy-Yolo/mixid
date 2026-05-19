"""Aggregate-only usage counter for the public web app.

We do not collect logins, IPs, emails, or any PII. The point of this
module is to answer one question — "how many mixes has MixID processed
since launch" — and surface that number on the homepage.

Storage: a tiny SQLite file at ~/.mixid/stats.db (outside the repo's
gitignored data/ so the counter survives across deploys / reinstalls).
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# ~/.mixid/stats.db on all platforms
STATS_DIR = Path(os.path.expanduser("~")) / ".mixid"
STATS_DB = STATS_DIR / "stats.db"


def _connect() -> sqlite3.Connection:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STATS_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY,
            ts              REAL NOT NULL,
            duration_secs   REAL,
            identified      INTEGER,
            unidentified    INTEGER,
            source          TEXT NOT NULL DEFAULT 'file'  -- 'file' | 'url' | 'cli'
        );
        """
    )
    return conn


@dataclass
class StatsSummary:
    total_runs: int
    total_duration_processed_secs: float
    total_tracks_identified: int


def record_run(
    duration_secs: float | None,
    identified: int,
    unidentified: int,
    source: str = "file",
) -> None:
    """Insert one row. Idempotent in the sense that it always inserts a new run."""
    if source not in ("file", "url", "cli"):
        source = "file"
    conn = _connect()
    conn.execute(
        "INSERT INTO runs (ts, duration_secs, identified, unidentified, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), duration_secs, int(identified), int(unidentified), source),
    )
    conn.commit()
    conn.close()


def summary() -> StatsSummary:
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(duration_secs), 0), COALESCE(SUM(identified), 0) FROM runs"
    ).fetchone()
    conn.close()
    return StatsSummary(
        total_runs=int(row[0]),
        total_duration_processed_secs=float(row[1]),
        total_tracks_identified=int(row[2]),
    )
