"""Optional library accelerator — fingerprint a user's local music catalog.

MixID's PRIMARY matcher is AcoustID (the public ~50M-track database),
which works for anyone with no setup. This indexer is an *optional*
speedup for DJs who happen to have their own catalog — once built, the
matcher checks the local index first and only falls back to AcoustID
for unknown segments. Saves API calls and is instant. Skip this step
entirely if you don't have a local library; the rest of the pipeline
still works.

Reads track filepaths from the sibling DJAgent's `library.db` if present;
otherwise walks `config.MUSIC_DIR` recursively. Skips non-music files
(notably .vdjsample, which is VirtualDJ's per-deck sampler audio).

Output: `data/fingerprints.db` with one row per indexed track. Run once
(~1 sec/track on CPU); incremental updates pick up new files added later.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import config
from mixid.pipeline import fingerprint as fp_mod

MUSIC_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus", ".wma"}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id              INTEGER PRIMARY KEY,
            filepath        TEXT UNIQUE NOT NULL,
            duration_secs   REAL,
            fingerprint_b64 TEXT NOT NULL,
            title           TEXT NOT NULL DEFAULT '',
            artist          TEXT NOT NULL DEFAULT '',
            album           TEXT NOT NULL DEFAULT '',
            indexed_at      REAL
        );
        CREATE INDEX IF NOT EXISTS idx_tracks_filepath ON tracks(filepath);
        """
    )
    conn.commit()


def _track_iter_from_djagent(db_path: Path):
    """Yield (filepath, title, artist, album) tuples filtered to music files."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT filepath, title, artist, album FROM tracks"):
        fp = (row["filepath"] or "").strip()
        if not fp:
            continue
        if Path(fp).suffix.lower() not in MUSIC_EXTS:
            continue
        yield fp, row["title"] or "", row["artist"] or "", row["album"] or ""
    conn.close()


def _track_iter_from_filesystem(root: Path):
    """Yield (filepath, '', '', '') tuples by walking a music root directory."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in MUSIC_EXTS:
            yield str(p), "", "", ""


def build(
    limit: int | None = None,
    rebuild: bool = False,
    source: Path | None = None,
) -> tuple[int, int]:
    """Fingerprint every track in the source. Returns (indexed_now, total_in_db)."""
    conn = sqlite3.connect(str(config.FINGERPRINTS_DB))
    _ensure_schema(conn)
    if rebuild:
        conn.execute("DELETE FROM tracks")
        conn.commit()

    djagent_db = config.DJAGENT_LIBRARY_DB
    if source is None:
        source = djagent_db if djagent_db.exists() else config.MUSIC_DIR
    src_iter = (
        _track_iter_from_djagent(source)
        if source.suffix == ".db"
        else _track_iter_from_filesystem(source)
    )

    indexed = 0
    seen = {row[0] for row in conn.execute("SELECT filepath FROM tracks")}
    for filepath, title, artist, album in src_iter:
        if limit is not None and indexed >= limit:
            break
        if filepath in seen:
            continue
        if not Path(filepath).exists():
            continue
        try:
            fp = fp_mod.fingerprint_file(filepath)
        except Exception as e:
            print(f"  skip {filepath}: {e}", file=sys.stderr)
            continue
        conn.execute(
            "INSERT OR REPLACE INTO tracks "
            "(filepath, duration_secs, fingerprint_b64, title, artist, album, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (filepath, fp.duration_secs, fp.fingerprint, title, artist, album, time.time()),
        )
        indexed += 1
        if indexed % 50 == 0:
            conn.commit()
            print(f"  …{indexed} tracks indexed")
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    conn.close()
    return indexed, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mixid-build-index", description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="Stop after N new tracks")
    p.add_argument("--rebuild", action="store_true", help="Wipe and rebuild")
    p.add_argument("--source", type=Path, default=None, help="Override source (.db or dir)")
    args = p.parse_args(argv)
    t0 = time.time()
    indexed, total = build(limit=args.limit, rebuild=args.rebuild, source=args.source)
    dt = time.time() - t0
    print(f"indexed {indexed} new tracks in {dt:.1f}s; {total} total in {config.FINGERPRINTS_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
