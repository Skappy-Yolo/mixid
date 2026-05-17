"""Build the fingerprint library from Deezer's free public API.

Why Deezer: their `/search` and `/chart` endpoints return a `preview` URL
for every track (30-second MP3, freely downloadable), no auth needed, no
quota burn. Genre-agnostic — works the same for any music. This replaces
the previous Spotify-preview approach which Spotify deprecated.

The user picks which slice of Deezer to fingerprint:
  --from-chart                  global top tracks across all genres
  --from-chart --genre-id 116   tracks from a Deezer genre chart
                                (run --list-genres to see IDs)
  --from-artists "X,Y,Z"        top tracks per named artist
  --from-search "query"         arbitrary search results

Outputs land in the same `data/fingerprints.db` consumed by Tier-1's
local matcher. Subsequent runs are incremental (Deezer track id is the
unique key); re-running adds new tracks, never duplicates.
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from pathlib import Path

import requests
import soundfile as sf

import config
from mixid.pipeline import fingerprint as fp_mod

_BASE = "https://api.deezer.com"
_TIMEOUT_SECS = 15


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Same schema as the local-files indexer; one row per track."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id              INTEGER PRIMARY KEY,
            filepath        TEXT UNIQUE NOT NULL,
            duration_secs   REAL,
            fingerprint_raw BLOB NOT NULL,
            n_hashes        INTEGER NOT NULL,
            title           TEXT NOT NULL DEFAULT '',
            artist          TEXT NOT NULL DEFAULT '',
            album           TEXT NOT NULL DEFAULT '',
            indexed_at      REAL
        );
        CREATE INDEX IF NOT EXISTS idx_tracks_filepath ON tracks(filepath);
        """
    )
    conn.commit()


def _deezer_get(path: str, **params) -> dict:
    r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT_SECS)
    r.raise_for_status()
    return r.json()


def _iter_chart_tracks(limit: int = 100, genre_id: int = 0):
    """Yield tracks from the global or genre-specific chart."""
    path = f"/chart/{genre_id}/tracks" if genre_id else "/chart/0/tracks"
    data = _deezer_get(path, limit=str(limit))
    for t in data.get("data", []):
        yield t


def _iter_search_tracks(query: str, limit: int = 25):
    """Yield tracks matching a Deezer search query."""
    # Deezer caps page size at 25; paginate up to `limit`
    fetched = 0
    index = 0
    while fetched < limit:
        page_size = min(25, limit - fetched)
        data = _deezer_get("/search", q=query, limit=str(page_size), index=str(index))
        items = data.get("data") or []
        if not items:
            return
        for t in items:
            yield t
            fetched += 1
            if fetched >= limit:
                return
        index += page_size


def _iter_artist_top(artist: str, n_top: int = 20):
    """Resolve an artist name → top tracks (via Deezer's per-artist /top endpoint)."""
    sr = _deezer_get("/search/artist", q=artist, limit="1")
    artists = sr.get("data") or []
    if not artists:
        print(f"  artist not found on Deezer: {artist!r}", file=sys.stderr)
        return
    artist_id = artists[0]["id"]
    data = _deezer_get(f"/artist/{artist_id}/top", limit=str(n_top))
    for t in data.get("data", []):
        yield t


def _track_record(t: dict) -> tuple[str, str, str, str, str] | None:
    """Extract (deezer_uri, title, artist, album, preview_url) or None if no preview."""
    preview = t.get("preview")
    if not preview:
        return None
    deezer_id = t.get("id")
    if not deezer_id:
        return None
    artist = (t.get("artist") or {}).get("name", "") or ""
    album = (t.get("album") or {}).get("title", "") or ""
    title = t.get("title") or ""
    uri = f"deezer:{deezer_id}"
    return uri, title, artist, album, preview


def _fingerprint_preview_mp3(url: str) -> fp_mod.Fingerprint | None:
    """Download an MP3 preview, write to a temp file, run fpcalc -raw."""
    r = requests.get(url, timeout=_TIMEOUT_SECS)
    r.raise_for_status()
    audio_bytes = r.content
    # fpcalc reads from a file path
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = Path(f.name)
    try:
        return fp_mod.fingerprint_file(path, raw=True)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def index_tracks(track_iter, *, max_count: int | None = None) -> tuple[int, int]:
    """Common indexing loop. Returns (indexed_now, total_in_db)."""
    conn = sqlite3.connect(str(config.FINGERPRINTS_DB))
    _ensure_schema(conn)
    existing = {row[0] for row in conn.execute("SELECT filepath FROM tracks")}
    indexed = 0
    for t in track_iter:
        rec = _track_record(t)
        if rec is None:
            continue
        uri, title, artist, album, preview = rec
        if uri in existing:
            continue
        try:
            fp = _fingerprint_preview_mp3(preview)
        except requests.HTTPError as e:
            print(f"  skip {artist} - {title}: HTTP {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  skip {artist} - {title}: {e}", file=sys.stderr)
            continue
        if fp.raw_hashes is None or len(fp.raw_hashes) == 0:
            print(f"  skip {artist} - {title}: empty fingerprint", file=sys.stderr)
            continue
        conn.execute(
            "INSERT OR REPLACE INTO tracks "
            "(filepath, duration_secs, fingerprint_raw, n_hashes, title, artist, album, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uri, fp.duration_secs, fp.raw_hashes.tobytes(), int(len(fp.raw_hashes)),
             title, artist, album, time.time()),
        )
        indexed += 1
        if indexed % 10 == 0:
            conn.commit()
            print(f"  …{indexed} tracks indexed (latest: {artist} - {title})")
        if max_count and indexed >= max_count:
            break
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    conn.close()
    return indexed, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mixid-deezer-build", description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--from-chart", action="store_true", help="Global or genre chart")
    grp.add_argument("--from-artists", type=str, help="Comma-separated artist names")
    grp.add_argument("--from-search", type=str, help="Arbitrary Deezer search query")
    p.add_argument("--limit", type=int, default=100, help="Total tracks to fetch")
    p.add_argument("--top-n", type=int, default=20, help="Tracks per artist (with --from-artists)")
    p.add_argument("--genre-id", type=int, default=0, help="Deezer genre ID for chart mode")
    args = p.parse_args(argv)

    t0 = time.time()
    if args.from_chart:
        it = _iter_chart_tracks(limit=args.limit, genre_id=args.genre_id)
    elif args.from_artists:
        def _multi_artist():
            for name in [n.strip() for n in args.from_artists.split(",") if n.strip()]:
                print(f"--- {name} ---")
                yield from _iter_artist_top(name, n_top=args.top_n)
        it = _multi_artist()
    else:
        it = _iter_search_tracks(args.from_search, limit=args.limit)

    indexed, total = index_tracks(it, max_count=args.limit if args.from_chart or args.from_search else None)
    dt = time.time() - t0
    print(f"\nIndexed {indexed} new tracks in {dt:.1f}s; {total} total in {config.FINGERPRINTS_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
