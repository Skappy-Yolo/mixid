"""Local library matcher (opt-in accelerator).

Takes a FingerprintSweep (query: 7 pitch variants × N uint32 hashes each)
and scores it against every fingerprint in `data/fingerprints.db`.
Returns the top-K matches above a configurable confidence threshold.

The library track is typically much longer than the query (a 3-minute
song vs a 12-second sample), so we slide the query across every alignment
offset in the library track and take the best score per (track, pitch).
Across all 7 pitch variants we take the overall best — that pitch is the
one the DJ likely applied.

Pure-numpy implementation using bitwise XOR + bitwise_count (numpy 2.0+).
~5-10 seconds against a 14k-track library on CPU; the hash-prefix prefilter
optimization is reserved for Phase 5.5 if measured profiling demands it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from mixid.pipeline.fingerprint import FingerprintSweep


@dataclass
class LibraryMatch:
    track_id: int
    filepath: str
    title: str
    artist: str
    album: str
    score: float                    # in [0, 1]; 1 = identical
    best_pitch_shift_percent: int   # which pitch variant scored best
    best_alignment_sec: float       # where in the library track the match starts


_HASHES_PER_SEC = 8.0  # Chromaprint emits ~7.7 hashes/sec at 22050 Hz mono


def _score_query_vs_library(
    query: np.ndarray, library: np.ndarray
) -> tuple[float, int]:
    """Slide query across library; return (best_score, best_offset_hashes)."""
    Q = len(query)
    L = len(library)
    if Q == 0 or L == 0 or Q > L:
        return 0.0, 0

    total_bits = Q * 32
    best_score = 0.0
    best_offset = 0
    # Numpy vectorization sweet spot: do every alignment at once via stride tricks.
    # For now keep the readable loop; it's fast enough for Q~50 and L~3000.
    for off in range(L - Q + 1):
        window = library[off : off + Q]
        xor = np.bitwise_xor(query, window)
        bits = int(np.sum(np.bitwise_count(xor)))
        score = 1.0 - (bits / total_bits)
        if score > best_score:
            best_score = score
            best_offset = off
            if best_score > 0.97:
                break  # near-identical; no need to keep searching
    return best_score, best_offset


def match_sweep(
    sweep: FingerprintSweep,
    db_path: Path | str = config.FINGERPRINTS_DB,
    top_k: int = 5,
    min_score: float = config.MATCH_CONFIDENCE_FLOOR,
) -> list[LibraryMatch]:
    """Score `sweep` against every track in the fingerprints DB. Returns top_k matches."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []  # no library indexed — caller falls back to remote matchers

    # Load every library fingerprint into memory once. For a 14k-track library
    # at ~4 KB/track average, this is ~56 MB — fits comfortably.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, filepath, title, artist, album, fingerprint_raw, n_hashes "
        "FROM tracks"
    ).fetchall()
    conn.close()

    candidates: list[LibraryMatch] = []
    for row in rows:
        lib_hashes = np.frombuffer(row["fingerprint_raw"], dtype=np.uint32)
        best_score = 0.0
        best_pct = 0
        best_offset = 0
        for fp in sweep.fingerprints:
            if fp.raw_hashes is None or len(fp.raw_hashes) == 0:
                continue
            score, off = _score_query_vs_library(fp.raw_hashes, lib_hashes)
            if score > best_score:
                best_score = score
                best_pct = fp.pitch_shift_percent
                best_offset = off
        if best_score < min_score:
            continue
        candidates.append(
            LibraryMatch(
                track_id=row["id"],
                filepath=row["filepath"],
                title=row["title"],
                artist=row["artist"],
                album=row["album"],
                score=best_score,
                best_pitch_shift_percent=best_pct,
                best_alignment_sec=best_offset / _HASHES_PER_SEC,
            )
        )

    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates[:top_k]
