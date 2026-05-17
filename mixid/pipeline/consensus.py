"""Collapse adjacent re-ranked segments that point to the same track.

After re-ranking, multiple consecutive segments often resolve to the
same artist/title — typically because our segmentation was finer-grained
than the actual track boundaries. This module merges those runs into
one TracklistTrack so the output reads naturally (one entry per played
track, not per fingerprint window).
"""
from __future__ import annotations

from dataclasses import dataclass

from mixid.pipeline.reranker import RerankedResult


@dataclass
class TracklistTrack:
    start_sec: float
    end_sec: float
    artist: str
    title: str
    score: float                 # max score across merged segments
    source: str                  # source of the BEST-scoring merged segment
    n_segments_merged: int


def _same_track(a: RerankedResult, b: RerankedResult) -> bool:
    return (
        a.artist.strip().lower() == b.artist.strip().lower()
        and a.title.strip().lower() == b.title.strip().lower()
    )


def collapse(results: list[RerankedResult]) -> list[TracklistTrack]:
    """Merge consecutive segments with the same artist/title.

    Non-consecutive duplicates are NOT merged — they could be the DJ
    re-dropping the same track later in the set, which is meaningful.
    """
    if not results:
        return []
    # Stable-sort by start time first so consecutive-in-time runs are adjacent.
    ordered = sorted(results, key=lambda r: r.start_sec)
    out: list[TracklistTrack] = []
    current_run: list[RerankedResult] = [ordered[0]]
    for r in ordered[1:]:
        if _same_track(current_run[-1], r):
            current_run.append(r)
            continue
        out.append(_run_to_track(current_run))
        current_run = [r]
    out.append(_run_to_track(current_run))
    return out


def _run_to_track(run: list[RerankedResult]) -> TracklistTrack:
    best = max(run, key=lambda r: r.score)
    return TracklistTrack(
        start_sec=run[0].start_sec,
        end_sec=run[-1].end_sec,
        artist=best.artist,
        title=best.title,
        score=best.score,
        source=best.source,
        n_segments_merged=len(run),
    )
