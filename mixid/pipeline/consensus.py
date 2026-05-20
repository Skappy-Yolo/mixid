"""Collapse adjacent re-ranked segments that point to the same track.

After re-ranking, multiple consecutive segments often resolve to the
same artist/title — typically because our segmentation was finer-grained
than the actual track boundaries. This module merges those runs into
one TracklistTrack so the output reads naturally (one entry per played
track, not per fingerprint window).

Also contains `merge_scrape_with_audio()` — the scrape-first reconciler
that overlays community tracklists onto audio-derived results so partial
scrapes don't leave gaps and confident scrapes override mis-identifications.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mixid.pipeline.reranker import RerankedResult
from mixid.pipeline.url_shortcut import SOURCE_PRIORITY, TracklistEntry


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


# ── scrape ↔ audio merge ──────────────────────────────────────────────


_SCRAPE_OVERLAP_WINDOW_SEC = 30.0
# If two scraped entries are within this much of each other, treat the
# scrape as fully covering whatever the audio said in between (drop
# audio unknowns inside the bracket). Set to roughly one DJ track:
# above this, we assume the scrape missed a track and let audio's
# unknown stand. 300s (5 min) was too generous per CTO review.
_GAP_TOO_BIG_SEC = 180.0  # 3 min


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def merge_scrape_with_audio(
    scraped: list[TracklistEntry],
    audio_tracks: list[TracklistTrack],
    unknown_segments: list[tuple[float, float]],
    mix_duration_sec: float,
) -> tuple[list[TracklistTrack], list[tuple[float, float]]]:
    """Overlay community-scraped entries onto audio-derived tracks.

    Rules (community-curated > audio fingerprint):
      1. A timed scraped entry within ±30s of an audio track's start
         REPLACES that audio track's (artist, title, source).
      2. Audio tracks with no scraped entry in range stay as-is.
      3. Timed scraped entries with no overlapping audio track are
         inserted as new TracklistTrack rows.
      4. Unknown segments enclosed by consecutive scraped entries less
         than 5 min apart are dropped (the DJ told us what's there).
      5. Untimed scraped entries that match a final track's normalized
         (artist, title) are dropped silently (the audio match
         localized them). Untimed entries that don't match get
         appended after the last identified track with +60s spacing.
    """
    timed_scraped = [e for e in scraped if e.start_sec is not None]
    untimed_scraped = [e for e in scraped if e.start_sec is None]
    timed_scraped.sort(key=lambda e: e.start_sec)

    final: list[TracklistTrack] = list(audio_tracks)

    # Rule 1 + 3: walk scraped entries, override or insert.
    # `claimed_audio` tracks indices that have already been overwritten
    # by an earlier scraped entry — a second scraped entry inside the
    # same ±30s window must NOT clobber the first override; instead it
    # gets inserted as a new track.
    claimed_audio: set[int] = set()
    audio_priority = 0  # audio sources (shazam, acoustid, etc.) aren't in SOURCE_PRIORITY
    for s in timed_scraped:
        match_idx = -1
        best_dist = _SCRAPE_OVERLAP_WINDOW_SEC + 1
        for i, t in enumerate(final):
            if i in claimed_audio:
                continue
            dist = abs(t.start_sec - s.start_sec)
            if dist < best_dist and dist <= _SCRAPE_OVERLAP_WINDOW_SEC:
                best_dist = dist
                match_idx = i
        if match_idx >= 0 and SOURCE_PRIORITY.get(s.source, 0) > audio_priority:
            existing = final[match_idx]
            final[match_idx] = TracklistTrack(
                start_sec=existing.start_sec,
                end_sec=existing.end_sec,
                artist=s.artist,
                title=s.title,
                score=1.0,
                source=s.source,
                n_segments_merged=existing.n_segments_merged,
            )
            claimed_audio.add(match_idx)
            continue
        # No unclaimed audio match → insert a new track.
        final.append(
            TracklistTrack(
                start_sec=float(s.start_sec),
                end_sec=float(s.start_sec) + 60.0,  # placeholder, refined after sort
                artist=s.artist,
                title=s.title,
                score=1.0,
                source=s.source,
                n_segments_merged=1,
            )
        )

    final.sort(key=lambda t: t.start_sec)

    # Refine end_sec for inserted scrape-only tracks: use next track's start
    # (or mix duration for the last one). Only touch tracks where score==1.0
    # AND source is one of the scrape sources, to avoid clobbering audio.
    scrape_sources = set(SOURCE_PRIORITY.keys())
    for i, t in enumerate(final):
        if t.source not in scrape_sources:
            continue
        next_start = final[i + 1].start_sec if i + 1 < len(final) else mix_duration_sec
        # Only widen, never shrink (don't overwrite audio's measured end_sec)
        if next_start > t.start_sec:
            final[i] = TracklistTrack(
                start_sec=t.start_sec,
                end_sec=float(next_start),
                artist=t.artist,
                title=t.title,
                score=t.score,
                source=t.source,
                n_segments_merged=t.n_segments_merged,
            )

    # Rule 4: drop unknowns enclosed by consecutive scraped entries < 5 min apart
    scraped_starts = sorted({float(e.start_sec) for e in timed_scraped})
    surviving_unknowns: list[tuple[float, float]] = []
    for u_start, u_end in unknown_segments:
        u_mid = (u_start + u_end) / 2
        # Find scraped entries that bracket u_mid
        before = max((s for s in scraped_starts if s <= u_mid), default=None)
        after = min((s for s in scraped_starts if s >= u_mid), default=None)
        if before is not None and after is not None and (after - before) <= _GAP_TOO_BIG_SEC:
            continue  # covered by scrape
        surviving_unknowns.append((u_start, u_end))

    # Rule 5: untimed entries.
    # We don't know where these go in the mix. To avoid lying about
    # positions, cluster every unmatched untimed entry at a single
    # "tail" position one second before mix end. They sort last,
    # share the same timestamp, and a future UI can render them as
    # "may also include" rather than as real cue points.
    final_keys = {(_norm(t.artist), _norm(t.title)) for t in final}
    tail_pos = max(0.0, mix_duration_sec - 1.0)
    for u in untimed_scraped:
        key = (_norm(u.artist), _norm(u.title))
        if key in final_keys:
            continue
        final.append(
            TracklistTrack(
                start_sec=tail_pos,
                end_sec=mix_duration_sec,
                artist=u.artist,
                title=u.title,
                score=1.0,
                source=u.source,
                n_segments_merged=1,
            )
        )
        final_keys.add(key)

    return final, surviving_unknowns
