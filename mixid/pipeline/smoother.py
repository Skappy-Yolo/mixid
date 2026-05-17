"""Gap-fill smoother — rescue unidentified segments sandwiched by confident hits.

DJs typically hold each track for 60-180 seconds. When fingerprinting
fails on a middle segment but its neighbors confidently match the same
track, that segment is almost certainly the same track too — the
fingerprint failed because the DJ applied an effect (filter sweep,
delay, EQ kill) at that moment, not because the track actually changed.

This is the simplest 'HMM-like' rescue we can ship: a single-pass
neighbor median filter over identified results. No BPM model, no key
wheel — those land in a later phase. Real lift on long held-track
sections; conservative enough to never hallucinate.
"""
from __future__ import annotations

from dataclasses import replace

from mixid.pipeline.reranker import RerankedResult


def smooth_gaps(
    reranked: list[RerankedResult],
    unknown_segments: list[tuple[float, float]],
    *,
    max_gap_secs: float = 60.0,
) -> tuple[list[RerankedResult], list[tuple[float, float]]]:
    """Fill an unknown segment when both temporal neighbors agree.

    Returns (augmented_results, remaining_unknowns).

    Rules (intentionally conservative):
    - Unknown is rescued only if BOTH temporal neighbors exist and identify
      the same artist+title.
    - Unknown must be no longer than `max_gap_secs` — a 5-minute silent gap
      is not a single track held; that's two tracks with a break in between.
    - Rescued segments are tagged `source='smoothed:<original_source>'` so
      output consumers can tell they're inferred, not fingerprinted.
    """
    if not unknown_segments or not reranked:
        return list(reranked), list(unknown_segments)

    sorted_results = sorted(reranked, key=lambda r: r.start_sec)
    augmented = list(sorted_results)
    remaining: list[tuple[float, float]] = []

    for (us, ue) in unknown_segments:
        if ue - us > max_gap_secs:
            remaining.append((us, ue))
            continue
        before = _last_before(sorted_results, us)
        after = _first_after(sorted_results, ue)
        if before is None or after is None:
            remaining.append((us, ue))
            continue
        if not _same_track(before, after):
            remaining.append((us, ue))
            continue
        augmented.append(
            replace(
                before,
                segment_index=-1,  # synthetic; not a real input segment
                start_sec=us,
                end_sec=ue,
                source=f"smoothed:{before.source}",
                rationale=f"gap-filled between {before.start_sec:.0f}s and {after.start_sec:.0f}s",
                score=min(before.score, after.score),
            )
        )

    augmented.sort(key=lambda r: r.start_sec)
    return augmented, remaining


def _last_before(results: list[RerankedResult], t: float) -> RerankedResult | None:
    """Find the result whose end_sec is closest to (but at most) t."""
    found = None
    for r in results:
        if r.end_sec <= t + 1.0:  # allow 1-sec slack for boundary alignment
            if found is None or r.end_sec > found.end_sec:
                found = r
        else:
            break  # sorted_results is in order
    return found


def _first_after(results: list[RerankedResult], t: float) -> RerankedResult | None:
    """Find the result whose start_sec is closest to (but at least) t."""
    for r in results:
        if r.start_sec >= t - 1.0:
            return r
    return None


def _same_track(a: RerankedResult, b: RerankedResult) -> bool:
    return (
        a.artist.strip().lower() == b.artist.strip().lower()
        and a.title.strip().lower() == b.title.strip().lower()
    )
