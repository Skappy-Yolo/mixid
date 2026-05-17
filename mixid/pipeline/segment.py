"""Hybrid segmentation for DJ mixes.

A DJ mix is one continuous audio stream; identification needs *segments*
(track boundaries) so we know what to fingerprint. Three signals stacked:

1. Novelty curve over MFCC + chroma — catches genre/instrumentation shifts.
2. Beat-grid phrase boundaries (every 16/32 bars) — catches DJ-style swaps
   that happen on phrase grid without an MFCC novelty spike.
3. Max-gap cap — force a boundary every MAX_SEGMENT_GAP_SECS no matter what,
   so a long blend can never hide a track swap silently.

Pure novelty curves fail on long blends and stem swaps. The three signals
together don't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    boundary_source: str  # "novelty" | "phrase" | "gap" | "start" | "end"


def _novelty_boundaries(samples: np.ndarray, sr: int) -> list[float]:
    """Agglomerative segmentation on MFCC + chroma. Returns boundaries in seconds."""
    import librosa

    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)
    feat = np.vstack([mfcc, chroma])
    # Self-similarity-based segmentation. k=None lets librosa pick using
    # the gap statistic; we override below if we want a target count.
    bounds = librosa.segment.agglomerative(feat, k=max(8, int(len(samples) / sr / 60)))
    return librosa.frames_to_time(bounds, sr=sr).tolist()


def _phrase_boundaries(samples: np.ndarray, sr: int) -> list[float]:
    """Beat-grid phrase boundaries: every 16 bars at the detected tempo."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=samples, sr=sr, units="time")
    if len(beats) < 64:
        return []
    # Assume 4/4. 16 bars = 64 beats.
    phrase_size = 64
    return [float(beats[i]) for i in range(phrase_size, len(beats), phrase_size)]


def _enforce_gap_cap(
    boundaries: list[float],
    duration_sec: float,
    max_gap: float = config.MAX_SEGMENT_GAP_SECS,
) -> tuple[list[float], list[str]]:
    """Insert synthetic boundaries so no two consecutive ones are >max_gap apart.

    Returns (sorted_boundaries, sources) — sources marks which entries are
    "gap"-inserted vs original.
    """
    sorted_b = sorted(set(boundaries))
    out: list[float] = []
    src: list[str] = []
    last = 0.0
    for b in sorted_b:
        while b - last > max_gap:
            last = last + max_gap
            out.append(last)
            src.append("gap")
        out.append(b)
        src.append("orig")
        last = b
    while duration_sec - last > max_gap:
        last = last + max_gap
        out.append(last)
        src.append("gap")
    return out, src


def segment(samples: np.ndarray, sr: int) -> list[Segment]:
    """Run all three segmentation signals, fuse, return ordered segments.

    Boundaries within 3 seconds of each other are merged (the earlier wins).
    """
    duration = len(samples) / sr
    nov = _novelty_boundaries(samples, sr)
    phr = _phrase_boundaries(samples, sr)

    # Tag each by source.
    tagged: list[tuple[float, str]] = (
        [(b, "novelty") for b in nov] + [(b, "phrase") for b in phr]
    )
    tagged.sort(key=lambda x: x[0])

    # Merge near-duplicates (within 3 sec), preferring novelty source.
    merged: list[tuple[float, str]] = []
    for t, source in tagged:
        if merged and abs(t - merged[-1][0]) < 3.0:
            # Keep the one that's already there if it's a novelty hit.
            if merged[-1][1] != "novelty" and source == "novelty":
                merged[-1] = (t, source)
            continue
        merged.append((t, source))

    # Apply the gap cap to ensure no >MAX_GAP gaps.
    just_times = [t for t, _ in merged]
    gapped, gap_src = _enforce_gap_cap(just_times, duration)

    # Re-tag — preserve original sources, mark gap-inserted entries.
    final: list[tuple[float, str]] = []
    orig_idx = 0
    for t, gs in zip(gapped, gap_src):
        if gs == "gap":
            final.append((t, "gap"))
        else:
            # Same time as merged[orig_idx]
            while orig_idx < len(merged) and merged[orig_idx][0] < t - 0.01:
                orig_idx += 1
            if orig_idx < len(merged):
                final.append((t, merged[orig_idx][1]))
                orig_idx += 1
            else:
                final.append((t, "orig"))

    # Build segments from boundary list, prepending start and appending end.
    starts = [0.0] + [t for t, _ in final]
    ends = [t for t, _ in final] + [duration]
    sources = ["start"] + [s for _, s in final]
    out: list[Segment] = []
    for s, e, src in zip(starts, ends, sources):
        if e - s < 5.0:  # ignore degenerate sub-5-sec segments
            continue
        out.append(Segment(start_sec=s, end_sec=e, boundary_source=src))
    if not out or out[-1].end_sec < duration - 1.0:
        out.append(
            Segment(
                start_sec=out[-1].end_sec if out else 0.0,
                end_sec=duration,
                boundary_source="end",
            )
        )
    return out
