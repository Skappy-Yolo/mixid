"""Pick a 12-second 'best sample' from each segment for fingerprinting.

A segment can be 30s or 5min; we don't fingerprint the whole thing. We
look for a window where the music is steady (low spectral variance — not
inside a transition) and loud (high RMS — not a breakdown or a quiet
intro). That's the SAMPLE_WINDOW_SECS-long slice we send to Chromaprint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from mixid.pipeline.segment import Segment


@dataclass
class Sample:
    samples: np.ndarray  # 1-D float32, length = window_secs * sample_rate
    start_sec_in_mix: float
    sample_rate: int
    segment: Segment


def _windowed_rms(samples: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = (len(samples) - win) // hop + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        chunk = samples[s : s + win]
        out[i] = float(np.sqrt(np.mean(chunk**2)))
    return out


def _windowed_spectral_variance(samples: np.ndarray, sr: int, win: int, hop: int) -> np.ndarray:
    """Variance of the spectral centroid over each window — proxy for steadiness."""
    import librosa

    n = (len(samples) - win) // hop + 1
    if n <= 0:
        return np.array([], dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        chunk = samples[s : s + win]
        cent = librosa.feature.spectral_centroid(y=chunk, sr=sr).ravel()
        out[i] = float(np.var(cent))
    return out


def pick(
    full_samples: np.ndarray,
    sr: int,
    seg: Segment,
    window_secs: float = config.SAMPLE_WINDOW_SECS,
) -> Sample | None:
    """Pick the best window inside `seg`. None if segment too short."""
    picks = pick_top_k(full_samples, sr, seg, window_secs=window_secs, k=1)
    return picks[0] if picks else None


def pick_top_k(
    full_samples: np.ndarray,
    sr: int,
    seg: Segment,
    window_secs: float = config.SAMPLE_WINDOW_SECS,
    k: int = 3,
) -> list[Sample]:
    """Pick the top-K non-overlapping windows by (RMS - spectral_variance) score.

    Returns up to k Samples ordered by descending score. Used when one
    Shazam attempt per segment isn't enough — we try the loudest steadiest
    spot first, then the next-best, etc. Each pick is at least one window
    apart from the others so we don't repeatedly fingerprint the same drop.
    """
    win = int(window_secs * sr)
    seg_samples = full_samples[int(seg.start_sec * sr) : int(seg.end_sec * sr)]
    if len(seg_samples) < win:
        return []

    hop = max(win // 4, sr)
    rms = _windowed_rms(seg_samples, win, hop)
    var = _windowed_spectral_variance(seg_samples, sr, win, hop)
    if len(rms) == 0:
        return []

    rms_z = (rms - rms.mean()) / (rms.std() + 1e-9)
    var_z = (var - var.mean()) / (var.std() + 1e-9)
    score = rms_z - var_z

    # Greedy non-overlapping selection: pick highest, suppress its window,
    # repeat. min_gap_hops keeps picks ~one window apart.
    min_gap_hops = max(1, win // hop)
    chosen_idx: list[int] = []
    available = np.ones_like(score, dtype=bool)
    for _ in range(k):
        if not available.any():
            break
        masked = np.where(available, score, -np.inf)
        best = int(np.argmax(masked))
        if not np.isfinite(masked[best]):
            break
        chosen_idx.append(best)
        lo = max(0, best - min_gap_hops)
        hi = min(len(score), best + min_gap_hops + 1)
        available[lo:hi] = False

    out: list[Sample] = []
    for idx in chosen_idx:
        start_offset = idx * hop
        out.append(Sample(
            samples=seg_samples[start_offset : start_offset + win].copy(),
            start_sec_in_mix=seg.start_sec + start_offset / sr,
            sample_rate=sr,
            segment=seg,
        ))
    return out
