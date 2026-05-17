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
    win = int(window_secs * sr)
    seg_samples = full_samples[int(seg.start_sec * sr) : int(seg.end_sec * sr)]
    if len(seg_samples) < win:
        return None

    hop = max(win // 4, sr)  # step by 1 sec or quarter-window, whichever is bigger
    rms = _windowed_rms(seg_samples, win, hop)
    var = _windowed_spectral_variance(seg_samples, sr, win, hop)
    if len(rms) == 0:
        return None

    # Normalize each, combine: prefer high RMS, low spectral variance.
    rms_z = (rms - rms.mean()) / (rms.std() + 1e-9)
    var_z = (var - var.mean()) / (var.std() + 1e-9)
    score = rms_z - var_z
    best = int(np.argmax(score))
    start_offset = best * hop
    return Sample(
        samples=seg_samples[start_offset : start_offset + win].copy(),
        start_sec_in_mix=seg.start_sec + start_offset / sr,
        sample_rate=sr,
        segment=seg,
    )
