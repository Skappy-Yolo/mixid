"""Segmentation behaves correctly on synthetic and silence inputs."""
from __future__ import annotations

import numpy as np

import config
from mixid.pipeline import segment as seg_mod


def _make_two_track_mix(sr: int = config.TARGET_SR, dur_each: float = 30.0) -> np.ndarray:
    """Two distinct sections back-to-back, sharp boundary in the middle."""
    t = np.linspace(0, dur_each, int(sr * dur_each), endpoint=False, dtype=np.float32)
    track_a = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    track_b = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return np.concatenate([track_a, track_b])


def test_gap_cap_inserts_when_needed():
    boundaries, sources = seg_mod._enforce_gap_cap(
        [10.0, 200.0], duration_sec=300.0, max_gap=90.0
    )
    # Gaps: 0→10 OK, 10→100→190→200 (insert 100, 190), 200→290 (insert 290)
    assert any(s == "gap" for s in sources)
    # No two consecutive boundaries are >90s apart
    augmented = [0.0] + boundaries + [300.0]
    for prev, cur in zip(augmented[:-1], augmented[1:]):
        assert cur - prev <= 90.001


def test_segment_returns_ordered_non_overlapping():
    samples = _make_two_track_mix(dur_each=20.0)
    segs = seg_mod.segment(samples, config.TARGET_SR)
    assert len(segs) >= 1
    for prev, cur in zip(segs[:-1], segs[1:]):
        assert prev.end_sec <= cur.start_sec + 0.001
        assert cur.end_sec > cur.start_sec
