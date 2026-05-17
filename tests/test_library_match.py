"""End-to-end: fingerprint a clip from a known library track and match it back.

This is the smoke test that proves the local accelerator works: index a
few tracks, take a 12-second sample from one of them, fingerprint with
pitch sweep, ask the matcher — should get a high-confidence hit on the
original track.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

import config
from mixid.pipeline import audio_prep, fingerprint, library_match


pytestmark = pytest.mark.skipif(
    not config.FPCALC_EXE.exists() or not config.FINGERPRINTS_DB.exists(),
    reason="fpcalc binary or library index not present",
)


def _pick_real_track() -> str:
    """Pick a real, existing music file from the indexed library (not a sampler)."""
    conn = sqlite3.connect(str(config.FINGERPRINTS_DB))
    rows = conn.execute(
        "SELECT filepath FROM tracks WHERE duration_secs > 30"
    ).fetchall()
    conn.close()
    if not rows:
        pytest.skip("library index has no tracks longer than 30 secs to sample from")
    from pathlib import Path

    for (fp,) in rows:
        if Path(fp).exists():
            return fp
    pytest.skip("no indexed track files exist on disk")


def test_match_clip_from_indexed_track():
    track_path = _pick_real_track()
    prepared = audio_prep.prepare(track_path)
    # Take seconds 30-42 — well past intros, into the body of the song.
    sr = prepared.sample_rate
    start = 30 * sr
    end = start + 12 * sr
    if end > len(prepared.samples):
        pytest.skip("track is too short for a 12-sec sample at 30 seconds in")
    sample = prepared.samples[start:end]

    sweep = fingerprint.fingerprint_sample(
        sample,
        sr,
        sample_start_sec_in_mix=30.0,
        # Use a small pitch sweep so the test is fast (~3-4 sec instead of ~10)
        pitch_shifts_pct=(-2, 0, 2),
        include_b64=False,
    )

    matches = library_match.match_sweep(sweep, top_k=3, min_score=0.5)
    assert matches, "expected at least one match against the source track"
    # The top match must be the track we sampled from
    assert matches[0].filepath == track_path
    # Sampled directly from the track — should be near-perfect at 0% pitch
    assert matches[0].score > 0.85
    assert matches[0].best_pitch_shift_percent == 0
