"""Chromaprint fingerprinting + pitch-shift sweep tests.

Requires fpcalc.exe in bin/. Skipped automatically if missing.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

import config
from mixid.pipeline import fingerprint


pytestmark = pytest.mark.skipif(
    not config.FPCALC_EXE.exists(),
    reason=f"fpcalc binary not present at {config.FPCALC_EXE}",
)


def _make_sample(sr: int = config.TARGET_SR, dur: float = 15.0) -> np.ndarray:
    """Broadband 'song-like' signal long enough for fpcalc to fingerprint.

    Pure sines are a poor proxy for music — Chromaprint chroma bins are
    1 semitone wide, so a 0.72-semitone (6%) shift on a 440 Hz sine may
    stay in the same bin and produce an identical fingerprint. Real music
    has broadband content that always crosses bin boundaries. We simulate
    that with a chord + filtered noise + an FM sweep.
    """
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    # C major chord: C4, E4, G4, plus their octaves
    chord = sum(
        0.15 * np.sin(2 * np.pi * f * t) for f in (262.0, 330.0, 392.0, 524.0, 660.0, 784.0)
    )
    # Slow FM sweep to spread spectral content
    sweep = 0.1 * np.sin(2 * np.pi * (500 + 200 * np.sin(2 * np.pi * 0.3 * t)) * t)
    # Pink-ish broadband noise (low-passed white)
    noise = rng.standard_normal(len(t)).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, 2000.0, btype="low", fs=sr, output="sos")
    noise = 0.05 * sosfiltfilt(sos, noise)
    return (chord + sweep + noise).astype(np.float32)


def test_fingerprint_file_returns_nonempty(tmp_path):
    sig = _make_sample()
    p = tmp_path / "sig.wav"
    sf.write(str(p), sig, config.TARGET_SR)
    fp = fingerprint.fingerprint_file(p)
    assert fp.duration_secs > 10
    assert len(fp.fingerprint) > 50  # base64 fingerprint string


def test_pitch_sweep_produces_one_per_shift():
    sig = _make_sample()
    sweep = fingerprint.fingerprint_sample(
        sig,
        config.TARGET_SR,
        sample_start_sec_in_mix=42.0,
        pitch_shifts_pct=(-4, 0, 4),
    )
    assert len(sweep.fingerprints) == 3
    shifts = {fp.pitch_shift_percent for fp in sweep.fingerprints}
    assert shifts == {-4, 0, 4}
    assert sweep.sample_start_sec_in_mix == 42.0


def test_pitch_sweep_produces_distinct_fingerprints():
    """Different pitches must yield different fingerprints; otherwise the sweep is a no-op."""
    sig = _make_sample()
    sweep = fingerprint.fingerprint_sample(
        sig,
        config.TARGET_SR,
        pitch_shifts_pct=(-6, 0, 6),
    )
    fps = {fp.pitch_shift_percent: fp.fingerprint for fp in sweep.fingerprints}
    assert fps[-6] != fps[0]
    assert fps[6] != fps[0]
    assert fps[-6] != fps[6]
