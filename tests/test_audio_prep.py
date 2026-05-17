"""Real tests against synthesized signals — no mocks.

The preprocessing pipeline must:
1. Resample to TARGET_SR
2. Downmix to mono
3. Attenuate energy below the highpass cutoff
4. Pull loudness close to the LUFS target
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

import config
from mixid.pipeline import audio_prep


def _make_test_wav(tmp_path, freq_hz: float = 440.0, sr: int = 44100, dur: float = 3.0):
    """Synthesize a stereo sine, write it. Returns the path."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    mono = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    stereo = np.stack([mono, mono * 0.9], axis=-1)
    path = tmp_path / "tone.wav"
    sf.write(str(path), stereo, sr)
    return path


def test_load_mono_downmixes_and_resamples(tmp_path):
    path = _make_test_wav(tmp_path, sr=44100)
    samples, sr = audio_prep.load_mono(path, sr=config.TARGET_SR)
    assert samples.ndim == 1
    assert sr == config.TARGET_SR
    # ~3 seconds at TARGET_SR ± a few samples for resampler edge effects
    assert abs(len(samples) - int(3.0 * config.TARGET_SR)) < 200


def test_highpass_attenuates_subbass(tmp_path):
    sr = config.TARGET_SR
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    # 40 Hz tone — below the 80 Hz highpass cutoff
    sub = 0.5 * np.sin(2 * np.pi * 40.0 * t).astype(np.float32)
    filtered = audio_prep.highpass(sub, sr)
    pre_rms = float(np.sqrt(np.mean(sub**2)))
    post_rms = float(np.sqrt(np.mean(filtered**2)))
    assert post_rms < pre_rms * 0.2, "highpass should kill most sub-cutoff energy"


def test_highpass_preserves_supra_cutoff(tmp_path):
    sr = config.TARGET_SR
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    # 1 kHz tone — well above the highpass
    sig = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    filtered = audio_prep.highpass(sig, sr)
    pre_rms = float(np.sqrt(np.mean(sig**2)))
    post_rms = float(np.sqrt(np.mean(filtered**2)))
    assert post_rms > pre_rms * 0.95, "highpass must not attenuate above cutoff"


def test_normalize_loudness_targets_lufs(tmp_path):
    sr = config.TARGET_SR
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32)
    # Very quiet 1 kHz tone
    quiet = 0.01 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    out, pre, post = audio_prep.normalize_loudness(quiet, sr)
    assert pre < post, "should be louder after normalization"
    assert abs(post - config.LOUDNESS_TARGET_LUFS) < 1.5


def test_prepare_end_to_end(tmp_path):
    path = _make_test_wav(tmp_path, freq_hz=1000.0, sr=44100, dur=2.0)
    prepared = audio_prep.prepare(path)
    assert prepared.sample_rate == config.TARGET_SR
    assert prepared.samples.ndim == 1
    assert prepared.duration_secs > 1.5
    assert prepared.source_path == path
    assert np.isfinite(prepared.post_norm_lufs)
