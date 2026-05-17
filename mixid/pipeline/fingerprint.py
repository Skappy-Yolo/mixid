"""Chromaprint fingerprinting with a pitch-shift sweep.

The single highest-ROI insight in the MixID design: Chromaprint is brittle
to the ±3-6% pitch DJs apply for beatmatching. A naive 1-fingerprint query
silently fails on every track the DJ touched. We fingerprint 7 variants
(0%, ±2%, ±4%, ±6%) and take the best match. Tradeoff: ~6× compute per
sample, still well under 1 second on CPU.

We don't use pyacoustid.fingerprint_file because it can only ingest paths.
The pipeline passes us a numpy array; we write a temp wav for fpcalc.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

import config


@dataclass
class Fingerprint:
    pitch_shift_percent: int  # 0 for original; -6..+6 for shifts
    duration_secs: float
    fingerprint: str  # base64 Chromaprint string


@dataclass
class FingerprintSweep:
    """All variants fingerprinted for one sample. Use the matcher's best score."""
    fingerprints: list[Fingerprint]
    sample_start_sec_in_mix: float

    @property
    def base(self) -> Fingerprint:
        """The unshifted (0%) fingerprint — used for naive comparison."""
        for fp in self.fingerprints:
            if fp.pitch_shift_percent == 0:
                return fp
        return self.fingerprints[0]


def _ensure_fpcalc() -> Path:
    if not config.FPCALC_EXE.exists():
        raise FileNotFoundError(
            f"fpcalc binary not found at {config.FPCALC_EXE}. "
            "Download from https://acoustid.org/chromaprint and place in bin/."
        )
    return config.FPCALC_EXE


def _run_fpcalc(wav_path: Path) -> Fingerprint:
    fpcalc = _ensure_fpcalc()
    res = subprocess.run(
        [str(fpcalc), "-json", "-length", "120", str(wav_path)],
        capture_output=True, text=True, timeout=30,
    )
    if res.returncode != 0:
        raise RuntimeError(f"fpcalc failed: {res.stderr}")
    import json

    data = json.loads(res.stdout)
    return Fingerprint(
        pitch_shift_percent=0,
        duration_secs=float(data["duration"]),
        fingerprint=str(data["fingerprint"]),
    )


def _pitch_shift_samples(samples: np.ndarray, sr: int, percent: float) -> np.ndarray:
    """Time-stretch + resample to shift pitch by `percent` without changing duration.

    For Chromaprint matching we only care that the *pitch class* shifts; we
    use the simplest approach: librosa.effects.pitch_shift with n_steps in
    semitones. 1% pitch ≈ 0.17 semitones.
    """
    import librosa

    n_steps = percent / 100.0 * 12.0  # +6% → +0.72 semitones
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=n_steps).astype(np.float32)


def fingerprint_sample(
    samples: np.ndarray,
    sr: int,
    sample_start_sec_in_mix: float = 0.0,
    pitch_shifts_pct: tuple[int, ...] = config.PITCH_SWEEP_PERCENT,
) -> FingerprintSweep:
    """Fingerprint one audio sample at every pitch shift in `pitch_shifts_pct`."""
    sweep: list[Fingerprint] = []
    with tempfile.TemporaryDirectory(prefix="mixid_fp_") as tmp:
        for pct in pitch_shifts_pct:
            shifted = samples if pct == 0 else _pitch_shift_samples(samples, sr, pct)
            wav_path = Path(tmp) / f"sample_{pct:+d}.wav"
            sf.write(str(wav_path), shifted, sr, subtype="PCM_16")
            fp = _run_fpcalc(wav_path)
            fp.pitch_shift_percent = pct
            sweep.append(fp)
    return FingerprintSweep(
        fingerprints=sweep,
        sample_start_sec_in_mix=sample_start_sec_in_mix,
    )


def fingerprint_file(path: Path | str) -> Fingerprint:
    """Convenience: fingerprint a whole audio file (no pitch sweep).

    Used by the library indexer to build the per-track reference fingerprints.
    """
    return _run_fpcalc(Path(path))
