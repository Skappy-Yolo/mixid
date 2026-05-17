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
    """One Chromaprint fingerprint, possibly at a pitch shift.

    Carries both representations: `b64` for AcoustID submission and
    `raw_hashes` for local Hamming-distance matching. Either may be
    empty depending on how the fingerprint was generated; matchers
    that need a specific form check first.
    """
    pitch_shift_percent: int  # 0 for original; -6..+6 for shifts
    duration_secs: float
    b64: str = ""                   # Chromaprint base64 — AcoustID input
    raw_hashes: np.ndarray | None = None  # uint32 hashes — for local match


@dataclass
class FingerprintSweep:
    """All pitch variants fingerprinted for one sample. Matchers take the best."""
    fingerprints: list[Fingerprint]
    sample_start_sec_in_mix: float

    @property
    def base(self) -> Fingerprint:
        """The unshifted (0%) fingerprint."""
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


def _run_fpcalc(wav_path: Path, raw: bool = False, length_secs: int = 120) -> Fingerprint:
    """Invoke fpcalc on wav_path. raw=True → uint32 hashes; raw=False → base64."""
    fpcalc = _ensure_fpcalc()
    args = [str(fpcalc), "-json", "-length", str(length_secs)]
    if raw:
        args.insert(1, "-raw")
    args.append(str(wav_path))
    res = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"fpcalc failed: {res.stderr}")
    import json as _json

    data = _json.loads(res.stdout)
    fp = Fingerprint(pitch_shift_percent=0, duration_secs=float(data["duration"]))
    if raw:
        fp.raw_hashes = np.asarray(data["fingerprint"], dtype=np.uint32)
    else:
        fp.b64 = str(data["fingerprint"])
    return fp


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
    include_b64: bool = True,
) -> FingerprintSweep:
    """Fingerprint one audio sample at every pitch shift in `pitch_shifts_pct`.

    Computes raw uint32 hashes for every variant (used by the local matcher).
    Also computes base64 form when include_b64=True (used by AcoustID submission).
    """
    sweep: list[Fingerprint] = []
    with tempfile.TemporaryDirectory(prefix="mixid_fp_") as tmp:
        for pct in pitch_shifts_pct:
            shifted = samples if pct == 0 else _pitch_shift_samples(samples, sr, pct)
            wav_path = Path(tmp) / f"sample_{pct:+d}.wav"
            sf.write(str(wav_path), shifted, sr, subtype="PCM_16")
            fp_raw = _run_fpcalc(wav_path, raw=True)
            fp = Fingerprint(
                pitch_shift_percent=pct,
                duration_secs=fp_raw.duration_secs,
                raw_hashes=fp_raw.raw_hashes,
            )
            if include_b64:
                fp_b64 = _run_fpcalc(wav_path, raw=False)
                fp.b64 = fp_b64.b64
            sweep.append(fp)
    return FingerprintSweep(
        fingerprints=sweep,
        sample_start_sec_in_mix=sample_start_sec_in_mix,
    )


def fingerprint_file(path: Path | str, raw: bool = False) -> Fingerprint:
    """Fingerprint a whole audio file (no pitch sweep).

    raw=False (default) → base64 form, used by AcoustID submission.
    raw=True            → uint32 hash array, used by the library indexer
                          and the local Hamming-distance matcher.
    """
    return _run_fpcalc(Path(path), raw=raw)
