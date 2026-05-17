"""Audio preprocessing for MixID.

Input: any container ffmpeg understands (mp3, m4a, wav, mp4, flac, ...).
Output: a numpy float32 mono waveform at TARGET_SR, loudness-normalized
to LOUDNESS_TARGET_LUFS, with a Butterworth highpass at HIGHPASS_CUTOFF_HZ.

The whole point of preprocessing is to put every input into the same
predictable shape before segmentation and fingerprinting — phone party
recordings, YouTube rips, and clean studio sets all converge here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import config


@dataclass
class PreparedAudio:
    samples: np.ndarray  # 1-D float32, mono, TARGET_SR
    sample_rate: int
    duration_secs: float
    source_path: Path
    pre_norm_lufs: float
    post_norm_lufs: float


def load_mono(path: Path | str, sr: int = config.TARGET_SR) -> tuple[np.ndarray, int]:
    """Load any audio file, downmix to mono, resample to sr.

    Uses librosa, which delegates to soundfile or audioread depending on
    container. Returns float32 in [-1, 1].
    """
    import librosa  # local import keeps cold-start fast for non-audio code paths

    samples, native_sr = librosa.load(str(path), sr=sr, mono=True)
    return samples.astype(np.float32, copy=False), int(native_sr)


def highpass(
    samples: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = config.HIGHPASS_CUTOFF_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth highpass. Attenuates sub-rumble and crowd boom."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32, copy=False)


def normalize_loudness(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float = config.LOUDNESS_TARGET_LUFS,
) -> tuple[np.ndarray, float, float]:
    """ITU-R BS.1770 loudness normalization. Returns (samples, pre, post)."""
    meter = pyln.Meter(sample_rate)
    pre = float(meter.integrated_loudness(samples))
    if not np.isfinite(pre):
        return samples, pre, pre
    normalized = pyln.normalize.loudness(samples, pre, target_lufs).astype(
        np.float32, copy=False
    )
    post = float(meter.integrated_loudness(normalized))
    return normalized, pre, post


def prepare(path: Path | str) -> PreparedAudio:
    """Full preprocessing chain: load → mono → resample → highpass → loudness norm."""
    path = Path(path)
    samples, sr = load_mono(path)
    samples = highpass(samples, sr)
    samples, pre, post = normalize_loudness(samples, sr)
    return PreparedAudio(
        samples=samples,
        sample_rate=sr,
        duration_secs=len(samples) / sr,
        source_path=path,
        pre_norm_lufs=pre,
        post_norm_lufs=post,
    )


def write_wav(prepared: PreparedAudio, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), prepared.samples, prepared.sample_rate, subtype="PCM_16")
    return out_path
