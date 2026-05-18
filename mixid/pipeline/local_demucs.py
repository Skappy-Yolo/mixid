"""Local Demucs stem separation for noise-removal-then-retry.

When Shazam fails on a segment (likely because crowd chatter + reverb
corrupts the music too much for Apple's fingerprint to grip), we can
run Demucs to separate the audio into vocals/drums/bass/other stems,
then retry Shazam on a stem that's less polluted by chatter.

For most electronic / rave music, the `no_vocals` composite (drums+bass+
other) is the cleanest target — crowd chatter is mostly vocal-frequency
content, so demucs puts it into the vocals stem alongside any actual
singing, leaving the instrumental backbone for Shazam.

CPU-only Demucs is slow (~10-30 sec per 16-sec segment with htdemucs_ft).
We gate this stage behind an explicit --with-demucs flag because most
users will be fine with the Shazam-only pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


_MODEL_CACHE: dict = {}


def is_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load_model(name: str = "htdemucs_ft"):
    """Cache the Demucs model — loading takes 5-10 sec each time."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    from demucs.pretrained import get_model

    log.info("Loading demucs %s (one-time, downloads ~80 MB if missing)…", name)
    model = get_model(name)
    model.eval()
    _MODEL_CACHE[name] = model
    return model


def separate(samples: np.ndarray, sr: int, model_name: str = "htdemucs_ft") -> Optional[dict[str, np.ndarray]]:
    """Run Demucs on the given mono audio. Returns dict of stems → mono float32.

    Stems returned: 'vocals', 'drums', 'bass', 'other', plus a synthetic
    'no_vocals' = drums+bass+other. Returns None if demucs not installed.
    """
    if not is_available():
        return None
    import torch
    from demucs.apply import apply_model
    import librosa

    model = _load_model(model_name)
    target_sr = int(model.samplerate)
    if sr != target_sr:
        samples = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    samples = samples.astype(np.float32)

    # Demucs wants (batch, channels, samples) at model.samplerate; channels
    # must match model.audio_channels (usually 2 stereo). Duplicate mono.
    if samples.ndim == 1:
        stereo = np.stack([samples, samples])
    else:
        stereo = samples
    audio = torch.from_numpy(stereo).float().unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(model, audio, split=True, overlap=0.25)
    sources = sources.squeeze(0).cpu().numpy()  # (n_sources, channels, samples)

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(model.sources):
        out[name] = sources[i].mean(axis=0).astype(np.float32)  # downmix to mono
    # Synthetic 'no_vocals' = instrumental composite
    instrumental_keys = [k for k in out if k != "vocals"]
    if instrumental_keys:
        out["no_vocals"] = sum(out[k] for k in instrumental_keys).astype(np.float32)
    return out


def separate_at_sr(samples: np.ndarray, sr: int, output_sr: int) -> Optional[dict[str, np.ndarray]]:
    """Like `separate` but resample each stem back to `output_sr` (matches the rest of the pipeline)."""
    stems = separate(samples, sr)
    if stems is None:
        return None
    import librosa
    target_sr = int(_load_model().samplerate)
    if output_sr == target_sr:
        return stems
    return {
        name: librosa.resample(arr, orig_sr=target_sr, target_sr=output_sr).astype(np.float32)
        for name, arr in stems.items()
    }
