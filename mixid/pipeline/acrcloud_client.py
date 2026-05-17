"""ACRCloud Identification client — Tier-2 fallback for AcoustID gaps.

ACRCloud's catalogue (~100M+ tracks) covers electronic / Afrobeats /
edits / white-labels that AcoustID's MusicBrainz-backed index often
misses. The free trial gives ~14k queries / 14 days — enough to clear
a backlog of 200+ mixes before the trial expires. Burned correctly, it
is a meaningful coverage win without recurring cost.

We submit a 10-second raw audio snippet (their recommended length) per
unknown segment. Authentication is HMAC-SHA1 over a small canonical
string; no SDK required.

If ACRCLOUD_KEY / ACRCLOUD_SECRET / ACRCLOUD_HOST are not set, the
client returns None gracefully and logs a one-line warning — the rest
of the pipeline keeps working.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import logging
import time
from dataclasses import dataclass

import numpy as np
import requests
import soundfile as sf

import config


log = logging.getLogger(__name__)

_DATA_TYPE = "audio"
_SIG_VERSION = "1"
_ENDPOINT_PATH = "/v1/identify"
_DEFAULT_TIMEOUT_SECS = 15
_RECOMMENDED_SAMPLE_SECS = 10


@dataclass
class ACRCloudMatch:
    score: float
    title: str
    artist: str
    album: str
    duration_ms: int | None
    acrid: str          # ACRCloud internal ID


def _build_signature(access_key: str, access_secret: str, timestamp: str) -> str:
    """HMAC-SHA1 over the canonical string ACRCloud expects."""
    string_to_sign = (
        f"POST\n{_ENDPOINT_PATH}\n{access_key}\n{_DATA_TYPE}\n{_SIG_VERSION}\n{timestamp}"
    )
    digest = hmac.new(
        access_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _parse_response(resp: dict) -> ACRCloudMatch | None:
    """Pull the highest-confidence music match out of the ACRCloud envelope."""
    if not isinstance(resp, dict):
        return None
    status = resp.get("status") or {}
    if status.get("code") != 0:  # ACRCloud returns code 0 for success
        return None
    metadata = resp.get("metadata") or {}
    music = metadata.get("music") or []
    if not music:
        return None
    # First entry is highest-confidence in ACRCloud's ordering
    m = music[0]
    artists = m.get("artists") or []
    artist = ", ".join(a.get("name", "") for a in artists if a.get("name"))
    album = (m.get("album") or {}).get("name", "")
    return ACRCloudMatch(
        score=float(m.get("score", 0.0)) / 100.0,  # ACRCloud reports 0-100
        title=m.get("title", "") or "",
        artist=artist,
        album=album,
        duration_ms=int(m["duration_ms"]) if m.get("duration_ms") else None,
        acrid=m.get("acrid", "") or "",
    )


def lookup_sample(
    samples: np.ndarray,
    sr: int,
) -> ACRCloudMatch | None:
    """Submit one audio snippet (up to ~10 sec) to ACRCloud. Returns top match or None."""
    if not (config.ACRCLOUD_KEY and config.ACRCLOUD_SECRET and config.ACRCLOUD_HOST):
        log.warning(
            "ACRCloud not configured — set ACRCLOUD_KEY/SECRET/HOST in .env "
            "to enable. Tier-2 fallback for AcoustID gaps."
        )
        return None

    # Trim to ~10 sec — ACRCloud truncates anyway, less upload = less latency
    max_len = int(_RECOMMENDED_SAMPLE_SECS * sr)
    if len(samples) > max_len:
        samples = samples[:max_len]

    audio_bytes = _samples_to_wav_bytes(samples, sr)
    timestamp = str(int(time.time()))
    signature = _build_signature(
        config.ACRCLOUD_KEY, config.ACRCLOUD_SECRET, timestamp
    )

    files = {"sample": ("sample.wav", audio_bytes, "audio/wav")}
    data = {
        "access_key": config.ACRCLOUD_KEY,
        "data_type": _DATA_TYPE,
        "signature_version": _SIG_VERSION,
        "signature": signature,
        "sample_bytes": str(len(audio_bytes)),
        "timestamp": timestamp,
    }
    url = f"https://{config.ACRCLOUD_HOST}{_ENDPOINT_PATH}"
    try:
        r = requests.post(url, files=files, data=data, timeout=_DEFAULT_TIMEOUT_SECS)
    except requests.RequestException as e:
        log.warning("ACRCloud request failed: %s", e)
        return None
    if not r.ok:
        log.warning("ACRCloud HTTP %s: %s", r.status_code, r.text[:200])
        return None
    try:
        return _parse_response(r.json())
    except ValueError:
        return None
