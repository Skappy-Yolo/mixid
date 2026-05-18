"""Shazam recognition via the shazamio reverse-engineered Python client.

This is the same Shazam everyone uses on their phone. Coverage is
dramatically better than AcoustID for non-mainstream music (Afrobeats,
electronic edits, niche indie). No API key, no quotas — but it's an
unofficial client that Apple/Shazam could break at any time. As of
2026-05 the library is updated and working.

We expose a synchronous `recognize_sample(samples, sr)` that wraps the
async shazamio call so the rest of the pipeline doesn't need to know
about asyncio. Falls back to None on any error so the pipeline degrades
gracefully if Shazam blocks our IP or shazamio gets stale.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


log = logging.getLogger(__name__)


@dataclass
class ShazamMatch:
    title: str
    artist: str
    score: float = 1.0          # Shazam returns binary match; we report 1.0 if recognized
    shazam_key: str = ""        # internal Shazam track key
    isrc: str = ""              # ISRC code if returned


# Per-process rate limit. Shazam tolerates ~1 query/sec from unofficial
# clients; faster gets you transient IP blocks.
_MIN_INTERVAL_SECS = 1.0
_LAST_CALL_AT: float = 0.0


def _throttle() -> None:
    global _LAST_CALL_AT
    elapsed = time.monotonic() - _LAST_CALL_AT
    if elapsed < _MIN_INTERVAL_SECS:
        time.sleep(_MIN_INTERVAL_SECS - elapsed)
    _LAST_CALL_AT = time.monotonic()


def _parse_shazam_response(resp: dict) -> ShazamMatch | None:
    track = resp.get("track") if isinstance(resp, dict) else None
    if not track:
        return None
    title = track.get("title") or ""
    artist = track.get("subtitle") or ""
    if not title and not artist:
        return None
    key = track.get("key") or ""
    # ISRC sits in genres/isrc or hub.actions/etc — best-effort extract.
    isrc = ""
    for section in (track.get("sections") or []):
        for meta in (section.get("metadata") or []):
            if (meta.get("title") or "").lower() == "isrc":
                isrc = meta.get("text", "") or ""
    return ShazamMatch(title=title, artist=artist, shazam_key=str(key), isrc=isrc)


async def _recognize_async(path: Path) -> dict:
    from shazamio import Shazam
    sh = Shazam()
    return await sh.recognize(str(path))


def recognize_file(path: Path | str) -> ShazamMatch | None:
    """Recognize an audio file via Shazam. Path must be readable by ffmpeg."""
    try:
        from shazamio import Shazam  # noqa: F401  triggers ImportError if not installed
    except ImportError:
        log.warning("shazamio not installed — skipping Shazam (pip install shazamio)")
        return None
    _throttle()
    try:
        resp = asyncio.run(_recognize_async(Path(path)))
    except Exception as e:
        log.warning("Shazam recognize failed for %s: %s", path, e)
        return None
    return _parse_shazam_response(resp)


def recognize_sample(samples: np.ndarray, sr: int) -> ShazamMatch | None:
    """Write samples to a temp wav and recognize. Returns None if no match."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, samples, sr, subtype="PCM_16")
        path = Path(f.name)
    try:
        return recognize_file(path)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
