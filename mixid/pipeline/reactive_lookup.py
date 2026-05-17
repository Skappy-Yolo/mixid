"""Reactive lookup: identify unknown segments without a pre-built library.

The architecture insight: most tracks the user plays are never going to be
in any pre-built fingerprint cache, no matter how big. Instead of trying
to fingerprint the world, we work reactively per unknown segment:

  1. Transcribe ~12 sec of audio with Whisper       (free, CPU)
  2. Take the longest meaningful phrase
  3. Fan out a fuzzy search across free catalogs
     (iTunes Search + Deezer Search + Genius if key)
  4. Download each candidate's preview MP3          (free)
  5. Fingerprint each preview, compare to our segment locally
  6. Best match above floor wins

Works for any genre. Doesn't need any precomputed library. The transcribed
phrase is the only audio-to-text bridge — without working vocals, this
falls back through the chain just like the rest of the pipeline.
"""
from __future__ import annotations

import io
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

import config
from mixid.pipeline import fingerprint as fp_mod
from mixid.pipeline.library_match import _score_query_vs_library


log = logging.getLogger(__name__)


@dataclass
class ReactiveMatch:
    score: float                # fingerprint-compare score in [0, 1]
    artist: str
    title: str
    source: str                 # "reactive:itunes" | "reactive:deezer" | ...
    preview_url: str = ""
    transcript: str = ""


# Score floor for accepting a reactive hit. Conservative because we're
# matching a 12-sec query against a 30-sec preview that may not even
# overlap in time within the source track.
_MIN_MATCH_SCORE = 0.55


# ── Transcription ──────────────────────────────────────────────────────────


def _whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


_WHISPER_MODEL: dict = {}


def transcribe(samples: np.ndarray, sr: int, model_name: str = "tiny") -> str:
    """Whisper transcription of a numpy audio sample. Returns plain text."""
    import whisper

    if model_name not in _WHISPER_MODEL:
        log.info("Loading whisper-%s (~30 sec one-time)…", model_name)
        _WHISPER_MODEL[model_name] = whisper.load_model(model_name)
    model = _WHISPER_MODEL[model_name]
    if sr != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    samples = samples.astype(np.float32)
    audio = whisper.pad_or_trim(samples)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    return (whisper.decode(model, mel, opts).text or "").strip()


def _clean_phrase(text: str) -> str:
    """Take the longest comma/period-separated chunk. Strip bracket noise."""
    text = re.sub(r"[\(\)\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in re.split(r"[.,!?;]", text) if p.strip()]
    if not parts:
        return text
    return max(parts, key=len)[:120]


# ── Fuzzy multi-service lookup ─────────────────────────────────────────────


def _search_itunes(phrase: str, limit: int = 5) -> list[dict]:
    """iTunes Search API — free, no auth, fuzzy text match."""
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": phrase, "entity": "song", "limit": str(limit)},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("iTunes search failed: %s", e)
        return []
    out: list[dict] = []
    for r in data.get("results", []):
        preview = r.get("previewUrl")
        if not preview:
            continue
        out.append({
            "source": "reactive:itunes",
            "artist": r.get("artistName", ""),
            "title": r.get("trackName", ""),
            "preview_url": preview,
        })
    return out


def _search_deezer(phrase: str, limit: int = 5) -> list[dict]:
    """Deezer Search API — free, no auth, fuzzy text match."""
    try:
        r = requests.get(
            "https://api.deezer.com/search",
            params={"q": phrase, "limit": str(limit)},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Deezer search failed: %s", e)
        return []
    out: list[dict] = []
    for t in data.get("data", []):
        preview = t.get("preview")
        if not preview:
            continue
        out.append({
            "source": "reactive:deezer",
            "artist": (t.get("artist") or {}).get("name", ""),
            "title": t.get("title", ""),
            "preview_url": preview,
        })
    return out


def _fingerprint_url(url: str) -> np.ndarray | None:
    """Download an MP3/AAC preview, fingerprint via fpcalc -raw."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception:
        return None
    suffix = ".m4a" if url.endswith(".m4a") else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(r.content)
        path = Path(f.name)
    try:
        fp = fp_mod.fingerprint_file(path, raw=True)
        return fp.raw_hashes
    except Exception:
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def identify_reactive(
    samples: np.ndarray,
    sr: int,
    *,
    n_candidates: int = 4,
) -> ReactiveMatch | None:
    """Full chain: transcribe → search → download → fingerprint-verify."""
    if not _whisper_available():
        log.warning("openai-whisper not installed — reactive lookup unavailable")
        return None

    try:
        text = transcribe(samples, sr)
    except Exception as e:
        log.warning("transcription failed: %s", e)
        return None
    phrase = _clean_phrase(text)
    if len(phrase) < 8:
        return None  # not enough lyric to search on

    # Fingerprint the query once (raw uint32 hashes)
    query_fp = fp_mod.fingerprint_sample(
        samples, sr,
        pitch_shifts_pct=config.PITCH_SWEEP_PERCENT,
        include_b64=False,
    )

    # Fan out search
    candidates = _search_itunes(phrase, limit=n_candidates) + _search_deezer(phrase, limit=n_candidates)

    best: ReactiveMatch | None = None
    for cand in candidates:
        preview_hashes = _fingerprint_url(cand["preview_url"])
        if preview_hashes is None or len(preview_hashes) == 0:
            continue
        # Pick the best score across our 7 pitch variants
        best_score = 0.0
        for variant in query_fp.fingerprints:
            if variant.raw_hashes is None:
                continue
            score, _ = _score_query_vs_library(variant.raw_hashes, preview_hashes)
            if score > best_score:
                best_score = score
        if best_score < _MIN_MATCH_SCORE:
            continue
        if best is None or best_score > best.score:
            best = ReactiveMatch(
                score=best_score,
                artist=cand["artist"],
                title=cand["title"],
                source=cand["source"],
                preview_url=cand["preview_url"],
                transcript=phrase,
            )

    return best
