"""Lyrics-based identification fallback for unknown segments.

When neither the local library nor AcoustID nor ACRCloud identifies a
segment but it has audible vocals, lyrics search often does. Pipeline:

  audio sample → Whisper-tiny transcription → take the most lyric-like
  phrase → Genius search → best match → return artist+title

Why this matters for the Afrobeats and electronic edits in this user's
mixes: AcoustID coverage is sparse for niche/indie releases but lyrics
indexes (Genius) cover them well via UGC.

CPU vs GPU: Whisper-tiny runs ~3-5x real-time on CPU — fine for the
~5-10 unknown segments per mix. Bigger Whisper models live in Tier-2
on Colab.

Both `openai-whisper` and `GENIUS_API_KEY` are optional. If either is
missing, lyrics fallback is silently skipped and the pipeline runs
without it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import requests

import config


log = logging.getLogger(__name__)

_GENIUS_API = "https://api.genius.com/search"


@dataclass
class LyricsMatch:
    score: float       # transcription_confidence × genius_search_position_score
    title: str
    artist: str
    genius_id: int     # for later metadata enrichment if wanted
    transcript: str    # the lyric phrase we searched


def _whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


_WHISPER_MODEL_CACHE: dict = {}


def _load_whisper(model_name: str = "tiny"):
    if model_name in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[model_name]
    import whisper

    log.info("Loading whisper-%s (one-time, ~30 sec)…", model_name)
    model = whisper.load_model(model_name)
    _WHISPER_MODEL_CACHE[model_name] = model
    return model


def transcribe_sample(samples: np.ndarray, sr: int, model_name: str = "tiny") -> str:
    """Transcribe a numpy audio sample to text using Whisper. CPU-friendly tiny model."""
    import whisper

    model = _load_whisper(model_name)
    # Whisper expects 16kHz mono float32 in [-1, 1]. Resample if needed.
    if sr != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    samples = samples.astype(np.float32)
    # Pad or trim to 30 seconds — Whisper's expected input length
    audio = whisper.pad_or_trim(samples)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    result = whisper.decode(model, mel, options)
    return (result.text or "").strip()


def _clean_phrase(text: str) -> str:
    """Strip filler and short noise. Keep the meatiest lyric snippet."""
    # Lowercase, collapse whitespace, strip punctuation noise
    text = re.sub(r"[\(\)\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Take the longest comma/period-separated fragment — typically the actual lyric
    parts = [p.strip() for p in re.split(r"[.,!?;]", text) if p.strip()]
    if not parts:
        return text
    return max(parts, key=len)[:120]  # cap to ~120 chars for the search query


def search_genius(phrase: str, api_key: str = "") -> LyricsMatch | None:
    """Hit Genius search; return top hit or None."""
    api_key = api_key or config.GENIUS_API_KEY
    if not api_key:
        log.warning(
            "GENIUS_API_KEY not set — skipping lyrics search. "
            "Get a free key at https://genius.com/api-clients."
        )
        return None
    if not phrase:
        return None
    try:
        r = requests.get(
            _GENIUS_API,
            params={"q": phrase},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Genius search failed: %s", e)
        return None
    hits = (data.get("response") or {}).get("hits") or []
    if not hits:
        return None
    top = hits[0].get("result") or {}
    title = top.get("title", "") or ""
    artist = (top.get("primary_artist") or {}).get("name", "") or ""
    if not title and not artist:
        return None
    # Position score: top hit gets 1.0, 2nd 0.9, etc.
    return LyricsMatch(
        score=1.0,
        title=title,
        artist=artist,
        genius_id=int(top.get("id", 0) or 0),
        transcript=phrase,
    )


def identify_via_lyrics(
    samples: np.ndarray, sr: int, model_name: str = "tiny"
) -> LyricsMatch | None:
    """Full chain: transcribe → search Genius → return best match.

    Skipped silently if whisper or GENIUS_API_KEY missing.
    """
    if not _whisper_available():
        log.warning(
            "openai-whisper not installed — skipping lyrics fallback. "
            "Install with: pip install openai-whisper"
        )
        return None
    if not config.GENIUS_API_KEY:
        return None  # search_genius will warn

    try:
        text = transcribe_sample(samples, sr, model_name=model_name)
    except Exception as e:
        log.warning("Whisper transcription failed: %s", e)
        return None
    phrase = _clean_phrase(text)
    if len(phrase) < 10:
        return None  # too short to be a useful lyric query
    return search_genius(phrase)
