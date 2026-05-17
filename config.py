"""
MixID configuration. Loads .env from the repo root and resolves paths
that may be shared with the sibling DJAgent toolkit.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Repo + .env ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── Paths shared with the sibling DJAgent toolkit ───────────────────────────
# Defaults match DJAgent's config so MixID can read DJAgent's library.db
# read-only without requiring DJAgent to be installed.
YTMUSIC_DIR = Path(os.getenv("YTMUSIC_DIR", r"C:\ytmusic"))
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", str(YTMUSIC_DIR / "Music")))
FFMPEG_DIR = Path(os.getenv("FFMPEG_DIR", str(YTMUSIC_DIR)))
VDJ_DIR = Path(
    os.getenv(
        "VDJ_DIR",
        str(Path.home() / "Documents" / "VirtualDJ"),
    )
)
VDJ_DB_PATH = VDJ_DIR / "database.xml"

DJAGENT_LIBRARY_DB = Path(
    os.getenv(
        "DJAGENT_LIBRARY_DB",
        str(Path.home() / "OneDrive" / "Documents" / "DJAgent" / "data" / "library.db"),
    )
)

# ── MixID-private data ──────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("MIXID_DATA_DIR", str(BASE_DIR / "data")))
OUTPUTS_DIR = Path(os.getenv("MIXID_OUTPUTS_DIR", str(BASE_DIR / "outputs")))
BIN_DIR = BASE_DIR / "bin"

FINGERPRINTS_DB = DATA_DIR / "fingerprints.db"
EMBEDDINGS_INDEX = DATA_DIR / "embeddings.faiss"
FPCALC_EXE = BIN_DIR / ("fpcalc.exe" if os.name == "nt" else "fpcalc")

for _d in (DATA_DIR, OUTPUTS_DIR, BIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Tier-1 free APIs ────────────────────────────────────────────────────────
ACOUSTID_API_KEY = os.getenv("ACOUSTID_API_KEY", "")

# ── Spotify (free dev app, used to build a preview-fingerprint library) ────
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# ── Tier-2 free APIs (Colab/HF) ─────────────────────────────────────────────
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY", "")
ACRCLOUD_KEY = os.getenv("ACRCLOUD_KEY", "")
ACRCLOUD_SECRET = os.getenv("ACRCLOUD_SECRET", "")
ACRCLOUD_HOST = os.getenv("ACRCLOUD_HOST", "")

# ── LLM provider ────────────────────────────────────────────────────────────
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
AI_MODEL = os.getenv("AI_MODEL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Pipeline tuning (Tier 1) ────────────────────────────────────────────────
TARGET_SR = 22050                  # mono resample target for fingerprinting
LOUDNESS_TARGET_LUFS = -16.0       # ITU-R BS.1770 normalization target
HIGHPASS_CUTOFF_HZ = 80            # attenuate sub-rumble + room boom
SAMPLE_WINDOW_SECS = 12            # per-segment fingerprint window
MAX_SEGMENT_GAP_SECS = 90          # force a sample if no novelty hit in this long
PITCH_SWEEP_PERCENT = (-6, -4, -2, 0, 2, 4, 6)   # ±6% in 2% steps, 7 variants
MATCH_CONFIDENCE_FLOOR = 0.85      # local match score considered "confident"

# ── ffmpeg PATH injection (mirrors DJAgent pattern) ─────────────────────────
if FFMPEG_DIR.exists() and str(FFMPEG_DIR) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
