# MixID Dockerfile for Hugging Face Spaces (or any container host).
#
# Notes:
# - Based on slim python; explicitly skips Demucs/torch heavy GPU deps
#   because HF Spaces free tier is CPU-only and Demucs is too slow there.
# - The Shazam path (shazamio + AcoustID + iTunes/Deezer reactive + LLM
#   re-rank) fits comfortably in 2 GB RAM.

FROM python:3.11-slim

# System deps: ffmpeg for librosa/shazamio, libchromaprint-tools for fpcalc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libchromaprint-tools \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# fpcalc is at /usr/bin/fpcalc after libchromaprint-tools install
ENV MIXID_FPCALC_PATH=/usr/bin/fpcalc

WORKDIR /app

# Copy only what's needed for installs first (better Docker cache)
COPY pyproject.toml /app/

# Install Python deps WITHOUT [enrich] extras — Demucs/Whisper-base/CLAP/
# FAISS are deliberately excluded from cloud deploy to keep memory in check.
# We DO install yt-dlp + FastAPI + Shazam + AcoustID + reactive-lookup
# components.
RUN pip install --no-cache-dir \
    pyacoustid \
    librosa \
    pyloudnorm \
    soundfile \
    requests \
    mwclient \
    python-dotenv \
    numpy \
    scipy \
    shazamio \
    spotipy \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    sse-starlette \
    openai-whisper

# yt-dlp: install the NIGHTLY channel, not the stable PyPI release. YouTube
# changes its internal API constantly and the nightly tracks those fixes
# weeks ahead of stable — important because this runs from a datacenter IP
# that YouTube actively throttles. Rebuild the Space to pull a fresh nightly
# when YouTube downloads start failing again.
RUN pip install --no-cache-dir -U --pre "yt-dlp[default]"

# Now the application code
COPY mixid /app/mixid
COPY config.py /app/config.py

# Hugging Face Spaces routes traffic to $PORT (default 7860)
ENV MIXID_DISABLE_DEMUCS=1
EXPOSE 7860

# Use uvicorn directly (faster startup than `python -m mixid --serve`)
CMD ["uvicorn", "mixid.web.server:app", "--host", "0.0.0.0", "--port", "7860"]
