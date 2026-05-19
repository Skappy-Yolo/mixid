"""Download audio from a public URL (YouTube / SoundCloud / Mixcloud / Audiomack).

The V1 `pipeline.url_shortcut` module only scrapes tracklists from
descriptions and mixesdb. For the public web app the user pastes a URL
expecting the audio itself to be processed — so this module is the
audio-download bridge.

Uses yt-dlp under the hood (the standard open-source extractor). We
intentionally keep this module thin: download, return a Path, let the
caller route it through the normal `pipeline.run` flow. yt-dlp is
mass-tolerated for personal-use downloads but TOS-gray on some
platforms; we deliberately don't headline the YouTube path in
marketing.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


SUPPORTED_HOSTS = (
    "youtube.com", "youtu.be", "music.youtube.com",
    "soundcloud.com", "mixcloud.com",
    "audiomack.com",
)


@dataclass
class DownloadedAudio:
    path: Path                 # local file path (m4a/mp3/webm — caller passes to ffmpeg-aware loader)
    title: str
    duration_secs: float | None
    source_url: str
    source_host: str


def is_supported_url(url: str) -> bool:
    return any(h in url.lower() for h in SUPPORTED_HOSTS)


def _ytdlp_path() -> str:
    """Find a working yt-dlp. Prefer the venv's, then PATH, then $YTDLP_EXE."""
    candidates = [
        shutil.which("yt-dlp"),
        shutil.which("yt-dlp.exe"),
        r"C:\ytmusic\yt-dlp.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    # fall back to module entry — works if yt-dlp was pip-installed
    return "yt-dlp"


def download(url: str, output_dir: Path | str | None = None) -> DownloadedAudio:
    """Download the audio of a public URL. Returns a DownloadedAudio.

    Raises ValueError if URL host isn't supported.
    Raises RuntimeError if yt-dlp fails.
    """
    if not is_supported_url(url):
        raise ValueError(
            f"Unsupported URL host. Supported: {', '.join(SUPPORTED_HOSTS)}"
        )
    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mixid_dl_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # %(id)s avoids filename collisions; ext from postprocessor (always .m4a)
    template = str(output_dir / "%(id)s.%(ext)s")
    cmd = [
        _ytdlp_path(),
        "--no-progress",
        "--no-warnings",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "m4a",
        "--audio-quality", "0",   # best
        "--print-json",
        "-o", template,
        url,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError("yt-dlp not found. Install with `pip install yt-dlp`.") from e
    if res.returncode != 0:
        # yt-dlp can succeed at download but exit nonzero on warnings; check for stdout JSON
        if not res.stdout.strip():
            raise RuntimeError(f"yt-dlp failed: {res.stderr[:500]}")
    # Last JSON line on stdout has metadata
    import json as _json
    meta = None
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                meta = _json.loads(line)
                break
            except _json.JSONDecodeError:
                continue
    if not meta:
        raise RuntimeError(f"yt-dlp produced no metadata. stderr: {res.stderr[:300]}")

    # Resolve the downloaded file path. yt-dlp's metadata can carry it directly
    # via 'filepath' (newer versions) or we can compute it from id + ext.
    downloaded_path: Path | None = None
    if meta.get("filepath"):
        downloaded_path = Path(meta["filepath"])
    if downloaded_path is None or not downloaded_path.exists():
        # Fallback: find the freshest m4a in output_dir
        candidates = sorted(output_dir.glob("*.m4a"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"No .m4a found in {output_dir} after yt-dlp")
        downloaded_path = candidates[0]

    host = _extract_host(url)
    return DownloadedAudio(
        path=downloaded_path,
        title=str(meta.get("title", "") or ""),
        duration_secs=float(meta["duration"]) if meta.get("duration") else None,
        source_url=url,
        source_host=host,
    )


def _extract_host(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/", url)
    return (m.group(1).lower() if m else "").replace("www.", "")
