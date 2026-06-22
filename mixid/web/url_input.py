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


class PlatformBlockedError(RuntimeError):
    """The host (usually YouTube) refused our server's request.

    Almost always means YouTube is blocking the datacenter IP the app
    runs on — a free-cloud-host problem, not a bug. The caller should
    steer the user to SoundCloud/Mixcloud/Audiomack or a file upload.
    """

    def __init__(self, platform: str, raw: str = ""):
        self.platform = platform
        self.raw = raw
        super().__init__(f"{platform} blocked the request")


# Signatures that mean "the platform refused us / blocked the IP" rather
# than "the URL is bad". YouTube drops datacenter IPs with an SSL EOF or
# a bot-check; these strings show up in yt-dlp's stderr when it happens.
_BLOCK_SIGNS = (
    "unexpected_eof_while_reading",
    "unable to download api page",
    "sign in to confirm",
    "confirm you're not a bot",
    "http error 403",
    "failed to extract any player response",
    "not available on this app",
    "the read operation timed out",
)

# YouTube player clients to try, in order. Different InnerTube clients
# get blocked differently from datacenter IPs; rotating through a few
# recovers downloads that the default 'web' client can't get. This is a
# moving target — YouTube tightens it over time — so it's best-effort.
_YT_PLAYER_CLIENTS = "tv,mweb,web_safari,android"


def is_supported_url(url: str) -> bool:
    return any(h in url.lower() for h in SUPPORTED_HOSTS)


def _is_youtube(url: str) -> bool:
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


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
    ]
    # Rotate YouTube player clients — recovers downloads the default 'web'
    # client can't get from a blocked datacenter IP. Ignored by other
    # extractors, but we only add it for YouTube to keep things clean.
    if _is_youtube(url):
        cmd += ["--extractor-args", f"youtube:player_client={_YT_PLAYER_CLIENTS}"]
    cmd.append(url)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError("yt-dlp not found. Install with `pip install yt-dlp`.") from e
    if res.returncode != 0:
        # yt-dlp can succeed at download but exit nonzero on warnings; check for stdout JSON
        if not res.stdout.strip():
            stderr = res.stderr or ""
            if any(sign in stderr.lower() for sign in _BLOCK_SIGNS):
                platform = "YouTube" if _is_youtube(url) else _extract_host(url)
                raise PlatformBlockedError(platform, raw=stderr[:300])
            raise RuntimeError(f"yt-dlp failed: {stderr[:500]}")
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
