"""Find DJ-mix mirrors for Spotify / Apple Music URLs.

Spotify locks its audio and Apple Music has no public stream, so when a
user pastes one of those links we need to find the same mix on a
platform we can actually read: YouTube, SoundCloud, Mixcloud, Audiomack.

Public surface:
    - is_locked_platform(url) -> bool
    - find_mirrors(url) -> {"source": {...}, "candidates": [...]}

`find_mirrors` resolves the title/artist via og-tags (no API key needed),
then searches each supported platform. Candidates are scored by title
overlap and duration plausibility (DJ mixes are usually >5 min).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser

from mixid.web.url_input import _ytdlp_path

log = logging.getLogger(__name__)


LOCKED_HOSTS = ("open.spotify.com", "spotify.com", "music.apple.com")

# Mixes shorter than this are almost certainly not the right result.
MIN_MIX_DURATION_SEC = 300  # 5 min


@dataclass
class MirrorCandidate:
    platform: str          # 'youtube' | 'soundcloud' | 'mixcloud' | 'audiomack'
    url: str
    title: str
    duration_sec: float | None
    score: float


def is_locked_platform(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in LOCKED_HOSTS)


# ---------- og-tag resolution ----------

class _OGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        d = {k: (v or "") for k, v in attrs}
        prop = d.get("property") or d.get("name") or ""
        if prop.startswith("og:") or prop.startswith("music:") or prop == "twitter:title":
            self.tags[prop] = d.get("content", "")


def _fetch_og(url: str, timeout: float = 8.0) -> dict[str, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; MixID/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Some sites send several MB of HTML; cap to avoid surprises.
        body = resp.read(512 * 1024).decode("utf-8", errors="replace")
    p = _OGParser()
    p.feed(body)
    return p.tags


def resolve_title_artist(url: str) -> dict[str, str] | None:
    """Return {'title': ..., 'artist': ..., 'query': '<title> <artist>'} or None."""
    try:
        tags = _fetch_og(url)
    except Exception as e:
        log.warning("og fetch failed for %s: %s", url, e)
        return None
    title = tags.get("og:title") or tags.get("twitter:title") or ""
    desc = tags.get("og:description") or ""
    musician = tags.get("music:musician") or tags.get("music:musician:name") or ""

    # Spotify og:title is usually "Track Name", og:description is "Artist · Song · ..."
    # Apple Music og:title is usually "Song by Artist - Album" or "Playlist Title by Curator"
    artist = ""
    if musician:
        # music:musician is a URL on Spotify; extract the path tail.
        m = re.search(r"/artist/[^/]+/?[^?]*", musician)
        if not m:
            artist = musician.split("/")[-1].replace("-", " ").strip()
    if not artist and " by " in title.lower():
        # Apple: "Song Name by Artist Name"
        parts = re.split(r"\s+by\s+", title, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            title, artist = parts[0].strip(), parts[1].strip()
    if not artist and desc:
        # Spotify description often starts with "Artist · Song · ..."
        bits = [b.strip() for b in desc.split("·")]
        if len(bits) >= 2 and bits[0]:
            artist = bits[0]

    if not title:
        return None
    query = f"{title} {artist}".strip()
    return {"title": title, "artist": artist, "query": query}


# ---------- yt-dlp-based search ----------

def _ytdlp_search(prefix: str, query: str, n: int = 3, timeout: float = 25.0) -> list[dict]:
    """Call yt-dlp with a search prefix (ytsearchN: / scsearchN:) and return raw JSON dicts.

    Note: `--flat-playlist` strips duration on YouTube/SoundCloud, which kills
    our score's <5min penalty for non-mix results. We do the full per-entry
    fetch so the duration field is populated.
    """
    search_expr = f"{prefix}{n}:{query}"
    cmd = [
        _ytdlp_path(),
        "--no-warnings",
        "--no-progress",
        "--dump-json",
        "--skip-download",
        search_expr,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp search timed out: %s", search_expr)
        return []
    if not res.stdout.strip():
        return []
    out: list[dict] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _ytdlp_to_candidate(j: dict, platform: str) -> MirrorCandidate | None:
    url = j.get("url") or j.get("webpage_url") or ""
    if not url:
        return None
    # ytsearch returns 'url' as the video id, not a full URL. Normalize.
    if platform == "youtube" and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={url}"
    if platform == "soundcloud" and not url.startswith("http"):
        url = url  # scsearch already returns full URLs in modern yt-dlp
    title = j.get("title") or ""
    dur = j.get("duration")
    return MirrorCandidate(platform=platform, url=url, title=title, duration_sec=dur, score=0.0)


# ---------- Mixcloud public API ----------

def _search_mixcloud(query: str, n: int = 3, timeout: float = 8.0) -> list[MirrorCandidate]:
    q = urllib.parse.quote_plus(query)
    url = f"https://api.mixcloud.com/search/?q={q}&type=cloudcast&limit={n}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MixID/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("mixcloud search failed: %s", e)
        return []
    out: list[MirrorCandidate] = []
    for item in data.get("data", []):
        out.append(MirrorCandidate(
            platform="mixcloud",
            url=item.get("url") or "",
            title=item.get("name") or "",
            duration_sec=item.get("audio_length"),
            score=0.0,
        ))
    return out


# ---------- Audiomack scrape ----------

def _search_audiomack(query: str, n: int = 3, timeout: float = 8.0) -> list[MirrorCandidate]:
    """Audiomack has no open API. Scrape their public search page (best-effort)."""
    q = urllib.parse.quote_plus(query)
    url = f"https://audiomack.com/search/song?q={q}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MixID/1.0)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(512 * 1024).decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("audiomack search failed: %s", e)
        return []
    # Find Next.js JSON payload which embeds search results.
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    # Navigate: pageProps -> results.songs (shape varies; be defensive)
    page_props = data.get("props", {}).get("pageProps", {})
    songs = (
        page_props.get("results", {}).get("songs")
        or page_props.get("songs")
        or []
    )
    out: list[MirrorCandidate] = []
    for s in songs[:n]:
        url_path = s.get("url") or s.get("urlSlug") or ""
        if url_path and not url_path.startswith("http"):
            url_path = f"https://audiomack.com/{url_path.lstrip('/')}"
        title = s.get("title") or s.get("name") or ""
        artist = s.get("artist") or ""
        out.append(MirrorCandidate(
            platform="audiomack",
            url=url_path,
            title=f"{artist} - {title}".strip(" -"),
            duration_sec=s.get("duration"),
            score=0.0,
        ))
    return out


# ---------- scoring ----------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD_RE.findall((s or "").lower()))


def _score(query_tokens: set[str], cand: MirrorCandidate) -> float:
    cand_tokens = _tokens(cand.title)
    if not cand_tokens or not query_tokens:
        return 0.0
    overlap = len(query_tokens & cand_tokens)
    base = overlap / max(1, len(query_tokens))
    # Duration penalty: DJ mixes are usually 5+ min. Shorts get a soft demotion,
    # not a hard zero (a short track might still be useful if it's the source).
    if cand.duration_sec is not None and cand.duration_sec < MIN_MIX_DURATION_SEC:
        base *= 0.6
    return base


# ---------- composite ----------

def find_mirrors(url: str, n_per_platform: int = 3) -> dict:
    """Resolve a Spotify/Apple URL to a query, then search the four platforms.

    Returns:
        {
            "source": {"title": str, "artist": str},
            "candidates": [{"platform": ..., "url": ..., "title": ..., "duration_sec": ..., "score": ...}, ...]
        }
    Raises ValueError if URL isn't a locked-platform URL.
    Raises RuntimeError if og-tag resolution fails.
    """
    if not is_locked_platform(url):
        raise ValueError("Not a Spotify or Apple Music URL")
    resolved = resolve_title_artist(url)
    if not resolved:
        raise RuntimeError("Could not resolve title/artist from URL")

    query = resolved["query"]
    qt = _tokens(query)

    all_candidates: list[MirrorCandidate] = []
    # YouTube + SoundCloud via yt-dlp
    for prefix, platform in (("ytsearch", "youtube"), ("scsearch", "soundcloud")):
        raw = _ytdlp_search(prefix, query, n=n_per_platform)
        for j in raw:
            c = _ytdlp_to_candidate(j, platform)
            if c and c.url:
                all_candidates.append(c)
    # Mixcloud + Audiomack
    all_candidates.extend(_search_mixcloud(query, n=n_per_platform))
    all_candidates.extend(_search_audiomack(query, n=n_per_platform))

    for c in all_candidates:
        c.score = _score(qt, c)
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    return {
        "source": {"title": resolved["title"], "artist": resolved["artist"]},
        "candidates": [asdict(c) for c in all_candidates],
    }
