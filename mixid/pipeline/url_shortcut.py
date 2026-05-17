"""URL shortcut: skip the audio pipeline when a tracklist already exists.

Many DJ mixes uploaded to YouTube, SoundCloud, or Mixcloud have a
tracklist in the description, and mixesdb.com is a crowd-curated wiki
of mix tracklists. Before spending compute and API budget on audio
fingerprinting, ask:

1. Is the input a URL? → fetch the description via yt-dlp; parse any
   timestamped tracklist.
2. Either way, search mixesdb.com for the mix; if found, parse its
   wikitext tracklist.

Returns a list of TracklistEntry with optional timestamps, or None if
no usable tracklist could be located. The audio pipeline runs as the
fallback.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Literal

import requests


log = logging.getLogger(__name__)


@dataclass
class TracklistEntry:
    start_sec: float | None
    artist: str
    title: str
    source: str  # "youtube_description" | "mixesdb" | "soundcloud_description" | ...


SourceKind = Literal["youtube", "soundcloud", "mixcloud", "mixesdb", None]


_URL_PATTERNS: dict[str, re.Pattern] = {
    "youtube": re.compile(r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/"),
    "soundcloud": re.compile(r"^(https?://)?(www\.)?soundcloud\.com/"),
    "mixcloud": re.compile(r"^(https?://)?(www\.)?mixcloud\.com/"),
    "mixesdb": re.compile(r"^(https?://)?(www\.)?mixesdb\.com/"),
}


def detect_source(url: str) -> SourceKind:
    """Return the platform a URL belongs to, or None if not recognized."""
    if not url:
        return None
    url = url.strip()
    for kind, pat in _URL_PATTERNS.items():
        if pat.search(url):
            return kind  # type: ignore[return-value]
    return None


# ── Description / timestamp tracklist parsing ───────────────────────────────

# Match common YouTube tracklist line formats:
#   00:00 Artist - Title
#   0:00 - Artist - Title
#   [00:00:00] Artist – Title
#   1. 00:00 Artist – Title
_TS_LINE = re.compile(
    r"""
    ^\s*
    (?:\d+\s*[.\)]\s*)?                # optional leading "1." or "1)"
    \[?
    (?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2}) # 00:00 or 00:00:00
    \]?
    \s*[-–—]?\s*
    (?P<rest>.+?)
    \s*$
    """,
    re.VERBOSE,
)


def _parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0.0


def _strip_wrapping_brackets(s: str) -> str:
    """Remove a single pair of wrapping [] or () that decorate the whole field."""
    s = s.strip()
    if (s.startswith("[") and s.endswith("]")) or (
        s.startswith("(") and s.endswith(")")
    ):
        return s[1:-1].strip()
    return s


def _split_artist_title(rest: str) -> tuple[str, str]:
    """Split 'Artist - Title' on the FIRST dash-like separator. Strip wrapping brackets."""
    for sep in (" - ", " – ", " — ", " | "):
        if sep in rest:
            artist, title = rest.split(sep, 1)
            return _strip_wrapping_brackets(artist), _strip_wrapping_brackets(title)
    return "", _strip_wrapping_brackets(rest)


def parse_timestamped_tracklist(text: str, source: str) -> list[TracklistEntry]:
    """Parse a YouTube/SoundCloud description block. Returns ordered entries.

    Lines without a timestamp are ignored. Lines with a timestamp but
    no detectable artist/title separator land in `title` with empty artist —
    the LLM re-ranker can disambiguate downstream.
    """
    entries: list[TracklistEntry] = []
    for line in text.splitlines():
        m = _TS_LINE.match(line)
        if not m:
            continue
        start = _parse_timestamp(m.group("ts"))
        artist, title = _split_artist_title(m.group("rest"))
        if not title:
            continue
        entries.append(
            TracklistEntry(start_sec=start, artist=artist, title=title, source=source)
        )
    # Sort by timestamp ascending; many descriptions are written in order but
    # some intersperse non-track lines.
    entries.sort(key=lambda e: (e.start_sec or 0.0))
    return entries


# ── yt-dlp metadata fetch (description, title) ──────────────────────────────


def fetch_metadata(url: str, ytdlp_exe: str = "yt-dlp") -> dict | None:
    """Fetch metadata for a URL via yt-dlp --dump-json. Returns parsed dict or None."""
    try:
        res = subprocess.run(
            [ytdlp_exe, "--skip-download", "--dump-single-json", url],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        log.warning("yt-dlp not found at %r — install or set YTDLP_EXE", ytdlp_exe)
        return None
    if res.returncode != 0:
        log.warning("yt-dlp failed for %s: %s", url, res.stderr[:200])
        return None
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return None


# ── mixesdb MediaWiki ───────────────────────────────────────────────────────

_MIXESDB_API = "https://www.mixesdb.com/w/api.php"

# Wiki tracklist lines look like:
#   # [[Artist]] - [[Title]]
#   #1 Artist - Title
_WIKI_TRACK = re.compile(
    r"""
    ^\#+\s*                                  # leading hash bullets
    (?:\d+\s*[.\)]?\s*)?                     # optional track number
    (?:\[\[)?(?P<artist>[^\]\|\-–—]+?)(?:\]\])?
    \s*[-–—]\s*
    (?:\[\[)?(?P<title>[^\]\|]+?)(?:\]\])?
    \s*$
    """,
    re.VERBOSE,
)


def parse_mixesdb_wikitext(wikitext: str) -> list[TracklistEntry]:
    """Parse the Tracklist section out of a mixesdb wiki page."""
    # Find the tracklist heading; trail until the next heading.
    lines = wikitext.splitlines()
    in_tracklist = False
    entries: list[TracklistEntry] = []
    for raw in lines:
        line = raw.strip()
        if re.match(r"^==\s*(tracklist|track ?list)\s*==", line, re.I):
            in_tracklist = True
            continue
        if in_tracklist and line.startswith("==") and line.endswith("=="):
            # next section starts; tracklist over
            break
        if not in_tracklist:
            continue
        m = _WIKI_TRACK.match(line)
        if not m:
            continue
        entries.append(
            TracklistEntry(
                start_sec=None,
                artist=m.group("artist").strip(),
                title=m.group("title").strip(),
                source="mixesdb",
            )
        )
    return entries


def search_mixesdb(query: str, limit: int = 5) -> list[dict]:
    """opensearch via MediaWiki API. Returns [{title, url}, ...]."""
    if not query.strip():
        return []
    try:
        r = requests.get(
            _MIXESDB_API,
            params={
                "action": "opensearch",
                "search": query,
                "limit": str(limit),
                "namespace": "0",
                "format": "json",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {"title": t, "url": u}
            for t, u in zip(data[1], data[3])
        ]
    except Exception as e:
        log.warning("mixesdb opensearch failed: %s", e)
        return []


def fetch_mixesdb_page(page_title: str) -> str | None:
    """Fetch the wikitext of a mixesdb page."""
    try:
        r = requests.get(
            _MIXESDB_API,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "wikitext",
                "format": "json",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["parse"]["wikitext"]["*"]
    except Exception as e:
        log.warning("mixesdb fetch failed for %s: %s", page_title, e)
        return None


# ── Orchestrator ────────────────────────────────────────────────────────────


def try_url_shortcut(url: str, ytdlp_exe: str = "yt-dlp") -> list[TracklistEntry]:
    """Try every shortcut path. Returns whatever was found, ordered by likelihood."""
    src = detect_source(url)
    if src is None:
        return []

    candidates: list[TracklistEntry] = []
    inferred_title = ""

    if src in ("youtube", "soundcloud", "mixcloud"):
        meta = fetch_metadata(url, ytdlp_exe=ytdlp_exe)
        if meta:
            inferred_title = (meta.get("title") or "").strip()
            desc = meta.get("description") or ""
            description_entries = parse_timestamped_tracklist(
                desc, source=f"{src}_description"
            )
            if description_entries:
                candidates.extend(description_entries)

    # Always try mixesdb regardless — community lists often cover descriptions.
    search_term = inferred_title if inferred_title else url
    hits = search_mixesdb(search_term)
    if hits:
        wikitext = fetch_mixesdb_page(hits[0]["title"])
        if wikitext:
            mixesdb_entries = parse_mixesdb_wikitext(wikitext)
            if mixesdb_entries and not candidates:
                # Only fall back to mixesdb if the description scrape produced nothing
                candidates = mixesdb_entries

    return candidates
