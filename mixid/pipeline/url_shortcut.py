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


# Untimed numbered/bulleted lists. Some DJs write tracklists without
# timestamps. We require a 'tracklist' keyword in the preceding text
# AND at least 5 matching lines to suppress false positives like
# "follow me at: 1) twitter 2) instagram ...".
_UNTIMED_LINE = re.compile(
    r"""
    ^\s*
    (?:\d+\s*[.\)]\s*|[-•*]\s+)            # leading "1.", "1)", "-", "•", "*"
    (?P<rest>(?:[^-–—|]+?\s+[-–—|]\s+.+))  # must contain a separator
    \s*$
    """,
    re.VERBOSE,
)

_TRACKLIST_KEYWORD = re.compile(r"\b(tracklist|track\s*list|songs?\s*played|tracklisting)\b", re.I)

_MIN_UNTIMED_LINES = 5


def parse_untimed_tracklist(text: str, source: str) -> list[TracklistEntry]:
    """Parse a numbered/bulleted artist-title list with no timestamps.

    Returns entries with `start_sec=None`. Heuristic guard:
      - at least one 'tracklist' / 'songs played' keyword in the text
      - at least _MIN_UNTIMED_LINES matching lines

    Both guards exist to avoid mis-parsing 'follow me at: 1) twitter
    2) instagram' style social-link lists.
    """
    if not _TRACKLIST_KEYWORD.search(text):
        return []
    raw_matches: list[tuple[str, str]] = []  # (artist, title)
    for line in text.splitlines():
        if _TS_LINE.match(line):
            # If a line is already timed, parse_timestamped_tracklist owns it.
            continue
        m = _UNTIMED_LINE.match(line)
        if not m:
            continue
        artist, title = _split_artist_title(m.group("rest"))
        if not title:
            continue
        raw_matches.append((artist, title))
    if len(raw_matches) < _MIN_UNTIMED_LINES:
        return []
    return [
        TracklistEntry(start_sec=None, artist=a, title=t, source=source)
        for a, t in raw_matches
    ]


# Chapter titles that aren't tracks — skip these when reading native chapters.
_GENERIC_CHAPTER = re.compile(
    r"^\s*(intro|outro|id|id\s*[-–—]\s*id|outro\s*/\s*id|tracklist|track\s*list|"
    r"interlude|skit|mix|start|end|warm[\s-]*up|drop|break|transition|"
    r"thanks?( for watching)?|subscribe|like\s*&?\s*subscribe)\s*$",
    re.I,
)
_LEADING_TRACK_NUM = re.compile(r"^\s*\d{1,3}\s*[.\)\-–—]\s*")


def parse_chapters(meta: dict, source: str) -> list[TracklistEntry]:
    """Parse yt-dlp's `chapters` array into timed tracklist entries.

    yt-dlp populates `chapters` from a video's native chapter markers.
    YouTube auto-promotes "0:00 Title" description lines into native
    chapters when there are 3+ (first at 0:00), so many DJ-mix uploads
    expose a free, already-timed, structured tracklist here — more
    reliable than regex-parsing the free-text description.

    Each chapter is {start_time, end_time, title}. Generic non-track
    titles (Intro, Outro, ID, "thanks for watching", etc.) are skipped.
    """
    chapters = meta.get("chapters") or []
    if not isinstance(chapters, list):
        return []
    entries: list[TracklistEntry] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        title_raw = (ch.get("title") or "").strip()
        start = ch.get("start_time")
        if not title_raw or start is None:
            continue
        if _GENERIC_CHAPTER.match(title_raw):
            continue
        # Drop a leading track number like "01." / "1)" / "1 -" before splitting.
        title_raw = _LEADING_TRACK_NUM.sub("", title_raw)
        artist, title = _split_artist_title(title_raw)
        if not title:
            continue
        entries.append(
            TracklistEntry(start_sec=float(start), artist=artist, title=title, source=source)
        )
    return entries


# ── yt-dlp metadata fetch (description, title) ──────────────────────────────

# Tiny in-process cache so the density gate in run.py and the
# orchestrator in this module don't both pay the yt-dlp subprocess cost
# (2-5s each) on the same URL. 5-minute TTL is enough for a single
# pipeline run; not meant to survive across runs.
import time as _time

_META_CACHE: dict[str, tuple[float, dict | None]] = {}
_META_TTL_SEC = 300.0


def fetch_metadata(url: str, ytdlp_exe: str = "yt-dlp") -> dict | None:
    """Fetch metadata for a URL via yt-dlp --dump-json. Returns parsed dict or None.

    Cached in-process for 5 min so duplicate calls within the same
    pipeline run reuse the same subprocess result.
    """
    hit = _META_CACHE.get(url)
    if hit and (_time.time() - hit[0]) <= _META_TTL_SEC:
        return hit[1]
    try:
        res = subprocess.run(
            [ytdlp_exe, "--skip-download", "--dump-single-json", url],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        log.warning("yt-dlp not found at %r — install or set YTDLP_EXE", ytdlp_exe)
        _META_CACHE[url] = (_time.time(), None)
        return None
    if res.returncode != 0:
        log.warning("yt-dlp failed for %s: %s", url, res.stderr[:200])
        _META_CACHE[url] = (_time.time(), None)
        return None
    try:
        parsed = json.loads(res.stdout)
    except json.JSONDecodeError:
        parsed = None
    _META_CACHE[url] = (_time.time(), parsed)
    return parsed


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


# Source priority. Higher = wins on overlap inside the dedup window.
# Order: community-curated > user-written > untimed (lowest confidence).
SOURCE_PRIORITY: dict[str, int] = {
    "1001tracklists": 4,
    "mixesdb": 3,
    "youtube_chapters": 2,
    "soundcloud_chapters": 2,
    "mixcloud_chapters": 2,
    "youtube_description": 2,
    "soundcloud_description": 2,
    "mixcloud_description": 2,
    "youtube_description_untimed": 1,
    "soundcloud_description_untimed": 1,
    "mixcloud_description_untimed": 1,
}


def _normalize_for_dedup(s: str) -> str:
    """Lowercase, collapse whitespace, strip wrapping punctuation."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _merge_and_dedup(
    entries: list[TracklistEntry], window_sec: float = 30.0
) -> list[TracklistEntry]:
    """Merge across sources. Timed entries dedup by position; untimed by name.

    For two timed entries within `window_sec`:
      - same (normalized artist+title) → keep higher-priority source
      - different (artist+title)       → keep higher-priority source (community
                                         > user-written), discard the other
    Untimed entries are added only if their normalized (artist, title) is not
    already represented by a kept entry.
    """
    timed = [e for e in entries if e.start_sec is not None]
    untimed = [e for e in entries if e.start_sec is None]
    # Sort by start_sec, then by priority descending so the highest-priority
    # entry at a given timestamp shows up first in any window.
    timed.sort(key=lambda e: (e.start_sec, -SOURCE_PRIORITY.get(e.source, 0)))

    kept: list[TracklistEntry] = []
    for e in timed:
        if kept and abs(kept[-1].start_sec - e.start_sec) < window_sec:
            prev = kept[-1]
            if SOURCE_PRIORITY.get(e.source, 0) > SOURCE_PRIORITY.get(prev.source, 0):
                kept[-1] = e
            # else: drop e (prev had higher or equal priority)
            continue
        kept.append(e)

    seen_names = {
        (_normalize_for_dedup(e.artist), _normalize_for_dedup(e.title)) for e in kept
    }
    for e in untimed:
        key = (_normalize_for_dedup(e.artist), _normalize_for_dedup(e.title))
        if key in seen_names:
            continue
        kept.append(e)
        seen_names.add(key)
    return kept


def try_url_shortcut(url: str, ytdlp_exe: str = "yt-dlp") -> list[TracklistEntry]:
    """Aggregate tracklist entries from every available source.

    Sources tried:
      - YouTube/SoundCloud/Mixcloud description (timed)
      - Same description (untimed numbered/bulleted list) — fallback
      - mixesdb.com search + wikitext
      - 1001tracklists.com search + page

    Returns the merged + deduped list, sorted by start_sec
    (entries with start_sec=None sort last).
    """
    src = detect_source(url)
    if src is None:
        return []

    all_entries: list[TracklistEntry] = []
    inferred_title = ""

    if src in ("youtube", "soundcloud", "mixcloud"):
        meta = fetch_metadata(url, ytdlp_exe=ytdlp_exe)
        if meta:
            inferred_title = (meta.get("title") or "").strip()
            desc = meta.get("description") or ""
            # Collect BOTH native chapters and the description timestamps,
            # then let _merge_and_dedup arbitrate. On YouTube the two are
            # usually the same data (the 30s-window dedup collapses the
            # overlap, chapters win the tie on equal priority via insertion
            # order). But when they DIVERGE — a mix with 3 chapter pins but a
            # 25-line description tracklist, common on long Afrobeats sets —
            # this keeps every description-only track instead of letting one
            # surviving chapter suppress the whole list.
            chapter_entries = parse_chapters(meta, source=f"{src}_chapters")
            all_entries.extend(chapter_entries)
            description_entries = parse_timestamped_tracklist(
                desc, source=f"{src}_description"
            )
            all_entries.extend(description_entries)
            # Untimed parser only when NO timed source produced anything —
            # i.e. the description lists tracks but without timestamps.
            if not chapter_entries and not description_entries:
                all_entries.extend(
                    parse_untimed_tracklist(desc, source=f"{src}_description_untimed")
                )

    # mixesdb: community-curated, free, no auth
    search_term = inferred_title if inferred_title else url
    hits = search_mixesdb(search_term)
    if hits:
        wikitext = fetch_mixesdb_page(hits[0]["title"])
        if wikitext:
            all_entries.extend(parse_mixesdb_wikitext(wikitext))

    # 1001tracklists: community-curated, Cloudflare-protected, may return []
    if inferred_title:
        try:
            # Lazy import to avoid a hard dep cycle and to keep this module
            # importable without bs4 for the unit tests that don't need it.
            from mixid.pipeline import tracklists1001
            all_entries.extend(
                tracklists1001.try_1001tracklists(url, inferred_title)
            )
        except ImportError:
            log.debug("tracklists1001 module unavailable; skipping")

    return _merge_and_dedup(all_entries)
