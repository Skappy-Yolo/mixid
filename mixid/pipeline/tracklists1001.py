"""1001tracklists.com scraper.

1001tracklists is the densest community-curated database of DJ-set
tracklists on the web. Many popular mixes are tracklisted there with
cue times.

Status (2026-05-20): 1001tracklists serves a stripped DOM to non-browser
HTTP clients — even with a realistic Chrome User-Agent, the body of a
known tracklist page does NOT contain the artist/title rows. They
hydrate the page client-side after a JS check.

In other words: plain `requests` scraping returns empty results in
practice today. This module remains in place for three reasons:
  1. It returns `[]` cleanly on the stripped-DOM case, so the rest of
     the pipeline (mixesdb, descriptions, untimed parser) is unaffected.
  2. If 1001tracklists ever relaxes their bot detection, or someone
     adds a Playwright/Selenium adapter inside `_polite_get`, this
     module starts delivering data without further plumbing.
  3. Even the empty hit is cached, so we don't hammer their site.

Default is OFF (the network call is always-empty today, so saving the
~2s subprocess cost on every URL run is the right default). To opt in
once 1001tracklists relaxes their bot detection (or once you've added
a Playwright adapter), set `MIXID_ENABLE_1001TRACKLISTS=1`.

Public surface:
    search_1001tracklists(query, max_results=5) -> list[dict]
    fetch_1001tracklist(url) -> list[TracklistEntry]
    try_1001tracklists(source_url, source_title) -> list[TracklistEntry]
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from mixid.pipeline.url_shortcut import TracklistEntry, _parse_timestamp, _split_artist_title

log = logging.getLogger(__name__)


_BASE = "https://www.1001tracklists.com"
_SEARCH_URL = _BASE + "/search/result.html"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TIMEOUT = 15.0
_MIN_INTERVAL = 1.5  # seconds between any two HTTP calls to the site

_CACHE_TTL_SEC = 7 * 24 * 3600  # 7 days
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_DB = _CACHE_DIR / "tracklists1001.sqlite"

_last_call_ts = 0.0
_call_lock = threading.Lock()


def _is_disabled() -> bool:
    """Default OFF. Opt in via MIXID_ENABLE_1001TRACKLISTS=1."""
    return os.environ.get("MIXID_ENABLE_1001TRACKLISTS", "").strip() not in ("1", "true", "yes")


# ── cache ─────────────────────────────────────────────────────────────


def _cache_conn() -> sqlite3.Connection:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS search_cache "
        "(query TEXT PRIMARY KEY, ts INTEGER, json TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tracklist_cache "
        "(url TEXT PRIMARY KEY, ts INTEGER, json TEXT)"
    )
    return conn


def _cache_get(table: str, key: str) -> list | None:
    try:
        with _cache_conn() as conn:
            row = conn.execute(f"SELECT ts, json FROM {table} WHERE {'query' if table == 'search_cache' else 'url'} = ?", (key,)).fetchone()
    except sqlite3.Error as e:
        log.debug("cache get failed: %s", e)
        return None
    if not row:
        return None
    ts, payload = row
    if time.time() - ts > _CACHE_TTL_SEC:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _cache_put(table: str, key: str, value: list) -> None:
    col = "query" if table == "search_cache" else "url"
    try:
        with _cache_conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} ({col}, ts, json) VALUES (?, ?, ?)",
                (key, int(time.time()), json.dumps(value)),
            )
    except sqlite3.Error as e:
        log.debug("cache put failed: %s", e)


# ── HTTP ──────────────────────────────────────────────────────────────


def _polite_get(url: str, params: dict | None = None) -> str | None:
    """GET with rate-limit + Cloudflare-aware fallback. Returns HTML text or None."""
    global _last_call_ts
    with _call_lock:
        elapsed = time.time() - _last_call_ts
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_ts = time.time()
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        log.debug("1001tracklists request failed: %s", e)
        return None
    if r.status_code != 200:
        log.debug("1001tracklists %s returned %d", url, r.status_code)
        return None
    # Cloudflare challenge bodies contain a known marker
    if "Just a moment" in r.text[:5000] or "cf-browser-verification" in r.text[:5000]:
        log.debug("1001tracklists returned Cloudflare challenge for %s", url)
        return None
    return r.text


# ── search ────────────────────────────────────────────────────────────


def search_1001tracklists(query: str, max_results: int = 5) -> list[dict]:
    """Search 1001tracklists for a mix by free-text query.

    Returns up to `max_results` items: [{'url': ..., 'title': ...}].
    Cached for 7 days. Returns [] if disabled via env var.
    """
    if _is_disabled():
        return []
    query = (query or "").strip()
    if not query:
        return []
    cached = _cache_get("search_cache", query)
    if cached is not None:
        return cached[:max_results]
    html = _polite_get(_SEARCH_URL, params={"main_search": query, "search_selection": "ALL"})
    if not html:
        return []
    out: list[dict] = []
    soup = BeautifulSoup(html, "html.parser")
    # Result rows on the search page link to tracklist pages via <a href="/tracklist/...">
    for a in soup.select('a[href^="/tracklist/"]'):
        href = a.get("href") or ""
        text = " ".join(a.get_text(" ", strip=True).split())
        if not href or not text:
            continue
        full = _BASE + href if href.startswith("/") else href
        if any(item["url"] == full for item in out):
            continue
        out.append({"url": full, "title": text})
        if len(out) >= max_results:
            break
    _cache_put("search_cache", query, out)
    return out


# ── tracklist page parse ──────────────────────────────────────────────


_DURATION_RE = re.compile(r"^\d{1,2}:\d{1,2}(?::\d{2})?$")


def fetch_1001tracklist(url: str) -> list[TracklistEntry]:
    """Fetch and parse a 1001tracklists tracklist page.

    Returns entries with start_sec when a cue time is present, else
    start_sec=None. Cached for 7 days.

    Empty on Cloudflare challenge, page missing, or structure change.
    """
    if _is_disabled():
        return []
    if not url:
        return []
    cached = _cache_get("tracklist_cache", url)
    if cached is not None:
        return [TracklistEntry(**e) for e in cached]
    html = _polite_get(url)
    if not html:
        return []
    entries = _parse_tracklist_html(html)
    if entries:
        _cache_put(
            "tracklist_cache",
            url,
            [{"start_sec": e.start_sec, "artist": e.artist, "title": e.title, "source": e.source} for e in entries],
        )
    return entries


def _parse_tracklist_html(html: str) -> list[TracklistEntry]:
    """1001tracklists page parser. Tolerates layout changes by falling back."""
    soup = BeautifulSoup(html, "html.parser")
    entries: list[TracklistEntry] = []

    # Each track row has id="tlp_<n>" (stable across recent layouts)
    rows = soup.select('div[id^="tlp_"]')
    if not rows:
        # Layout changed; return empty so caller falls through
        log.debug("1001tracklists: no tlp_* rows found, layout may have changed")
        return []

    for row in rows:
        # Cue time: may be in a span/td with class containing 'cue' or 'time'
        cue_text = ""
        for sel in (".cueValueField", ".cueValue", ".cue", ".trackCueValue"):
            el = row.select_one(sel)
            if el:
                cue_text = el.get_text(" ", strip=True)
                break
        start_sec: float | None = None
        if cue_text and _DURATION_RE.match(cue_text):
            try:
                start_sec = _parse_timestamp(cue_text)
            except (ValueError, AttributeError):
                start_sec = None

        # Track value: "Artist - Title" (sometimes "Artist - Title (Remix)")
        track_text = ""
        for sel in (".trackFormat", ".trackValue", ".tlToglerOff", ".meta-music"):
            el = row.select_one(sel)
            if el:
                track_text = el.get_text(" ", strip=True)
                # Strip nested label/remix annotations that are siblings
                if track_text:
                    break

        if not track_text:
            # Fallback: take the row's full text, strip cue if present at start
            track_text = row.get_text(" ", strip=True)
            if cue_text and track_text.startswith(cue_text):
                track_text = track_text[len(cue_text):].strip()

        track_text = " ".join(track_text.split())
        if not track_text:
            continue

        artist, title = _split_artist_title(track_text)
        if not (artist or title):
            continue
        entries.append(
            TracklistEntry(
                start_sec=start_sec,
                artist=artist,
                title=title or track_text,
                source="1001tracklists",
            )
        )

    # Deduplicate exact (start_sec, artist, title) repeats that occasionally
    # appear when the page has both a primary and an "ID'd by community" row
    seen: set[tuple] = set()
    dedup: list[TracklistEntry] = []
    for e in entries:
        key = (e.start_sec, e.artist.lower().strip(), e.title.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup


# ── orchestrator ──────────────────────────────────────────────────────


def try_1001tracklists(source_url: str, source_title: str | None = None) -> list[TracklistEntry]:
    """Search for `source_title` on 1001tracklists, parse the top hit's page.

    `source_url` is currently unused for matching (search-by-URL needs a
    paid API key). Kept in the signature for forward compatibility.

    Returns [] on any failure path. Never raises.
    """
    if _is_disabled():
        return []
    if not source_title:
        return []
    try:
        hits = search_1001tracklists(source_title, max_results=3)
    except Exception as e:
        log.debug("1001tracklists search exception for %r: %s", source_title, e)
        return []
    if not hits:
        return []
    try:
        return fetch_1001tracklist(hits[0]["url"])
    except Exception as e:
        log.debug("1001tracklists fetch exception for %s: %s", hits[0]["url"], e)
        return []
