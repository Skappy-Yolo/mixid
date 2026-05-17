"""Offline tests for the URL shortcut parsers.

Live yt-dlp / mixesdb network calls are tested separately in eval/.
Here we cover the source detector and the two parsers.
"""
from __future__ import annotations

from mixid.pipeline import url_shortcut as us


def test_detect_source_handles_canonical_urls():
    assert us.detect_source("https://www.youtube.com/watch?v=abc") == "youtube"
    assert us.detect_source("https://youtu.be/abc") == "youtube"
    assert us.detect_source("https://music.youtube.com/watch?v=abc") == "youtube"
    assert us.detect_source("https://soundcloud.com/dj/mix") == "soundcloud"
    assert us.detect_source("https://www.mixcloud.com/dj/mix") == "mixcloud"
    assert us.detect_source("https://www.mixesdb.com/w/2024-12-31_-_DJ_X") == "mixesdb"
    assert us.detect_source("/local/path/to.mp3") is None
    assert us.detect_source("") is None


def test_parse_timestamped_tracklist_handles_common_formats():
    desc = """
Some intro text that should be ignored.

00:00 Artist One - Track Title
03:45 Artist Two – Other Track
1:23:45 [Artist Three] - Late Track
1. 06:30 Numbered Artist - Numbered Track
non-track line without timestamps
07:15 No Separator Title Goes Here
"""
    entries = us.parse_timestamped_tracklist(desc, source="youtube_description")
    assert len(entries) == 5
    # Entries are sorted by timestamp ascending
    times = [e.start_sec for e in entries]
    assert times == sorted(times)
    # Every expected timestamp shows up exactly once
    assert set(times) == {0, 225, 390, 435, 5025}
    # The "hh:mm:ss" parse worked (1:23:45 = 5025 sec)
    late = [e for e in entries if e.start_sec == 5025][0]
    assert late.artist == "Artist Three"
    # The "no separator" line still parses as title-only
    no_sep = [e for e in entries if e.title == "No Separator Title Goes Here"]
    assert no_sep and no_sep[0].artist == ""


def test_parse_mixesdb_wikitext_extracts_tracks():
    wiki = """
== Description ==
Some preamble.

== Tracklist ==
# [[Burna Boy]] - [[Last Last]]
# Artist Two - Some Track
# 3. [[Artist Three]] – Hyphen Style

== Comments ==
Other stuff here.
# This should NOT appear because we left the tracklist section.
"""
    entries = us.parse_mixesdb_wikitext(wiki)
    assert len(entries) == 3
    assert entries[0].artist == "Burna Boy"
    assert entries[0].title == "Last Last"
    assert entries[0].source == "mixesdb"
    assert entries[2].artist == "Artist Three"


def test_parse_mixesdb_wikitext_returns_empty_when_no_tracklist():
    wiki = "== Description ==\nSome text only."
    assert us.parse_mixesdb_wikitext(wiki) == []


def test_split_artist_title_picks_first_dash():
    assert us._split_artist_title("A - B - C") == ("A", "B - C")
    assert us._split_artist_title("A – B") == ("A", "B")
    assert us._split_artist_title("Title Only") == ("", "Title Only")
