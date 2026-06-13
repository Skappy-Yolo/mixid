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


# ── YouTube native chapters ──────────────────────────────────────────


def test_parse_chapters_extracts_tracks_and_skips_generic():
    meta = {
        "chapters": [
            {"start_time": 0.0, "end_time": 200.0, "title": "Intro"},
            {"start_time": 200.0, "end_time": 420.0, "title": "Burna Boy - Last Last"},
            {"start_time": 420.0, "end_time": 600.0, "title": "02. Asake - Lonely At The Top"},
            {"start_time": 600.0, "end_time": 800.0, "title": "ID"},
            {"start_time": 800.0, "end_time": 999.0, "title": "Outro / ID"},
        ]
    }
    entries = us.parse_chapters(meta, source="youtube_chapters")
    assert len(entries) == 2
    assert entries[0].start_sec == 200.0
    assert entries[0].artist == "Burna Boy" and entries[0].title == "Last Last"
    assert entries[0].source == "youtube_chapters"
    # leading "02." track number is stripped before the artist/title split
    assert entries[1].artist == "Asake" and entries[1].title == "Lonely At The Top"


def test_parse_chapters_handles_missing_or_malformed():
    assert us.parse_chapters({}, "youtube_chapters") == []
    assert us.parse_chapters({"chapters": None}, "youtube_chapters") == []
    assert us.parse_chapters({"chapters": "nope"}, "youtube_chapters") == []
    # missing title or start_time → skipped
    assert us.parse_chapters(
        {"chapters": [{"title": "", "start_time": 5}, {"title": "X - Y"}]},
        "youtube_chapters",
    ) == []


def test_source_priority_includes_chapters():
    assert us.SOURCE_PRIORITY.get("youtube_chapters") == 2
    assert us.SOURCE_PRIORITY.get("soundcloud_chapters") == 2


def test_merge_keeps_description_only_tracks_alongside_chapters():
    """Regression: a sparse chapter set must NOT suppress a fuller description.

    Chapters and timed-description entries are both collected; the dedup
    collapses the overlap but keeps description-only tracks the chapters
    never mentioned.
    """
    TE = us.TracklistEntry
    chapters = [
        TE(start_sec=0.0, artist="A", title="Opener", source="youtube_chapters"),
        TE(start_sec=600.0, artist="B", title="Highlight", source="youtube_chapters"),
    ]
    # Description has the same two (near the same times) plus three extras
    description = [
        TE(start_sec=2.0, artist="A", title="Opener", source="youtube_description"),
        TE(start_sec=200.0, artist="C", title="Extra One", source="youtube_description"),
        TE(start_sec=400.0, artist="D", title="Extra Two", source="youtube_description"),
        TE(start_sec=605.0, artist="B", title="Highlight", source="youtube_description"),
        TE(start_sec=800.0, artist="E", title="Extra Three", source="youtube_description"),
    ]
    merged = us._merge_and_dedup(chapters + description)
    titles = {e.title for e in merged}
    # The three description-only tracks survive
    assert {"Extra One", "Extra Two", "Extra Three"} <= titles
    # The two overlapping tracks are not double-counted
    assert sum(1 for e in merged if e.title == "Opener") == 1
    assert sum(1 for e in merged if e.title == "Highlight") == 1
    # Overlap resolved to the chapter source (equal priority, chapters inserted first)
    opener = [e for e in merged if e.title == "Opener"][0]
    assert opener.source == "youtube_chapters"
