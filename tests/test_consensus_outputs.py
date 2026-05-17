"""Tests for consensus merging and the three output writers."""
from __future__ import annotations

import json

from mixid.pipeline.consensus import collapse
from mixid.pipeline.reranker import RerankedResult
from mixid import outputs


def _r(idx: int, start: float, end: float, artist: str, title: str, score: float = 0.9) -> RerankedResult:
    return RerankedResult(
        segment_index=idx,
        start_sec=start,
        end_sec=end,
        artist=artist,
        title=title,
        score=score,
        source="acoustid",
    )


def test_collapse_merges_consecutive_duplicates():
    rs = [
        _r(0, 0, 30, "Burna Boy", "Last Last"),
        _r(1, 30, 60, "Burna Boy", "Last Last"),
        _r(2, 60, 90, "Asake", "Lonely at the Top"),
        _r(3, 90, 120, "Asake", "Lonely at the Top"),
        _r(4, 120, 150, "Burna Boy", "Last Last"),  # different run, must NOT merge with first
    ]
    merged = collapse(rs)
    assert len(merged) == 3
    assert merged[0].title == "Last Last" and merged[0].n_segments_merged == 2
    assert merged[0].start_sec == 0 and merged[0].end_sec == 60
    assert merged[2].title == "Last Last"  # second drop is a separate entry
    assert merged[2].start_sec == 120


def test_collapse_handles_empty():
    assert collapse([]) == []


def test_collapse_handles_single():
    r = [_r(0, 0, 60, "A", "B")]
    m = collapse(r)
    assert len(m) == 1 and m[0].artist == "A"


def test_write_json_round_trips(tmp_path):
    tracks = collapse([_r(0, 0, 30, "A", "T1", 0.9), _r(1, 30, 60, "A", "T1", 0.85)])
    out = outputs.write_json(
        tracks,
        tmp_path / "tl.json",
        source_input="http://example.com/mix",
        unknown_segments=[(120.0, 150.0)],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source_input"] == "http://example.com/mix"
    assert len(data["tracks"]) == 1
    assert data["tracks"][0]["title"] == "T1"
    assert data["tracks"][0]["n_segments_merged"] == 2
    assert data["unknown_segments"][0] == {"start_sec": 120.0, "end_sec": 150.0}


def test_write_m3u_uses_lookup_when_present(tmp_path):
    tracks = collapse([_r(0, 0, 60, "Burna Boy", "Last Last")])
    out = outputs.write_m3u(
        tracks,
        tmp_path / "tl.m3u",
        music_library_lookup={"burna boy|last last": "C:/music/burna - last last.mp3"},
    )
    content = out.read_text(encoding="utf-8")
    assert "#EXTM3U" in content
    assert "Burna Boy - Last Last" in content
    assert "C:/music/burna - last last.mp3" in content


def test_write_txt_includes_unknown_markers_in_timestamp_order(tmp_path):
    tracks = collapse([_r(0, 0, 30, "A", "X"), _r(1, 60, 90, "B", "Y")])
    out = outputs.write_txt(
        tracks, tmp_path / "tl.txt", unknown_segments=[(30.0, 60.0)]
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # Lines should be in start_sec order
    assert "A - X" in lines[0]
    assert "[unidentified:" in lines[1]
    assert "B - Y" in lines[2]
