"""Tests for the Tier-2 → Tier-1 merge."""
from __future__ import annotations

import json
from pathlib import Path

from mixid.enrich import merge


def _write_tier1(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_input": "mix.mp3",
                "tracks": [
                    {"start_sec": 0.0, "end_sec": 60.0, "artist": "A", "title": "X",
                     "score": 0.95, "source": "library", "n_segments_merged": 2}
                ],
                "unknown_segments": [
                    {"start_sec": 60.0, "end_sec": 90.0},
                    {"start_sec": 120.0, "end_sec": 150.0},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_enriched(path: Path, score: float = 0.92) -> None:
    path.write_text(
        json.dumps(
            {
                "source_mix": "mix.mp3",
                "method": "tier2",
                "matches": [
                    {
                        "segment_index": 1,
                        "start_sec": 60.0,
                        "end_sec": 90.0,
                        "best": {
                            "score": score,
                            "artist": "B",
                            "title": "Y",
                            "pitch_pct": 2,
                            "recording_id": "rec1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_merge_adds_tier2_track_into_unknown_gap(tmp_path: Path):
    _write_tier1(tmp_path / "tracklist.json")
    _write_enriched(tmp_path / "enriched.json")
    payload = merge.merge(tmp_path, tmp_path / "enriched.json")
    assert payload["tier2_added"] == 1
    titles = [t["title"] for t in payload["tracks"]]
    assert "X" in titles and "Y" in titles
    # The resolved unknown should be gone; the other should remain
    starts = [u["start_sec"] for u in payload["unknown_segments"]]
    assert 60.0 not in starts
    assert 120.0 in starts


def test_merge_skips_low_confidence_matches(tmp_path: Path):
    _write_tier1(tmp_path / "tracklist.json")
    _write_enriched(tmp_path / "enriched.json", score=0.70)  # below 0.85 floor
    payload = merge.merge(tmp_path, tmp_path / "enriched.json")
    assert payload["tier2_added"] == 0
    # Both unknowns survive
    assert len(payload["unknown_segments"]) == 2


def test_merge_does_not_override_confident_tier1_track(tmp_path: Path):
    """A tier-2 match that overlaps a tier-1 confident track must NOT clobber it."""
    _write_tier1(tmp_path / "tracklist.json")
    # Tier-2 match overlaps tier-1's "A - X" range (0-60s), but should be ignored
    (tmp_path / "enriched.json").write_text(
        json.dumps({
            "matches": [{
                "segment_index": 0, "start_sec": 0.0, "end_sec": 60.0,
                "best": {"score": 0.99, "artist": "WRONG", "title": "BAD", "pitch_pct": 0},
            }]
        }),
        encoding="utf-8",
    )
    payload = merge.merge(tmp_path, tmp_path / "enriched.json")
    assert payload["tier2_added"] == 0
    titles = [t["title"] for t in payload["tracks"]]
    assert "X" in titles and "BAD" not in titles
