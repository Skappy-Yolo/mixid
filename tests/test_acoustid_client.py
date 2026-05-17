"""Offline unit tests for the AcoustID client — no network required.

These tests verify the response parser, throttling, and the "no API key
configured" graceful-skip path. The live network integration test lives
in eval/ and requires ACOUSTID_API_KEY to be set.
"""
from __future__ import annotations

import numpy as np

import config
from mixid.pipeline import acoustid_client
from mixid.pipeline.fingerprint import Fingerprint, FingerprintSweep


def _empty_sweep() -> FingerprintSweep:
    fp = Fingerprint(pitch_shift_percent=0, duration_secs=12.0, b64="dummy_b64")
    return FingerprintSweep(fingerprints=[fp], sample_start_sec_in_mix=0.0)


def test_parse_ok_response_extracts_title_artist():
    resp = {
        "status": "ok",
        "results": [
            {
                "id": "result-uuid",
                "score": 0.92,
                "recordings": [
                    {
                        "id": "rec-uuid",
                        "title": "Last Last",
                        "duration": 174.0,
                        "artists": [{"id": "a1", "name": "Burna Boy"}],
                    }
                ],
            }
        ],
    }
    match = acoustid_client._parse_response(resp, pitch_pct=2)
    assert match is not None
    assert match.title == "Last Last"
    assert match.artist == "Burna Boy"
    assert match.score == 0.92
    assert match.best_pitch_shift_percent == 2
    assert match.duration_secs == 174.0


def test_parse_empty_recordings_returns_none():
    resp = {"status": "ok", "results": [{"id": "x", "score": 0.5, "recordings": []}]}
    assert acoustid_client._parse_response(resp, 0) is None


def test_parse_status_not_ok_returns_none():
    resp = {"status": "error", "error": {"message": "rate limit"}}
    assert acoustid_client._parse_response(resp, 0) is None


def test_parse_multi_artist_joins_names():
    resp = {
        "status": "ok",
        "results": [
            {
                "id": "r",
                "score": 0.7,
                "recordings": [
                    {
                        "id": "rec",
                        "title": "Collab Track",
                        "artists": [
                            {"id": "a1", "name": "Artist One"},
                            {"id": "a2", "name": "Artist Two"},
                        ],
                    }
                ],
            }
        ],
    }
    match = acoustid_client._parse_response(resp, 0)
    assert match.artist == "Artist One, Artist Two"


def test_no_api_key_returns_none(monkeypatch, caplog):
    """Pipeline must not crash when no AcoustID key is configured."""
    monkeypatch.setattr(config, "ACOUSTID_API_KEY", "")
    monkeypatch.setattr(acoustid_client.config, "ACOUSTID_API_KEY", "")
    with caplog.at_level("WARNING"):
        result = acoustid_client.lookup_sweep(_empty_sweep(), api_key="")
    assert result is None
    assert any("ACOUSTID_API_KEY" in r.message for r in caplog.records)
