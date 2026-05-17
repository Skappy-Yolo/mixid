"""ACRCloud client unit tests — signature, parser, graceful skip."""
from __future__ import annotations

import base64
import hashlib
import hmac

import numpy as np

import config
from mixid.pipeline import acrcloud_client


def test_signature_matches_canonical_form():
    """HMAC-SHA1 over the exact canonical string ACRCloud expects."""
    key, secret, ts = "myKey", "mySecret", "1700000000"
    expected_str = f"POST\n/v1/identify\nmyKey\naudio\n1\n1700000000"
    expected = base64.b64encode(
        hmac.new(b"mySecret", expected_str.encode(), hashlib.sha1).digest()
    ).decode()
    assert acrcloud_client._build_signature(key, secret, ts) == expected


def test_parse_response_extracts_top_music_match():
    resp = {
        "status": {"code": 0, "msg": "Success"},
        "metadata": {
            "music": [
                {
                    "acrid": "rec-abc",
                    "title": "Last Last",
                    "artists": [{"name": "Burna Boy"}],
                    "album": {"name": "Love, Damini"},
                    "score": 87,
                    "duration_ms": 173000,
                }
            ]
        },
    }
    m = acrcloud_client._parse_response(resp)
    assert m is not None
    assert m.title == "Last Last"
    assert m.artist == "Burna Boy"
    assert m.album == "Love, Damini"
    assert m.score == 0.87  # normalized from 87
    assert m.duration_ms == 173000


def test_parse_response_no_match_returns_none():
    resp = {"status": {"code": 1001, "msg": "No result"}, "metadata": {"music": []}}
    assert acrcloud_client._parse_response(resp) is None


def test_parse_response_empty_metadata_returns_none():
    resp = {"status": {"code": 0}, "metadata": {}}
    assert acrcloud_client._parse_response(resp) is None


def test_no_credentials_returns_none(monkeypatch, caplog):
    monkeypatch.setattr(config, "ACRCLOUD_KEY", "")
    monkeypatch.setattr(config, "ACRCLOUD_SECRET", "")
    monkeypatch.setattr(config, "ACRCLOUD_HOST", "")
    monkeypatch.setattr(acrcloud_client.config, "ACRCLOUD_KEY", "")
    monkeypatch.setattr(acrcloud_client.config, "ACRCLOUD_SECRET", "")
    monkeypatch.setattr(acrcloud_client.config, "ACRCLOUD_HOST", "")
    samples = np.zeros(22050, dtype=np.float32)
    with caplog.at_level("WARNING"):
        result = acrcloud_client.lookup_sample(samples, sr=22050)
    assert result is None
    assert any("ACRCloud" in r.message for r in caplog.records)


def test_samples_to_wav_bytes_round_trips():
    """Sanity: the WAV bytes we'd POST are decodable back to the same shape."""
    import io
    import soundfile as sf

    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050, dtype=np.float32)) * 0.5
    blob = acrcloud_client._samples_to_wav_bytes(samples, sr=22050)
    decoded, sr = sf.read(io.BytesIO(blob))
    assert sr == 22050
    assert len(decoded) == len(samples)
