"""Lyrics module tests — phrase cleanup, Genius response parsing, graceful skips."""
from __future__ import annotations

import numpy as np

import config
from mixid.pipeline import lyrics


def test_clean_phrase_picks_longest_segment():
    text = "yeah, I been walking down the avenue thinking about you, ay ay"
    cleaned = lyrics._clean_phrase(text)
    assert cleaned == "I been walking down the avenue thinking about you"


def test_clean_phrase_strips_brackets_and_collapses_space():
    text = "(intro)   well   well  [chorus]  here we go now"
    cleaned = lyrics._clean_phrase(text)
    assert "[" not in cleaned and "(" not in cleaned
    assert "  " not in cleaned


def test_clean_phrase_caps_length():
    long = " ".join(["word"] * 100)
    cleaned = lyrics._clean_phrase(long)
    assert len(cleaned) <= 120


def test_search_genius_no_key_returns_none(monkeypatch, caplog):
    monkeypatch.setattr(config, "GENIUS_API_KEY", "")
    monkeypatch.setattr(lyrics.config, "GENIUS_API_KEY", "")
    with caplog.at_level("WARNING"):
        assert lyrics.search_genius("any phrase") is None
    assert any("GENIUS_API_KEY" in r.message for r in caplog.records)


def test_search_genius_handles_no_hits(monkeypatch):
    def fake_get(url, **kwargs):
        class R:
            ok = True
            def raise_for_status(self): pass
            def json(self_):
                return {"response": {"hits": []}}
        return R()

    monkeypatch.setattr(config, "GENIUS_API_KEY", "fake_key")
    monkeypatch.setattr(lyrics.config, "GENIUS_API_KEY", "fake_key")
    monkeypatch.setattr(lyrics.requests, "get", fake_get)
    assert lyrics.search_genius("query") is None


def test_search_genius_parses_top_hit(monkeypatch):
    def fake_get(url, **kwargs):
        class R:
            ok = True
            def raise_for_status(self): pass
            def json(self_):
                return {
                    "response": {
                        "hits": [
                            {
                                "result": {
                                    "id": 12345,
                                    "title": "Last Last",
                                    "primary_artist": {"name": "Burna Boy"},
                                }
                            }
                        ]
                    }
                }
        return R()

    monkeypatch.setattr(config, "GENIUS_API_KEY", "fake_key")
    monkeypatch.setattr(lyrics.config, "GENIUS_API_KEY", "fake_key")
    monkeypatch.setattr(lyrics.requests, "get", fake_get)
    m = lyrics.search_genius("I need igbo and shayo")
    assert m is not None
    assert m.title == "Last Last"
    assert m.artist == "Burna Boy"
    assert m.genius_id == 12345


def test_identify_via_lyrics_skips_when_whisper_missing(monkeypatch):
    monkeypatch.setattr(lyrics, "_whisper_available", lambda: False)
    samples = np.zeros(16000, dtype=np.float32)
    assert lyrics.identify_via_lyrics(samples, sr=16000) is None


def test_identify_via_lyrics_skips_short_transcript(monkeypatch):
    monkeypatch.setattr(lyrics, "_whisper_available", lambda: True)
    monkeypatch.setattr(lyrics, "transcribe_sample", lambda *a, **kw: "yeah")
    monkeypatch.setattr(config, "GENIUS_API_KEY", "fake_key")
    monkeypatch.setattr(lyrics.config, "GENIUS_API_KEY", "fake_key")
    samples = np.zeros(16000, dtype=np.float32)
    assert lyrics.identify_via_lyrics(samples, sr=16000) is None
