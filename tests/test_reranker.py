"""Offline tests: greedy fallback, JSON parsing, no-LLM-configured path.

Live LLM calls are skipped — those happen in eval/ end-to-end runs.
"""
from __future__ import annotations

from mixid.pipeline import reranker, ai_provider


def _pool(idx: int, *cands: tuple[str, str, float, str]) -> reranker.SegmentCandidates:
    return reranker.SegmentCandidates(
        segment_index=idx,
        start_sec=idx * 30.0,
        end_sec=(idx + 1) * 30.0,
        candidates=[
            reranker.Candidate(artist=a, title=t, score=s, source=src)
            for a, t, s, src in cands
        ],
    )


def test_greedy_fallback_picks_highest_score():
    p = _pool(
        0,
        ("Artist A", "Track A", 0.6, "library"),
        ("Artist B", "Track B", 0.92, "acoustid"),
    )
    r = reranker._greedy_fallback(p)
    assert r is not None and r.title == "Track B" and r.score == 0.92


def test_greedy_fallback_returns_none_below_threshold():
    p = _pool(0, ("X", "Y", 0.2, "library"))
    assert reranker._greedy_fallback(p) is None


def test_greedy_fallback_returns_none_when_no_candidates():
    p = reranker.SegmentCandidates(
        segment_index=0, start_sec=0, end_sec=10, candidates=[]
    )
    assert reranker._greedy_fallback(p) is None


def test_rerank_uses_greedy_when_only_one_candidate_per_segment():
    """Skip the LLM call when no segment has a choice to make."""
    pools = [
        _pool(0, ("Artist A", "Track A", 0.91, "library")),
        _pool(1, ("Artist B", "Track B", 0.88, "acoustid")),
    ]
    results = reranker.rerank(pools)
    assert len(results) == 2
    assert results[0].title == "Track A"
    assert results[1].title == "Track B"


def test_rerank_falls_back_to_greedy_when_llm_not_configured(monkeypatch):
    def boom(*a, **kw):
        raise ai_provider.NoProviderConfigured("no key")

    monkeypatch.setattr(ai_provider, "complete", boom)
    pools = [
        _pool(
            0,
            ("Artist A", "Track A", 0.6, "library"),
            ("Artist B", "Track B", 0.92, "acoustid"),
        ),
    ]
    results = reranker.rerank(pools)
    assert len(results) == 1 and results[0].title == "Track B"


def test_parse_llm_picks_extracts_segment_choices():
    raw = """
Here you go:
{
  "picks": [
    {"segment_index": 0, "candidate_index": 1, "rationale": "BPM matches"},
    {"segment_index": 1, "candidate_index": null, "rationale": "all wrong"}
  ]
}
"""
    parsed = reranker._parse_llm_picks(raw)
    assert parsed[0]["candidate_index"] == 1
    assert parsed[1]["candidate_index"] is None


def test_rerank_applies_llm_picks(monkeypatch):
    def fake_llm(sys_p, usr_p):
        return '{"picks":[{"segment_index":0,"candidate_index":0,"rationale":"continuity"}]}'

    monkeypatch.setattr(ai_provider, "complete", fake_llm)
    pools = [
        _pool(
            0,
            ("Artist A", "Track A", 0.6, "library"),
            ("Artist B", "Track B", 0.92, "acoustid"),
        ),
    ]
    results = reranker.rerank(pools)
    assert len(results) == 1
    # LLM picked index 0 even though candidate 1 had higher score —
    # that's the whole point of re-ranking
    assert results[0].title == "Track A"
    assert "continuity" in results[0].rationale


def test_claude_code_provider_invokes_cli(monkeypatch):
    """ai_provider._claude_code shells out to `claude --print` correctly."""
    import config as _config

    monkeypatch.setattr(_config, "AI_PROVIDER", "claude_code")
    monkeypatch.setattr(ai_provider.config, "AI_PROVIDER", "claude_code")

    captured = {}

    def fake_which(name):
        if name == "claude":
            return "/fake/path/claude"
        return None

    class FakeCompleted:
        returncode = 0
        stdout = "claude-code-replied"
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input", "")
        return FakeCompleted()

    import shutil, subprocess
    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    out = ai_provider.complete("be terse", "say hi")
    assert out == "claude-code-replied"
    assert captured["cmd"] == ["/fake/path/claude", "--print", "--output-format", "text"]
    assert "be terse" in captured["input"]
    assert "say hi" in captured["input"]


def test_claude_code_provider_raises_when_cli_missing(monkeypatch):
    import config as _config

    monkeypatch.setattr(_config, "AI_PROVIDER", "claude_code")
    monkeypatch.setattr(ai_provider.config, "AI_PROVIDER", "claude_code")
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    import pytest as _pytest
    with _pytest.raises(ai_provider.NoProviderConfigured, match="Claude Code CLI"):
        ai_provider.complete("", "hi")


def test_rerank_skips_segment_when_llm_picks_unknown(monkeypatch):
    def fake_llm(sys_p, usr_p):
        return '{"picks":[{"segment_index":0,"candidate_index":null,"rationale":"all wrong"}]}'

    monkeypatch.setattr(ai_provider, "complete", fake_llm)
    pools = [
        _pool(
            0,
            ("Artist A", "Track A", 0.6, "library"),
            ("Artist B", "Track B", 0.55, "acoustid"),
        ),
    ]
    results = reranker.rerank(pools)
    assert results == []  # segment marked unknown, dropped from output
