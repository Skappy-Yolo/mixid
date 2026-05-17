"""LLM re-ranker — constrained to candidates produced by the audio pipeline.

The fingerprint matchers (local library + AcoustID) may each return one
or more candidates per segment with different confidence scores. When
multiple matchers disagree, or when a single matcher is unsure, this
module asks the configured LLM to pick the best fit from the candidates
or declare 'unknown'. The LLM CANNOT invent a new title — it can only
choose among candidates we already trust the audio pipeline to have
found, plus an 'unknown' opt-out. This makes hallucination impossible
by construction.

If no LLM is configured, the re-ranker degrades gracefully: it picks
the highest-scoring candidate per segment. This is what you'd do
anyway in the absence of better context.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from mixid.pipeline import ai_provider


log = logging.getLogger(__name__)


@dataclass
class Candidate:
    artist: str
    title: str
    score: float            # the matcher's own confidence in [0, 1]
    source: str             # "library" | "acoustid" | "mixesdb" | "youtube_description" | ...
    extra: dict = field(default_factory=dict)  # arbitrary matcher-specific info


@dataclass
class SegmentCandidates:
    segment_index: int
    start_sec: float
    end_sec: float
    candidates: list[Candidate]
    context: dict | None = None  # bpm, key, prior-track, etc. — informs LLM


@dataclass
class RerankedResult:
    segment_index: int
    start_sec: float
    end_sec: float
    artist: str
    title: str
    score: float
    source: str
    rationale: str = ""     # short string from LLM if it provided one


_SYSTEM_PROMPT = """\
You re-rank candidate songs that fingerprint matchers identified for each segment of a DJ mix.

You receive, per segment:
- A list of candidates, each with: index, source, artist, title, score (0..1)
- Optional context: BPM, key, prior identified track (continuity hint)

Your job is to pick exactly ONE candidate per segment by its `index`,
OR pick `null` if NONE of the candidates is plausible.

HARD RULES:
1. You MUST pick from the given candidates. NEVER invent a new artist or title.
2. Higher fingerprint score is a strong prior; do not override it without reason.
3. Continuity with the prior track (genre, BPM, key) can break ties, but never
   beats a candidate with score >= 0.85 from the audio pipeline.
4. If candidates are empty or all scores are below 0.5 and they don't fit
   any sensible reading of the segment, pick `null`.

Output STRICT JSON, no markdown, no commentary:
{
  "picks": [
    {"segment_index": 0, "candidate_index": 2, "rationale": "..."},
    {"segment_index": 1, "candidate_index": null, "rationale": "..."},
    ...
  ]
}
"""


def _greedy_fallback(pool: SegmentCandidates) -> RerankedResult | None:
    """Pick highest-scoring candidate; None if none above 0.5."""
    if not pool.candidates:
        return None
    best = max(pool.candidates, key=lambda c: c.score)
    if best.score < 0.5:
        return None
    return RerankedResult(
        segment_index=pool.segment_index,
        start_sec=pool.start_sec,
        end_sec=pool.end_sec,
        artist=best.artist,
        title=best.title,
        score=best.score,
        source=best.source,
        rationale="greedy:highest-score (no LLM)",
    )


def _format_user_prompt(pools: list[SegmentCandidates]) -> str:
    blocks: list[str] = []
    for pool in pools:
        cand_lines = [
            f"  [{i}] source={c.source!r} artist={c.artist!r} title={c.title!r} score={c.score:.2f}"
            for i, c in enumerate(pool.candidates)
        ]
        ctx_line = f"  context: {json.dumps(pool.context)}" if pool.context else ""
        blocks.append(
            f"Segment {pool.segment_index} ({pool.start_sec:.1f}-{pool.end_sec:.1f}s):\n"
            + "\n".join(cand_lines)
            + (("\n" + ctx_line) if ctx_line else "")
        )
    return (
        "\n\n".join(blocks)
        + "\n\nOutput the JSON object now."
    )


def _parse_llm_picks(raw: str) -> dict[int, dict]:
    """Extract {segment_index: {candidate_index, rationale}} from LLM JSON."""
    # Be tolerant of LLMs that wrap JSON in markdown fences despite the instruction
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    for entry in data.get("picks") or []:
        try:
            si = int(entry["segment_index"])
        except (KeyError, TypeError, ValueError):
            continue
        out[si] = {
            "candidate_index": entry.get("candidate_index"),
            "rationale": entry.get("rationale", ""),
        }
    return out


def rerank(pools: list[SegmentCandidates]) -> list[RerankedResult]:
    """Re-rank candidates across segments. Returns one result per pool, where present."""
    if not pools:
        return []

    # If no segment has more than one candidate, no LLM input is meaningful — go greedy.
    needs_llm = any(len(p.candidates) > 1 for p in pools)
    if not needs_llm:
        return [r for r in (_greedy_fallback(p) for p in pools) if r is not None]

    try:
        raw = ai_provider.complete(_SYSTEM_PROMPT, _format_user_prompt(pools))
    except ai_provider.NoProviderConfigured as e:
        log.warning("LLM not configured (%s). Falling back to greedy re-rank.", e)
        return [r for r in (_greedy_fallback(p) for p in pools) if r is not None]
    except Exception as e:
        log.warning("LLM re-rank failed (%s). Falling back to greedy.", e)
        return [r for r in (_greedy_fallback(p) for p in pools) if r is not None]

    picks = _parse_llm_picks(raw)
    out: list[RerankedResult] = []
    for pool in pools:
        pick = picks.get(pool.segment_index)
        if pick is None:
            # LLM didn't address this segment — fall back to greedy
            r = _greedy_fallback(pool)
            if r:
                out.append(r)
            continue
        ci = pick["candidate_index"]
        if ci is None:
            continue  # LLM picked 'unknown'
        try:
            chosen = pool.candidates[int(ci)]
        except (TypeError, ValueError, IndexError):
            continue
        out.append(
            RerankedResult(
                segment_index=pool.segment_index,
                start_sec=pool.start_sec,
                end_sec=pool.end_sec,
                artist=chosen.artist,
                title=chosen.title,
                score=chosen.score,
                source=chosen.source,
                rationale=str(pick.get("rationale", ""))[:200],
            )
        )
    return out
