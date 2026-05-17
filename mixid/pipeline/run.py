"""Tier-1 orchestrator — wires every stage from input to output tracklist.

Pipeline:
  1. Input dispatch: URL → URL shortcut; file → audio pipeline
  2. Audio prep         (mono, highpass, loudness norm)
  3. Hybrid segmentation
  4. Per segment:
       a. Pick best 12-sec sample
       b. Chromaprint fingerprint sweep (7 pitch variants)
       c. Local library match  → 0..K candidates
       d. AcoustID remote      → 0..1 candidates
       → collect into SegmentCandidates
  5. LLM re-ranker (constrained to candidates, or greedy fallback)
  6. Consensus collapse (merge consecutive same-track segments)
  7. Write JSON / M3U / TXT outputs to outputs/<run_id>/

Tracks whose segment produced no candidates above threshold appear in
the `unknown_segments` field — Tier-2 enrichment (Demucs/Whisper/CLAP/
ACRCloud) operates on exactly those gaps.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from mixid.pipeline import (
    acoustid_client,
    audio_prep,
    fingerprint,
    library_match,
    reactive_lookup,
    reranker,
    sample_picker,
    segment as segment_mod,
    smoother,
    url_shortcut,
)
from mixid.pipeline.consensus import TracklistTrack, collapse
from mixid.outputs import write_json, write_m3u, write_txt


log = logging.getLogger(__name__)


@dataclass
class MixIDResult:
    source_input: str
    tracks: list[TracklistTrack]
    unknown_segments: list[tuple[float, float]]
    timings_sec: dict[str, float] = field(default_factory=dict)
    output_dir: Path | None = None


def _library_to_candidate(m: library_match.LibraryMatch) -> reranker.Candidate:
    return reranker.Candidate(
        artist=m.artist or "",
        title=m.title or "",
        score=m.score,
        source="library",
        extra={"filepath": m.filepath, "pitch_shift_pct": m.best_pitch_shift_percent},
    )


def _acoustid_to_candidate(m: acoustid_client.AcoustIDMatch) -> reranker.Candidate:
    return reranker.Candidate(
        artist=m.artist,
        title=m.title,
        score=m.score,
        source="acoustid",
        extra={
            "recording_id": m.recording_id,
            "pitch_shift_pct": m.best_pitch_shift_percent,
        },
    )


def _shortcut_to_tracks(entries: list[url_shortcut.TracklistEntry]) -> list[TracklistTrack]:
    """Promote URL-shortcut entries directly to TracklistTracks (no audio needed)."""
    out: list[TracklistTrack] = []
    for i, e in enumerate(entries):
        next_start = (
            entries[i + 1].start_sec
            if i + 1 < len(entries) and entries[i + 1].start_sec is not None
            else (e.start_sec or 0.0) + 60.0
        )
        out.append(
            TracklistTrack(
                start_sec=e.start_sec or 0.0,
                end_sec=float(next_start),
                artist=e.artist,
                title=e.title,
                score=1.0,  # community tracklists are treated as ground truth
                source=e.source,
                n_segments_merged=1,
            )
        )
    return out


def _build_library_lookup() -> dict[str, str]:
    """artist|title (lowered) → filepath, for M3U output writing."""
    import sqlite3

    if not config.FINGERPRINTS_DB.exists():
        return {}
    conn = sqlite3.connect(str(config.FINGERPRINTS_DB))
    rows = conn.execute("SELECT artist, title, filepath FROM tracks").fetchall()
    conn.close()
    return {
        f"{(a or '').strip().lower()}|{(t or '').strip().lower()}": fp
        for a, t, fp in rows
    }


def _write_all_outputs(
    result: MixIDResult,
    library_lookup: dict[str, str] | None = None,
) -> None:
    assert result.output_dir is not None
    write_json(
        result.tracks,
        result.output_dir / "tracklist.json",
        source_input=result.source_input,
        unknown_segments=result.unknown_segments,
    )
    write_m3u(
        result.tracks,
        result.output_dir / "mix.m3u",
        music_library_lookup=library_lookup,
    )
    write_txt(
        result.tracks,
        result.output_dir / "mix.txt",
        unknown_segments=result.unknown_segments,
    )


def run(
    input_path_or_url: str,
    output_dir: Path | None = None,
) -> MixIDResult:
    """Run the Tier-1 pipeline end-to-end. Writes outputs and returns the result."""
    t_start = time.monotonic()
    run_id = uuid.uuid4().hex[:8]
    output_dir = (output_dir or (config.OUTPUTS_DIR / run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    is_url = url_shortcut.detect_source(input_path_or_url) is not None

    # ── 1. URL shortcut ─────────────────────────────────────────────────────
    if is_url:
        t = time.monotonic()
        entries = url_shortcut.try_url_shortcut(input_path_or_url)
        timings["url_shortcut"] = time.monotonic() - t
        if entries:
            result = MixIDResult(
                source_input=input_path_or_url,
                tracks=_shortcut_to_tracks(entries),
                unknown_segments=[],
                timings_sec=timings,
                output_dir=output_dir,
            )
            _write_all_outputs(result, _build_library_lookup())
            timings["total"] = time.monotonic() - t_start
            log.info("URL shortcut returned %d entries — pipeline short-circuit.", len(entries))
            return result
        else:
            log.info("URL shortcut returned no entries — would need audio.")
            # For v1, URL without a shortcut hit + without local audio = empty result.
            result = MixIDResult(
                source_input=input_path_or_url,
                tracks=[],
                unknown_segments=[],
                timings_sec=timings,
                output_dir=output_dir,
            )
            _write_all_outputs(result)
            timings["total"] = time.monotonic() - t_start
            return result

    # ── 2-4. Audio pipeline ────────────────────────────────────────────────
    t = time.monotonic()
    prepared = audio_prep.prepare(input_path_or_url)
    timings["audio_prep"] = time.monotonic() - t

    t = time.monotonic()
    segments = segment_mod.segment(prepared.samples, prepared.sample_rate)
    timings["segmentation"] = time.monotonic() - t
    log.info("Segmented %.1f sec mix into %d segments.", prepared.duration_secs, len(segments))

    pools: list[reranker.SegmentCandidates] = []
    unknown_segments: list[tuple[float, float]] = []

    t_seg = time.monotonic()
    library_lookup = _build_library_lookup()
    library_available = bool(library_lookup)
    for idx, seg in enumerate(segments):
        s = sample_picker.pick(prepared.samples, prepared.sample_rate, seg)
        if s is None:
            continue
        sweep = fingerprint.fingerprint_sample(
            s.samples,
            s.sample_rate,
            sample_start_sec_in_mix=s.start_sec_in_mix,
            # Use the configured sweep; can be narrowed via config if too slow
            include_b64=True,
        )
        candidates: list[reranker.Candidate] = []

        # 4c. Local library match (skipped silently when no index exists)
        if library_available:
            lib_matches = library_match.match_sweep(sweep, top_k=3)
            candidates.extend(_library_to_candidate(m) for m in lib_matches)

        # 4d. AcoustID remote (skipped when no API key)
        if not candidates or max(c.score for c in candidates) < 0.85:
            ac = acoustid_client.lookup_sweep(sweep)
            if ac is not None:
                candidates.append(_acoustid_to_candidate(ac))

        # 4e. Reactive lookup — whisper transcribe → multi-service lyric search
        # → fingerprint-verify against preview. Only when nothing confident yet,
        # because each call is ~5-10 sec (one Whisper invocation + ~8 API calls).
        if (not candidates or max(c.score for c in candidates) < 0.85) and reactive_lookup._whisper_available():
            rm = reactive_lookup.identify_reactive(s.samples, s.sample_rate)
            if rm is not None:
                candidates.append(reranker.Candidate(
                    artist=rm.artist, title=rm.title, score=rm.score,
                    source=rm.source,
                    extra={"preview_url": rm.preview_url, "transcript": rm.transcript},
                ))

        if not candidates:
            unknown_segments.append((seg.start_sec, seg.end_sec))
            continue
        pools.append(
            reranker.SegmentCandidates(
                segment_index=idx,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                candidates=candidates,
            )
        )
    timings["per_segment_matching"] = time.monotonic() - t_seg
    log.info("Collected candidates for %d segments; %d unknown.", len(pools), len(unknown_segments))

    # ── 5. Re-rank ──────────────────────────────────────────────────────────
    t = time.monotonic()
    reranked = reranker.rerank(pools)
    timings["rerank"] = time.monotonic() - t

    # Segments where re-rank returned 'unknown' join unknown_segments
    reranked_idxs = {r.segment_index for r in reranked}
    for pool in pools:
        if pool.segment_index not in reranked_idxs:
            unknown_segments.append((pool.start_sec, pool.end_sec))

    # ── 5.5 Gap-fill smoother — rescue unknowns sandwiched by agreement ────
    t = time.monotonic()
    reranked, unknown_segments = smoother.smooth_gaps(reranked, unknown_segments)
    timings["smoother"] = time.monotonic() - t

    # ── 6. Consensus collapse ──────────────────────────────────────────────
    tracks = collapse(reranked)

    # ── 7. Outputs ──────────────────────────────────────────────────────────
    result = MixIDResult(
        source_input=input_path_or_url,
        tracks=tracks,
        unknown_segments=unknown_segments,
        timings_sec=timings,
        output_dir=output_dir,
    )
    _write_all_outputs(result, library_lookup)
    timings["total"] = time.monotonic() - t_start
    return result
