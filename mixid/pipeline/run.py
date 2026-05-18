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
    local_demucs,
    reactive_lookup,
    reranker,
    sample_picker,
    segment as segment_mod,
    shazam_client,
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


def _shazam_to_candidate(m: shazam_client.ShazamMatch) -> reranker.Candidate:
    return reranker.Candidate(
        artist=m.artist,
        title=m.title,
        score=m.score,        # Shazam returns a match (binary); we trust it
        source="shazam",
        extra={"shazam_key": m.shazam_key, "isrc": m.isrc},
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


def _demucs_retry_for_unknowns(
    prepared: audio_prep.PreparedAudio,
    pools: list[reranker.SegmentCandidates],
    unknown_segments: list[tuple[float, float]],
    library_lookup: dict[str, str],
) -> tuple[list[reranker.SegmentCandidates], list[tuple[float, float]]]:
    """Run Demucs on segments that produced no confident candidate.

    For each unknown segment:
      1. Extract the 16-sec snippet from the prepared audio
      2. Demucs htdemucs_ft → vocals/drums/bass/other + no_vocals composite
      3. Try Shazam on the 'no_vocals' stem (less crowd contamination)
      4. If hit: append a SegmentCandidates pool for it

    Returns (added_pools, remaining_unknowns).
    """
    if not local_demucs.is_available():
        log.warning("--with-demucs requested but demucs not installed; skipping")
        return [], unknown_segments
    if not unknown_segments:
        return [], []

    added_pools: list[reranker.SegmentCandidates] = []
    remaining: list[tuple[float, float]] = []
    sr = prepared.sample_rate

    for i, (us, ue) in enumerate(unknown_segments):
        snippet = prepared.samples[int(us * sr) : int(ue * sr)]
        if len(snippet) < int(8 * sr):
            remaining.append((us, ue))
            continue
        log.info("demucs %d/%d: segment %.1f-%.1fs", i + 1, len(unknown_segments), us, ue)
        try:
            stems = local_demucs.separate_at_sr(snippet, sr, output_sr=sr)
        except Exception as e:
            log.warning("demucs failed on %.1f-%.1fs: %s", us, ue, e)
            remaining.append((us, ue))
            continue
        if stems is None or "no_vocals" not in stems:
            remaining.append((us, ue))
            continue
        sh = shazam_client.recognize_sample(stems["no_vocals"], sr)
        if sh is None:
            # also try the vocals stem in case it's a vocal-heavy track
            sh = shazam_client.recognize_sample(stems.get("vocals", snippet), sr)
        if sh is None:
            remaining.append((us, ue))
            continue
        cand = _shazam_to_candidate(sh)
        cand.source = "shazam+demucs"
        added_pools.append(reranker.SegmentCandidates(
            segment_index=-(i + 1),  # synthetic index
            start_sec=us,
            end_sec=ue,
            candidates=[cand],
        ))
    return added_pools, remaining


def run(
    input_path_or_url: str,
    output_dir: Path | None = None,
    with_demucs: bool = False,
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
        # Multi-position picks: up to N samples per segment. Shazam can miss a
        # 16-sec window that lands on a transition; trying 2-3 positions cheaply
        # rescues those misses.
        samples_in_seg = sample_picker.pick_top_k(
            prepared.samples, prepared.sample_rate, seg,
            k=config.SHAZAM_ATTEMPTS_PER_SEGMENT,
        )
        if not samples_in_seg:
            continue
        s = samples_in_seg[0]  # primary pick — used for fingerprint & non-Shazam matchers
        sweep = fingerprint.fingerprint_sample(
            s.samples,
            s.sample_rate,
            sample_start_sec_in_mix=s.start_sec_in_mix,
            include_b64=True,
        )
        candidates: list[reranker.Candidate] = []

        # 4c. Local library match (skipped silently when no index exists)
        if library_available:
            lib_matches = library_match.match_sweep(sweep, top_k=3)
            candidates.extend(_library_to_candidate(m) for m in lib_matches)

        # 4d. Shazam — try ALL picked positions until one hits.
        if not candidates or max(c.score for c in candidates) < 0.85:
            for s_try in samples_in_seg:
                sh = shazam_client.recognize_sample(s_try.samples, s_try.sample_rate)
                if sh is not None:
                    candidates.append(_shazam_to_candidate(sh))
                    break  # first hit wins; don't waste throttle budget

        # 4e. AcoustID remote (skipped when no API key). Lower priority than
        # Shazam — AcoustID's free DB is smaller and lacks edits/niche tracks.
        if not candidates or max(c.score for c in candidates) < 0.85:
            ac = acoustid_client.lookup_sweep(sweep)
            if ac is not None:
                candidates.append(_acoustid_to_candidate(ac))

        # 4f. Reactive lookup — whisper transcribe → multi-service lyric search
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

    # ── 5.25 Demucs noise-removal retry for unknowns (opt-in, slow) ────────
    if with_demucs and unknown_segments:
        t = time.monotonic()
        log.info("Demucs retry on %d unknown segments — may take several minutes", len(unknown_segments))
        added_pools, unknown_segments = _demucs_retry_for_unknowns(
            prepared, pools, unknown_segments, library_lookup
        )
        timings["demucs_retry"] = time.monotonic() - t
        # Re-rank the new pools too
        if added_pools:
            extra = reranker.rerank(added_pools)
            reranked.extend(extra)
            log.info("demucs added %d identifications", len(extra))

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
