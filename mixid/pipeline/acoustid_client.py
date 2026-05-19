"""AcoustID remote matcher — the PRIMARY matcher for MixID.

AcoustID (https://acoustid.org/) is a free open-source audio fingerprint
database backed by MusicBrainz, with ~50M tracks indexed. Anyone can
use it; the rate limit is 1 req/sec without an API key, 3 req/sec with
one (free at https://acoustid.org/api-key).

We submit the base64 Chromaprint fingerprint + duration; AcoustID
returns recording IDs with confidence scores and metadata. If no key
is configured, this module logs a warning and returns no matches —
the rest of the pipeline (URL shortcut, Tier-2 enrichment) still works.

Pitch-shift strategy: try the 0% (unshifted) variant first. If the score
is below threshold, fan out to the shifted variants in expanding order
(±2%, ±4%, ±6%) until we find a confident hit. Saves API calls on the
common case where the DJ didn't pitch the track much.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import acoustid

import config
from mixid.pipeline.fingerprint import FingerprintSweep


log = logging.getLogger(__name__)


@dataclass
class AcoustIDMatch:
    score: float                    # AcoustID's own confidence in [0, 1]
    title: str
    artist: str
    recording_id: str               # MusicBrainz recording UUID
    duration_secs: float | None
    best_pitch_shift_percent: int   # which variant produced this match


_CONFIDENT_FLOOR = 0.85
# Hard reject AcoustID matches below this score — observed false positives
# at 0.55 (Anthony Robbins self-help audiobook, violin concertos, etc.)
# when noisy phone audio randomly matches a niche MusicBrainz entry.
_ACCEPT_FLOOR = 0.80
_LAST_CALL_AT: float = 0.0
_MIN_INTERVAL_SECS = 0.34          # 3 req/sec with key; 1 req/sec without (see _throttle)


def _throttle(has_key: bool) -> None:
    """Enforce per-process rate limit before each call."""
    global _LAST_CALL_AT
    interval = _MIN_INTERVAL_SECS if has_key else 1.05
    elapsed = time.monotonic() - _LAST_CALL_AT
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _LAST_CALL_AT = time.monotonic()


def _parse_response(resp: dict, pitch_pct: int) -> AcoustIDMatch | None:
    if resp.get("status") != "ok":
        return None
    for result in resp.get("results") or []:
        score = float(result.get("score", 0.0))
        recordings = result.get("recordings") or []
        if not recordings:
            continue
        rec = recordings[0]
        title = rec.get("title", "") or ""
        artists = rec.get("artists") or []
        artist = ", ".join(a.get("name", "") for a in artists) if artists else ""
        if not title and not artist:
            continue
        return AcoustIDMatch(
            score=score,
            title=title,
            artist=artist,
            recording_id=rec.get("id", ""),
            duration_secs=float(rec["duration"]) if rec.get("duration") else None,
            best_pitch_shift_percent=pitch_pct,
        )
    return None


def lookup_sweep(
    sweep: FingerprintSweep,
    api_key: str = "",
    confident_floor: float = _CONFIDENT_FLOOR,
) -> AcoustIDMatch | None:
    """Submit each pitch variant in expanding order until a confident hit lands."""
    api_key = api_key or config.ACOUSTID_API_KEY
    if not api_key:
        log.warning(
            "No ACOUSTID_API_KEY in .env — skipping AcoustID lookup. Get a free "
            "key at https://acoustid.org/api-key."
        )
        return None

    # Order: 0% first, then expanding by absolute shift.
    ordered = sorted(sweep.fingerprints, key=lambda fp: abs(fp.pitch_shift_percent))
    best: AcoustIDMatch | None = None
    for fp in ordered:
        if not fp.b64:
            continue
        _throttle(has_key=bool(api_key))
        try:
            resp = acoustid.lookup(
                api_key, fp.b64, int(fp.duration_secs), meta="recordings"
            )
        except acoustid.WebServiceError as e:
            log.warning("AcoustID error at %+d%%: %s", fp.pitch_shift_percent, e)
            continue
        match = _parse_response(resp, fp.pitch_shift_percent)
        if match is None:
            continue
        if match.score < _ACCEPT_FLOOR:
            continue  # too low to trust — same random-noise floor we use elsewhere
        if best is None or match.score > best.score:
            best = match
        if best.score >= confident_floor:
            return best  # short-circuit on confident hit
    return best
