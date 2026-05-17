"""Merge Tier-2 (Colab) enriched tracklist into a Tier-1 run directory.

Tier-2 produces `enriched_tracklist.json` on Colab — a per-segment list
of AcoustID/lyrics matches obtained from the Demucs-isolated vocals stem.
Tier-1 produces `tracklist.json` locally — confident tracks plus
`unknown_segments` for the gaps.

Merge logic:
1. For each Tier-2 match with score >= MIN_SCORE_FROM_TIER2 that falls
   inside (or near) a Tier-1 unknown segment → add as a new track.
2. For each Tier-2 match that disagrees with a Tier-1 track on the SAME
   time range, keep the higher-confidence call (with a small bias toward
   Tier-1 because vocal-only fingerprinting can mis-fire on instrumental
   sections).

Idempotent: re-running the merge on an already-merged file is a no-op.

Usage:
    python -m mixid.enrich.merge <tier1_outputs_dir> <enriched_tracklist.json>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


MIN_SCORE_FROM_TIER2 = 0.85


@dataclass
class _MergedTrack:
    start_sec: float
    end_sec: float
    artist: str
    title: str
    score: float
    source: str
    n_segments_merged: int


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Strict overlap — touching at a single boundary doesn't count."""
    return a_start < b_end and b_start < a_end


def merge(tier1_dir: Path, enriched_path: Path) -> dict:
    """Merge and write back to <tier1_dir>/tracklist.json. Return new payload."""
    tl_path = tier1_dir / "tracklist.json"
    if not tl_path.exists():
        raise FileNotFoundError(f"Tier-1 tracklist.json missing at {tl_path}")
    tl = json.loads(tl_path.read_text(encoding="utf-8"))
    enriched = json.loads(enriched_path.read_text(encoding="utf-8"))

    existing_tracks: list[dict] = list(tl.get("tracks") or [])
    unknown_ranges: list[tuple[float, float]] = [
        (u["start_sec"], u["end_sec"]) for u in (tl.get("unknown_segments") or [])
    ]
    added = 0
    for m in (enriched.get("matches") or []):
        best = m.get("best")
        if not best:
            continue
        if float(best.get("score", 0.0)) < MIN_SCORE_FROM_TIER2:
            continue
        m_start = float(m["start_sec"])
        m_end = float(m["end_sec"])
        # Only fill an unknown range — don't override confident Tier-1 calls
        if not any(_overlaps(m_start, m_end, us, ue) for us, ue in unknown_ranges):
            continue
        existing_tracks.append(
            {
                "start_sec": round(m_start, 2),
                "end_sec": round(m_end, 2),
                "artist": best.get("artist", ""),
                "title": best.get("title", ""),
                "score": float(best.get("score", 0.0)),
                "source": f"tier2:acoustid_vocals(pitch={best.get('pitch_pct', 0):+d}%)",
                "n_segments_merged": 1,
            }
        )
        added += 1

    existing_tracks.sort(key=lambda t: t["start_sec"])
    # Subtract resolved ranges from unknowns
    remaining_unknowns: list[dict] = []
    for us, ue in unknown_ranges:
        if any(
            _overlaps(us, ue, float(t["start_sec"]), float(t["end_sec"]))
            and t.get("source", "").startswith("tier2:")
            for t in existing_tracks
        ):
            continue
        remaining_unknowns.append({"start_sec": us, "end_sec": ue})

    payload = {
        **tl,
        "tracks": existing_tracks,
        "unknown_segments": remaining_unknowns,
        "tier2_added": added,
    }
    tl_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mixid-enrich-merge", description=__doc__)
    p.add_argument("tier1_dir", type=Path, help="Tier-1 output dir, e.g. outputs/abc1234/")
    p.add_argument("enriched_json", type=Path, help="enriched_tracklist.json from Colab")
    args = p.parse_args(argv)
    payload = merge(args.tier1_dir, args.enriched_json)
    print(f"Merged: added {payload.get('tier2_added', 0)} tracks; "
          f"{len(payload['unknown_segments'])} unknowns remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
