"""Render the final tracklist to JSON / M3U / plain-text.

Each format is one self-contained writer; the orchestrator (Phase 10)
calls all three so users get all forms in one run directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from mixid.pipeline.consensus import TracklistTrack


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def write_json(
    tracks: list[TracklistTrack],
    out_path: Path | str,
    *,
    source_input: str = "",
    unknown_segments: list[tuple[float, float]] | None = None,
) -> Path:
    """Structured tracklist with per-track provenance + unknown segments."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_input": source_input,
        "tracks": [
            {
                "start_sec": round(t.start_sec, 2),
                "end_sec": round(t.end_sec, 2),
                "artist": t.artist,
                "title": t.title,
                "score": round(t.score, 3),
                "source": t.source,
                "n_segments_merged": t.n_segments_merged,
            }
            for t in tracks
        ],
        "unknown_segments": [
            {"start_sec": round(s, 2), "end_sec": round(e, 2)}
            for s, e in (unknown_segments or [])
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_m3u(
    tracks: list[TracklistTrack],
    out_path: Path | str,
    *,
    music_library_lookup: dict[str, str] | None = None,
) -> Path:
    """Extended M3U. Filepaths come from `music_library_lookup` if provided.

    Keys in `music_library_lookup` are 'artist|title' (lower-cased, stripped).
    Tracks not in the lookup are still emitted as #EXTINF entries with an
    empty file line — useful as a viewer-friendly tracklist even without
    audio resolution.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#EXTM3U"]
    for t in tracks:
        dur = max(1, int(round(t.end_sec - t.start_sec)))
        display = f"{t.artist} - {t.title}" if t.artist else t.title
        lines.append(f"#EXTINF:{dur},{display}")
        key = f"{t.artist.strip().lower()}|{t.title.strip().lower()}"
        path = (music_library_lookup or {}).get(key, "")
        lines.append(path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_txt(
    tracks: list[TracklistTrack],
    out_path: Path | str,
    *,
    unknown_segments: list[tuple[float, float]] | None = None,
) -> Path:
    """Human-readable tracklist with timestamps and unknown-segment markers."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[float, str]] = []
    for t in tracks:
        ts = _fmt_hms(t.start_sec)
        display = f"{t.artist} - {t.title}" if t.artist else t.title
        entries.append((t.start_sec, f"{ts}  {display}  [{t.source}, score={t.score:.2f}]"))
    for s, e in unknown_segments or []:
        entries.append(
            (s, f"{_fmt_hms(s)}  [unidentified: {_fmt_hms(s)}-{_fmt_hms(e)}]")
        )
    entries.sort(key=lambda x: x[0])
    out_path.write_text("\n".join(line for _, line in entries) + "\n", encoding="utf-8")
    return out_path
