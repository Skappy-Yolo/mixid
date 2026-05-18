"""MixID CLI — `python -m mixid <file_or_url>`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mixid.pipeline import run as run_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mixid",
        description="Identify every track in a long-form DJ mix.",
    )
    parser.add_argument(
        "input",
        help="Path to an audio file (mp3/wav/m4a/...) OR a URL to a "
        "YouTube/SoundCloud/Mixcloud/mixesdb DJ mix.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path, default=None,
        help="Directory to write tracklist.json / mix.m3u / mix.txt "
        "(default: outputs/<random run id>/).",
    )
    parser.add_argument(
        "--with-demucs",
        action="store_true",
        help="After Tier-1, run Demucs stem separation on unidentified segments "
        "and retry Shazam on the no-vocals stem. Slow (CPU Demucs is ~15-30s "
        "per segment) but rescues hits buried under crowd noise.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_mod.run(args.input, output_dir=args.output_dir, with_demucs=args.with_demucs)

    # Print a short summary to stdout so callers piping output get something useful
    print(f"\nMixID — {len(result.tracks)} tracks, {len(result.unknown_segments)} unidentified")
    print(f"Output dir: {result.output_dir}")
    if result.timings_sec:
        total = result.timings_sec.get("total", 0.0)
        print(f"Total time: {total:.1f}s")
        for stage, secs in result.timings_sec.items():
            if stage == "total":
                continue
            print(f"  {stage:>22s}: {secs:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
