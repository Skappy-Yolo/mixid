"""MixID CLI entry point. Implemented progressively across phases 1-10."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: mixid <file_or_url>", file=sys.stderr)
        return 2
    print(f"mixid v0.0.1 — pipeline not yet wired; see Phase 10. input: {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
