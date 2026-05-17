"""Allow `python -m mixid ...` to invoke the CLI."""
from mixid.cli import main

raise SystemExit(main())
