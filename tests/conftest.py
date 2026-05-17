"""Pytest configuration — make `import config` and `import mixid` work."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
