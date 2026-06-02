"""Ensure ``src/`` is on ``sys.path`` when running scripts without install."""

import sys
from pathlib import Path


def bootstrap() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
