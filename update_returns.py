"""兼容入口：转调 scripts/update_returns.py。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_returns import main

if __name__ == "__main__":
    raise SystemExit(main())
