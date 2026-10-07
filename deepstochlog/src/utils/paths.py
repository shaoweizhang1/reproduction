"""Directories of this project; upstream data resolve through upstream's own __file__."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.environ.get("DEEPSTOCHLOG_LOG_DIR", PROJECT_ROOT / "log"))
PROGRAMS_DIR = PROJECT_ROOT / "src" / "programs"
