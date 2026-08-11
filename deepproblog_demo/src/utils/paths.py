"""Path helpers.

The official example scripts assume the process cwd is the example's own
directory inside the library. deepproblog is a plain pip dependency here,
not vendored, so those paths are resolved explicitly.
"""

from pathlib import Path

import deepproblog.examples as dpl_examples

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = DEMO_ROOT / "data"
LOG_DIR = DEMO_ROOT / "log"
FIGS_DIR = DEMO_ROOT / "figs"

EXAMPLES_DIR = Path(dpl_examples.__file__).parent

def example_dir(*parts: str) -> Path:
    return EXAMPLES_DIR.joinpath(*parts)
