"""Where things live. Import from here; never hard-code paths in notebooks.

Everything is relative to the repo root, so the same code runs on the Mac (analysis) and on
Kaggle (extraction, where the repo is cloned into the working directory)."""
from pathlib import Path
import os

PROJECT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent))
DATA = PROJECT / "data"            # extraction outputs, committed (step 4)
ACTS = DATA / "acts.npy"           # (n_rows, d_model) layer-12 residual, mean over generated tokens
META = DATA / "meta.csv"           # one row per activation: concept, condition, prompt id, ...
