"""
utils.py — Seeding, config loading, and run-directory helpers shared by
train.py / eval.py / scripts.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False):
    """Seed python / numpy / torch (+ CUDA). With deterministic=True we also pin
    cuDNN kernels — exact repeatability at a small speed cost."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """Give each DataLoader worker a distinct but reproducible seed (otherwise all
    workers would apply identical 'random' crops/augmentations)."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str, data_root: str | None = None) -> dict:
    """Load a YAML experiment config. `data_root` (CLI --data-root) overrides
    data.root so the 24 GB dataset can live outside the repo / OneDrive."""
    cfg = yaml.safe_load(Path(path).read_text())
    if data_root:
        cfg.setdefault("data", {})["root"] = data_root
    return cfg


def save_metrics(run_dir, **kv) -> dict:
    """Merge key/values into runs/<name>/metrics.json. All tables and figures are
    regenerated from these files, never typed by hand."""
    f = Path(run_dir) / "metrics.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur.update(kv)
    f.write_text(json.dumps(cur, indent=2))
    return cur


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
