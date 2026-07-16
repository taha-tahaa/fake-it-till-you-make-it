"""
run_all.py — Run every experiment (train + eval) in priority order and collate
the results table. Cross-platform replacement for the old run_all.sh.

Skips a config if its runs/<name>/metrics.json already contains eval results,
so the script is safely re-runnable after interruptions (Colab disconnects).

Usage:
    python scripts/run_all.py [--data-root D:\asvspoof] [--subset] [--only wav2vec2_lora ...]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Priority order: guaranteed core first (P0), then ablations (P1/P2) — if time
# or GPU quota runs out, the most important results already exist.
ORDER = [
    "baseline_lcnn_lfcc",       # P0 baseline
    "wav2vec2_lora",            # P0 main model
    "wav2vec2_probe",           # P1 PEFT ablation (lower bound)
    "wav2vec2_dora",            # P1 PEFT ablation
    "baseline_lcnn_lfcc_aug",   # P1 augmentation ablation (CNN)
    "wav2vec2_lora_noaug",      # P1 augmentation ablation (SSL)
    "wav2vec2_lora_ocsoftmax",  # P2 loss ablation
    "baseline_lcnn_mel",        # P2 front-end ablation
]


def sh(cmd: list[str]) -> int:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--subset", action="store_true", help="fast subset run (debug only)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiment names")
    args = ap.parse_args()

    extra = ["--data-root", args.data_root] if args.data_root else []
    names = args.only or ORDER
    for name in names:
        cfg = REPO / "configs" / f"{name}.yaml"
        if not cfg.exists():
            print(f"!! missing config {cfg}, skipping")
            continue
        mfile = REPO / "runs" / name / "metrics.json"
        if mfile.exists() and "eval_eer_unseen" in json.loads(mfile.read_text()):
            print(f"== {name}: already evaluated, skipping (delete {mfile} to redo)")
            continue
        train_cmd = [sys.executable, "-m", "src.train", "--config", str(cfg),
                     "--resume"] + extra + (["--subset"] if args.subset else [])
        if sh(train_cmd) != 0:
            print(f"!! {name}: training failed, aborting run_all")
            sys.exit(1)
        eval_cmd = [sys.executable, "-m", "src.eval", "--config", str(cfg),
                    "--ckpt", str(REPO / "runs" / name / "best.pt"),
                    "--report", "per_attack", "--tsne"] + extra
        if sh(eval_cmd) != 0:
            print(f"!! {name}: eval failed, aborting run_all")
            sys.exit(1)

    sh([sys.executable, str(REPO / "scripts" / "collate_results.py")])


if __name__ == "__main__":
    main()
