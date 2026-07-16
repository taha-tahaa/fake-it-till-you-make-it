"""
make_figures.py — Regenerate every report/presentation figure from saved run
artifacts (runs/*/metrics.json, scores_*.npz, log.csv). No GPU needed.

Figures (written to results/figures/):
  gap_bars.png        seen vs unseen EER per model  — the money plot
  per_attack.png      per-attack EER bars, models side by side
  efficiency.png      trainable params vs unseen EER (PEFT story)
  training_curves.png dev EER vs epoch for each run

Usage:
    python scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures"

PRETTY = {  # run-dir name -> display label
    "baseline_lcnn_lfcc": "LFCC + LCNN",
    "baseline_lcnn_mel": "log-mel + LCNN",
    "baseline_lcnn_lfcc_aug": "LFCC + LCNN + RawBoost",
    "wav2vec2_probe": "wav2vec2 (frozen) probe",
    "wav2vec2_lora": "wav2vec2 + LoRA",
    "wav2vec2_dora": "wav2vec2 + DoRA",
    "wav2vec2_lora_noaug": "wav2vec2 + LoRA (no aug)",
    "wav2vec2_lora_ocsoftmax": "wav2vec2 + LoRA + OC-Softmax",
}


def load_runs():
    runs = {}
    for mfile in sorted((REPO / "runs").glob("*/metrics.json")):
        m = json.loads(mfile.read_text())
        if not m.get("subset") and "eval_eer_unseen" in m:
            runs[mfile.parent.name] = m
    return runs


def gap_bars(runs):
    names = list(runs)
    labels = [PRETTY.get(n, n) for n in names]
    seen = [runs[n].get("dev_eer_seen", np.nan) for n in names]
    unseen = [runs[n]["eval_eer_unseen"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4.5))
    ax.bar(x - 0.2, seen, 0.4, label="seen attacks (dev, A01–A06)", color="tab:blue")
    ax.bar(x + 0.2, unseen, 0.4, label="UNSEEN attacks (eval, A07–A19)", color="tab:red")
    for xi, (s, u) in zip(x, zip(seen, unseen)):
        ax.text(xi + 0.2, u + 0.3, f"{u:.1f}", ha="center", fontsize=9)
        if not np.isnan(s):
            ax.text(xi - 0.2, s + 0.3, f"{s:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("EER (%)  —  lower is better")
    ax.set_title("Generalization gap: seen vs unseen spoofing attacks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gap_bars.png", dpi=200)
    plt.close(fig)


def per_attack(runs, focus=("baseline_lcnn_lfcc", "wav2vec2_lora")):
    present = [n for n in focus if n in runs and "per_attack_eer" in runs[n]]
    if not present:
        return
    attacks = sorted(runs[present[0]]["per_attack_eer"])
    x = np.arange(len(attacks))
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, n in enumerate(present):
        vals = [runs[n]["per_attack_eer"].get(a, np.nan) for a in attacks]
        ax.bar(x + (i - len(present) / 2 + 0.5) * w, vals, w, label=PRETTY.get(n, n))
    ax.set_xticks(x, attacks)
    ax.set_ylabel("EER (%)")
    ax.set_title("Per-attack EER on unseen attacks (A07–A19)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "per_attack.png", dpi=200)
    plt.close(fig)


def efficiency(runs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, m in runs.items():
        if "trainable_params" in m:
            ax.scatter(m["trainable_params"] / 1e6, m["eval_eer_unseen"], s=60)
            ax.annotate(PRETTY.get(n, n), (m["trainable_params"] / 1e6,
                        m["eval_eer_unseen"]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (M, log scale)")
    ax.set_ylabel("Unseen-attack EER (%)")
    ax.set_title("Adaptation efficiency: parameters vs generalization")
    fig.tight_layout()
    fig.savefig(OUT / "efficiency.png", dpi=200)
    plt.close(fig)


def training_curves():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for log in sorted((REPO / "runs").glob("*/log.csv")):
        rows = list(csv.DictReader(log.open()))
        if not rows:
            continue
        ax.plot([int(r["epoch"]) for r in rows],
                [100 * float(r["dev_eer"]) for r in rows],
                marker="o", ms=3, label=PRETTY.get(log.parent.name, log.parent.name))
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev EER (%)")
    ax.set_title("Training progress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if runs:
        gap_bars(runs)
        per_attack(runs)
        efficiency(runs)
    training_curves()
    print(f"figures written to {OUT} ({len(runs)} completed runs found)")
