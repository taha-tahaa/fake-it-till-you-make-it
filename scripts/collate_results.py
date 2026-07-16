"""
collate_results.py — Read every runs/*/metrics.json and print/write the results
table as Markdown (paste-ready for README and report). Never type numbers by hand.

Usage:
    python scripts/collate_results.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

COLS = [
    ("Model", None),
    ("Trainable params", "trainable_params"),
    ("Dev EER (seen) %", "dev_eer_seen"),
    ("Eval EER (unseen) %", "eval_eer_unseen"),
    ("Gap (pp)", "generalization_gap"),
    ("min t-DCF", "eval_min_tdcf"),
]


def fmt(key, v):
    if v is None:
        return "—"
    if key == "trainable_params":
        return f"{v/1e6:.2f} M"
    if key == "eval_min_tdcf":
        return f"{v:.4f}"
    return f"{v:.2f}"


def main():
    rows = []
    for mfile in sorted((REPO / "runs").glob("*/metrics.json")):
        m = json.loads(mfile.read_text())
        if m.get("subset"):
            continue                    # never report subset numbers
        rows.append([mfile.parent.name] + [fmt(k, m.get(k)) for _, k in COLS[1:]])

    if not rows:
        print("no completed runs found under runs/")
        return
    header = "| " + " | ".join(c for c, _ in COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    table = "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in rows])
    print(table)
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    (out / "RESULTS.md").write_text("# Results\n\n" + table + "\n")
    print(f"\nwritten to {out/'RESULTS.md'}")


if __name__ == "__main__":
    main()
