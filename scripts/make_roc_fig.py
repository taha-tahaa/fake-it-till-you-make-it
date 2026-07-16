"""
make_roc_fig.py -- Two-panel figure from saved eval scores:
  (left)  ROC curves on the UNSEEN eval set: CNN baseline vs wav2vec2 + LoRA.
  (right) score distributions for wav2vec2 + LoRA (bona fide vs spoof), showing
          clean separation on attacks it never trained on.

Run:  python scripts/make_roc_fig.py
Output: results/figures/roc_scores.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"

CNN = "baseline_lcnn_mel"     # representative CNN baseline (has local scores)
SSL = "wav2vec2_lora"
CNN_LABEL = "log-mel + LCNN (CNN)"
SSL_LABEL = "wav2vec2 + LoRA (ours)"
CNN_COLOR = "#1f77b4"
SSL_COLOR = "#d62728"


def load(name):
    d = np.load(REPO / "runs" / name / "scores_eval.npz", allow_pickle=True)
    return d["scores"].astype(float), d["labels"].astype(int)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---- left: ROC ----
for name, label, color in [(CNN, CNN_LABEL, CNN_COLOR), (SSL, SSL_LABEL, SSL_COLOR)]:
    s, y = load(name)
    fpr, tpr, _ = roc_curve(y, s)          # y=1 bona fide, higher score = bona fide
    auc = roc_auc_score(y, s)
    # 4 decimals: at 3 dp the SSL model reads "1.000", which overstates a 0.9996 AUC
    # and would not match the number quoted in the report text.
    axL.plot(fpr, tpr, color=color, lw=2.2, label=f"{label}  (AUC={auc:.4f})")
axL.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
axL.set_xlabel("False positive rate", fontsize=11)
axL.set_ylabel("True positive rate", fontsize=11)
axL.set_title("ROC on unseen attacks (A07–A19)", fontsize=12, weight="bold")
axL.legend(loc="lower right", fontsize=9)
axL.set_xlim(0, 1); axL.set_ylim(0, 1.02)

# ---- right: score distributions for the SSL model ----
s, y = load(SSL)
axR.hist(s[y == 1], bins=60, color="#2ca02c", alpha=0.7, density=True, label="bona fide")
axR.hist(s[y == 0], bins=60, color="#d62728", alpha=0.6, density=True, label="spoof (unseen)")
axR.set_xlabel("detector score  (P(bona fide))", fontsize=11)
axR.set_ylabel("density", fontsize=11)
axR.set_title("wav2vec2 + LoRA score separation", fontsize=12, weight="bold")
axR.legend(fontsize=10)

fig.tight_layout()
out = FIG / "roc_scores.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
