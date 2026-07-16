"""
make_gap_animation.py -- Animated GIF of the generalization gap.

Shows the four main models. First the "seen" (dev) EER bars appear (all near zero);
then the "unseen" (eval) EER bars grow -- the CNN family shoots up while
wav2vec2 + LoRA barely moves, making the generalization gap visceral.

Run:  python scripts/make_gap_animation.py
Output: results/figures/gap_animation.gif
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures" / "gap_animation.gif"

labels = ["LFCC\n+ LCNN", "log-mel\n+ LCNN", "LFCC + LCNN\n+ RawBoost", "wav2vec2\n+ LoRA (ours)"]
seen   = np.array([0.23, 0.00, 0.31, 0.16])
unseen = np.array([15.36, 10.22, 9.76, 0.80])
colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]

x = np.arange(len(labels))
W = 0.38
HOLD, RISE, END = 12, 40, 25          # frames: hold seen, rise unseen, hold end
TOTAL = HOLD + RISE + END

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_ylim(0, 17); ax.set_xlim(-0.6, len(labels) - 0.4)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("EER (%)  —  lower is better", fontsize=11)
ax.set_title("Generalization gap: seen vs. unseen spoofing attacks", fontsize=12, weight="bold")

seen_bars = ax.bar(x - W/2, np.zeros_like(seen), W, color="#9ecae1", label="seen (dev)")
unseen_bars = ax.bar(x + W/2, np.zeros_like(unseen), W, color=colors, label="UNSEEN (eval)")
ax.legend(loc="upper right", fontsize=9)
val_txt = [ax.text(xi + W/2, 0, "", ha="center", va="bottom", fontsize=9) for xi in x]


def ease(t):                       # smoothstep for a pleasant rise
    return t * t * (3 - 2 * t)


def update(frame):
    # phase 1: seen bars appear
    if frame < HOLD:
        f = ease((frame + 1) / HOLD)
        for b, h in zip(seen_bars, seen):
            b.set_height(h * f)
    # phase 2: unseen bars rise
    elif frame < HOLD + RISE:
        for b, h in zip(seen_bars, seen):
            b.set_height(h)
        f = ease((frame - HOLD + 1) / RISE)
        for b, h, t in zip(unseen_bars, unseen, val_txt):
            b.set_height(h * f)
            t.set_text(f"{h*f:.1f}")
            t.set_y(h * f + 0.2)
    # phase 3: hold final
    else:
        for b, h in zip(seen_bars, seen):
            b.set_height(h)
        for b, h, t in zip(unseen_bars, unseen, val_txt):
            b.set_height(h); t.set_text(f"{h:.1f}"); t.set_y(h + 0.2)
    return list(seen_bars) + list(unseen_bars) + val_txt


anim = FuncAnimation(fig, update, frames=TOTAL, interval=60, blit=False)
anim.save(str(OUT), writer=PillowWriter(fps=18))
print("wrote", OUT)
