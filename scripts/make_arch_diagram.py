"""
make_arch_diagram.py -- Draw the two-architecture comparison figure.

Left:  classical baseline  waveform -> LFCC/log-mel -> LCNN -> score
Right: ours                waveform -> FROZEN wav2vec2 (+LoRA) -> pooling -> score

Trainable blocks are colored; the frozen wav2vec2 backbone is drawn in gray with a
snowflake marker. Output: results/figures/architecture.png

Run:  python scripts/make_arch_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures" / "architecture.png"

INK   = "#1d1d1f"
GRAY  = "#8e8e93"
FROZE = "#c7c7cc"   # frozen block fill
BLUE  = "#0066cc"   # trainable (ours)
ORANGE = "#e8833a"  # trainable (baseline)
LIGHTB = "#d9e8fb"
LIGHTO = "#fbe6d4"


def box(ax, x, y, w, h, text, face, edge, fontcolor=INK, fs=10, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=fontcolor, weight="bold" if bold else "normal", zorder=3)


def arrow(ax, x, y0, y1, color=GRAY):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color, zorder=1))


# Canvas widened to x=11.5 so the "+ LoRA" callout (spans x=9.35..10.9) is not
# clipped by the axes; ylim raised slightly for the input label + arrows.
fig, ax = plt.subplots(figsize=(11.4, 7.2))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 10.6); ax.axis("off")

# ---- column headers ----
ax.text(2.5, 10.0, "(A) Classical baseline", ha="center", fontsize=13, weight="bold", color=INK)
ax.text(7.5, 10.0, "(B) Self-supervised (ours)", ha="center", fontsize=13, weight="bold", color=BLUE)

# shared input label
ax.text(5.0, 9.5, "16 kHz speech waveform", ha="center", fontsize=10, style="italic", color=GRAY)

LW = 3.4   # box width
LX = 0.8   # left column x
RX = 5.8   # right column x

# ---------- LEFT (baseline) ----------
ys = [8.1, 6.7, 5.3, 3.9, 2.5, 1.1]
box(ax, LX, ys[0], LW, 0.9, "LFCC / log-mel\nspectrogram", LIGHTO, ORANGE, fs=10)
box(ax, LX, ys[1], LW, 0.9, "Conv + MFM + Pool\n(x4 blocks)", LIGHTO, ORANGE, fs=10)
box(ax, LX, ys[2], LW, 0.9, "Global average pool", LIGHTO, ORANGE, fs=10)
box(ax, LX, ys[3], LW, 0.9, "128-d embedding proj\n+ linear head", LIGHTO, ORANGE, fs=10)
box(ax, LX, ys[4], LW, 0.8, "bona fide / spoof score", "#ffffff", INK, fs=10, bold=True)
for a, b in zip(ys[:4], ys[1:5]):
    arrow(ax, LX + LW/2, a - 0.05, b + 0.85, ORANGE)
ax.text(LX + LW/2, 0.55, "0.41 M trainable params\n(trained from scratch)",
        ha="center", fontsize=9, color=GRAY)

# ---------- RIGHT (ours) ----------
box(ax, RX, ys[0], LW, 0.9, "wav2vec 2.0 CNN encoder", FROZE, GRAY, fontcolor=INK, fs=10)
box(ax, RX, ys[1], LW, 0.9, "wav2vec 2.0 Transformer\n(12 layers)  ❄ FROZEN", FROZE, GRAY, fontcolor=INK, fs=10)
# LoRA callout attached to the transformer block
box(ax, RX + LW + 0.15, ys[1] + 0.05, 1.55, 0.8, "+ LoRA\nr=8", LIGHTB, BLUE, fontcolor=BLUE, fs=9, bold=True)
ax.add_patch(FancyArrowPatch((RX + LW + 0.15, ys[1] + 0.45), (RX + LW, ys[1] + 0.45),
                             arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=BLUE))
box(ax, RX, ys[2], LW, 0.9, "Attentive mean pooling", LIGHTB, BLUE, fs=10)
box(ax, RX, ys[3], LW, 0.9, "Dropout + Linear head", LIGHTB, BLUE, fs=10)
box(ax, RX, ys[4], LW, 0.8, "bona fide / spoof score", "#ffffff", INK, fs=10, bold=True)
for a, b in zip(ys[:4], ys[1:5]):
    arrow(ax, RX + LW/2, a - 0.05, b + 0.85, BLUE)
ax.text(RX + LW/2, 0.55, "0.59 M trainable params  (0.62% of 95 M)\nfrozen backbone + LoRA adapters",
        ha="center", fontsize=9, color=GRAY)

# input arrows from the shared label down into both columns (long enough to read)
arrow(ax, LX + LW/2, 9.35, ys[0] + 0.92, GRAY)
arrow(ax, RX + LW/2, 9.35, ys[0] + 0.92, GRAY)

# legend
ax.add_patch(FancyBboxPatch((0.8, 0.02), 0.32, 0.3, boxstyle="round,pad=0.01", facecolor=LIGHTB, edgecolor=BLUE))
ax.text(1.2, 0.17, "trainable", fontsize=8.5, va="center", color=INK)
ax.add_patch(FancyBboxPatch((2.5, 0.02), 0.32, 0.3, boxstyle="round,pad=0.01", facecolor=FROZE, edgecolor=GRAY))
ax.text(2.9, 0.17, "❄ frozen (pretrained)", fontsize=8.5, va="center", color=INK)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
