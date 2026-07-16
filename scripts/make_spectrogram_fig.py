"""
make_spectrogram_fig.py -- Motivation figure: a real bona fide clip next to a real
spoof clip (a high-quality unseen attack) from ASVspoof 2019 LA, as log-mel
spectrograms. The point: they look almost identical to the eye -- which is why a
naive detector overfits and fails on new attack types.

Run:  python scripts/make_spectrogram_fig.py --root C:\asvspoof
Output: results/figures/spectrogram_compare.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from src.data import la_paths, parse_protocol  # noqa: E402


def logmel(wav, sr=16000):
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=160,
                                              n_mels=80)(wav)
    return m.clamp_min(1e-9).log().numpy()


def load_clip(path, sr=16000, seconds=3.0):
    import torchaudio
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    wav = wav.mean(0)
    n = int(seconds * sr)
    wav = wav[:n] if wav.numel() >= n else torch.nn.functional.pad(wav, (0, n - wav.numel()))
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--attack", default="A17", help="which unseen spoof attack to show")
    args = ap.parse_args()

    p = la_paths(args.root)
    items = parse_protocol(p["eval_protocol"], p["eval_audio"])
    bona = next(it for it in items if it.label == 1)
    spoof = next(it for it in items if it.attack == args.attack)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, it, title, color in [
        (axes[0], bona, "Bona fide (real human)", "#2ca02c"),
        (axes[1], spoof, f"Spoof (unseen attack {args.attack})", "#d62728"),
    ]:
        S = logmel(load_clip(it.path))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 3, 0, 8])
        ax.set_title(title, fontsize=12, color=color, weight="bold")
        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_yticks([])
    axes[0].set_ylabel("mel frequency", fontsize=10)
    fig.suptitle("Can you tell which is fake?  (log-mel spectrograms)",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    out = REPO / "results" / "figures" / "spectrogram_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out, "\nbona:", Path(bona.path).name, "| spoof:", Path(spoof.path).name)


if __name__ == "__main__":
    main()
