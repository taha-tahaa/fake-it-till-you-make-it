"""
eval.py — Evaluation with the seen-vs-unseen breakdown that defines this project.

Scores the dev set (seen attacks A01-A06) and the eval set (unseen A07-A19),
reports EER seen / EER unseen / generalization gap, per-attack EER, min t-DCF
(official ASV scores), and exports embeddings + a t-SNE plot.

All raw scores are saved to runs/<name>/scores_<split>.npz and summary numbers
to runs/<name>/metrics.json so every table/figure is regenerable without a GPU.

Usage:
    python -m src.eval --config configs/wav2vec2_lora.yaml --ckpt runs/wav2vec2_lora/best.pt \
                       --report per_attack --tsne
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import get_split_items, make_loader, la_paths, UNSEEN_ATTACKS
from .models import build_model
from .losses import build_criterion
from .metrics import compute_eer, compute_min_tdcf, compute_auc_acc
from .utils import load_config, save_metrics, get_device


@torch.no_grad()
def collect_scores(model, criterion, loader, device, use_amp=False, keep_emb=False):
    """Run the model over a loader. Returns (scores, labels, attacks, embeddings).
    Scoring rule comes from the criterion (sigmoid for BCE, cosine for OC-Softmax)."""
    model.eval()
    s, y, atk, embs = [], [], [], []
    for wav, label, attack in loader:
        with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
            logits, emb = model(wav.to(device, non_blocking=True))
        s += criterion.score(logits.float(), emb.float()).cpu().tolist()
        y += label.tolist()
        atk += list(attack)
        if keep_emb:
            embs.append(emb.float().cpu())
    embs = torch.cat(embs).numpy() if embs else None
    return np.array(s), np.array(y), np.array(atk), embs


def per_attack_eer(scores, labels, attacks):
    """EER for each spoof attack vs. the shared bona fide pool."""
    bona = labels == 1
    out = {}
    for a in sorted(set(attacks[labels == 0])):
        mask = bona | (attacks == a)
        out[a] = compute_eer(scores[mask], labels[mask]) * 100
    return out


def save_tsne(emb, labels, attacks, out_png, max_points=4000, seed=0):
    """t-SNE of embeddings (Tutorial 9 uses sklearn TSNE the same way), colored
    bona fide vs spoof; spoof points marked seen/unseen by attack id."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(emb))[:max_points]
    emb, labels, attacks = emb[idx], labels[idx], attacks[idx]
    perplexity = min(30, max(5, len(emb) - 1))     # must be < n_samples
    z = TSNE(n_components=2, init="pca", perplexity=perplexity,
             random_state=seed).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    unseen = np.isin(attacks, list(UNSEEN_ATTACKS))
    groups = [("bona fide", labels == 1, "tab:green", "o"),
              ("spoof (seen)", (labels == 0) & ~unseen, "tab:orange", "^"),
              ("spoof (unseen)", (labels == 0) & unseen, "tab:red", "x")]
    for name, m, c, mk in groups:
        if m.any():
            ax.scatter(z[m, 0], z[m, 1], s=6, c=c, marker=mk, alpha=0.5, label=name)
    ax.legend(markerscale=2)
    ax.set_title("t-SNE of detector embeddings")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved {out_png}")


def main(args):
    cfg = load_config(args.config, args.data_root)
    device = get_device()
    run = Path(cfg["train"]["out"])

    # --- model + criterion, STRICT load (a silent mismatch would mean quietly
    # evaluating a randomly initialized model) ---
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, model, 1, 1).to(device)   # pos_weight irrelevant at eval
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    criterion.load_state_dict(state["criterion"], strict=True)
    print(f"loaded {args.ckpt} (epoch {state['epoch']})")

    lkw = dict(num_workers=cfg["data"].get("num_workers", 2),
               max_seconds=cfg["data"].get("max_seconds", 4.0))
    bs = cfg["train"]["batch_size"]
    results = {}

    # --- dev = seen attacks ---
    if args.split in ("both", "dev"):
        items = get_split_items(cfg["data"], "dev")
        scores, labels, attacks, _ = collect_scores(
            model, criterion, make_loader(items, bs, **lkw), device)
        results["dev_eer_seen"] = compute_eer(scores, labels) * 100
        np.savez(run / "scores_dev.npz", scores=scores, labels=labels, attacks=attacks)
        print(f"Dev EER (seen A01-A06):    {results['dev_eer_seen']:.2f}%")

    # --- eval = unseen attacks ---
    if args.split in ("both", "eval"):
        items = get_split_items(cfg["data"], "eval")
        scores, labels, attacks, emb = collect_scores(
            model, criterion, make_loader(items, bs, **lkw), device, keep_emb=args.tsne)
        results["eval_eer_unseen"] = compute_eer(scores, labels) * 100
        auc, acc = compute_auc_acc(scores, labels)
        results["eval_auc"], results["eval_acc"] = auc, acc * 100
        np.savez(run / "scores_eval.npz", scores=scores, labels=labels, attacks=attacks)
        print(f"Eval EER (unseen A07-A19): {results['eval_eer_unseen']:.2f}%")

        asv = la_paths(cfg["data"]["root"])["asv_scores"]
        if asv.exists():
            results["eval_min_tdcf"] = compute_min_tdcf(scores, labels, asv)
            print(f"Eval min t-DCF:            {results['eval_min_tdcf']:.4f}")
        else:
            print("(min t-DCF skipped — ASV score file not found)")

        if args.report == "per_attack":
            pa = per_attack_eer(scores, labels, attacks)
            results["per_attack_eer"] = pa
            print("\nPer-attack EER (%):")
            for a, e in pa.items():
                print(f"  {a} (unseen): {e:.2f}")

        if args.tsne:
            save_tsne(emb, labels, attacks, run / "tsne_eval.png")

    if "dev_eer_seen" in results and "eval_eer_unseen" in results:
        results["generalization_gap"] = results["eval_eer_unseen"] - results["dev_eer_seen"]
        print(f"\nGeneralization gap (unseen - seen): {results['generalization_gap']:.2f} pp")

    save_metrics(run, **results)
    print(f"metrics written to {run/'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["both", "dev", "eval"], default="both")
    ap.add_argument("--report", choices=["basic", "per_attack"], default="per_attack")
    ap.add_argument("--tsne", action="store_true", help="export t-SNE of eval embeddings")
    ap.add_argument("--data-root", default=None)
    main(ap.parse_args())
