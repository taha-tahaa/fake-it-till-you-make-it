"""
train.py — Config-driven training loop.

Features: AMP mixed precision, gradient accumulation + clipping, linear warmup ->
cosine annealing (Tutorial 3: LR scheduling), early stopping on dev EER,
resumable checkpoints (Colab-preemption-safe), per-epoch CSV logging, full
reproducibility seeding.

Usage:
    python -m src.train --config configs/wav2vec2_lora.yaml
    python -m src.train --config ... --data-root D:\asvspoof --resume
    python -m src.train --config ... --subset          # fast-iteration subset
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
import yaml

from .data import get_split_items, make_loader, la_paths
from .models import build_model, count_trainable
from .losses import build_criterion
from .augment import Augmenter
from .eval import collect_scores
from .metrics import compute_eer
from .utils import set_seed, load_config, save_metrics, get_device


def build_optimizer(model, criterion, tcfg):
    """AdamW with param groups: LoRA/DoRA adapter weights can use a different LR
    (lr_backbone) than the pool/head; OC-Softmax center trains with the head."""
    lora, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (lora if "lora_" in n else rest).append(p)
    rest += [p for p in criterion.parameters() if p.requires_grad]
    groups = [{"params": rest, "lr": tcfg["lr"]}]
    if lora:
        groups.append({"params": lora, "lr": tcfg.get("lr_backbone", tcfg["lr"])})
    return torch.optim.AdamW(groups, weight_decay=tcfg.get("weight_decay", 1e-4))


def build_scheduler(opt, tcfg):
    warm, total = tcfg.get("warmup_epochs", 0), tcfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warm:
            return (epoch + 1) / max(warm, 1)
        t = (epoch - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def save_ckpt(path, model, criterion, opt, sched, scaler, epoch, best_eer, cfg):
    torch.save({
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_eer": best_eer,
        "config": cfg,
    }, path)


def main(args):
    cfg = load_config(args.config, args.data_root)
    tcfg, dcfg = cfg["train"], cfg["data"]
    set_seed(cfg.get("seed", 1234), deterministic=cfg.get("deterministic", False))
    device = get_device()

    # --- data ---
    if args.subset:                              # fast-iteration protocols (make_subset.py)
        sub = la_paths(dcfg["root"])["subset_dir"]
        dcfg["train_protocol"] = str(Path(sub.relative_to(dcfg["root"])) / "train.txt")
        dcfg["dev_protocol"] = str(Path(sub.relative_to(dcfg["root"])) / "dev.txt")
    train_items = get_split_items(dcfg, "train")
    dev_items = get_split_items(dcfg, "dev")
    aug = Augmenter(mode=dcfg.get("augment", "none"), p=dcfg.get("augment_p", 0.5))
    lkw = dict(num_workers=dcfg.get("num_workers", 2),
               max_seconds=dcfg.get("max_seconds", 4.0))
    train_loader = make_loader(train_items, tcfg["batch_size"], train=True, augment=aug, **lkw)
    dev_loader = make_loader(dev_items, tcfg["batch_size"], **lkw)

    # --- model / loss / optim ---
    model = build_model(cfg).to(device)
    n_pos = sum(it.label for it in train_items)          # bona fide (minority)
    n_neg = len(train_items) - n_pos
    criterion = build_criterion(cfg, model, n_pos, n_neg).to(device)
    opt = build_optimizer(model, criterion, tcfg)
    sched = build_scheduler(opt, tcfg)
    use_amp = tcfg.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = tcfg.get("grad_accum", 1)
    clip = tcfg.get("grad_clip", 1.0)

    run = Path(tcfg["out"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"device={device} amp={use_amp} | train {len(train_items)} "
          f"({n_pos} bona/{n_neg} spoof) dev {len(dev_items)} | "
          f"trainable params: {count_trainable(model)/1e6:.3f} M")

    # --- resume ---
    start_epoch, best, bad_epochs = 0, 1.0, 0
    last = run / "last.pt"
    if args.resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        criterion.load_state_dict(state["criterion"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best_eer"]
        print(f"resumed from epoch {state['epoch']} (best dev EER {best*100:.2f}%)")

    log_path = run / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,dev_eer,lr,seconds\n")

    from tqdm import tqdm
    epoch = start_epoch - 1                      # in case training already finished
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, total_loss, n_batches = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (wav, y, _) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, emb = model(wav)
                loss = criterion(logits, emb, y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * accum
            n_batches += 1

        # --- dev evaluation (seen attacks) ---
        scores, labels, _, _ = collect_scores(model, criterion, dev_loader, device, use_amp)
        eer = compute_eer(scores, labels)
        lr_now = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoch {epoch}  loss={total_loss/max(n_batches,1):.4f}  "
              f"dev EER (seen)={eer*100:.2f}%  lr={lr_now:.2e}  [{secs:.0f}s]")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{total_loss/max(n_batches,1):.5f}",
                                    f"{eer:.5f}", f"{lr_now:.2e}", f"{secs:.0f}"])
        sched.step()

        # Update `best` BEFORE saving last.pt, so last.pt always carries the true
        # running best. (Otherwise a resume from last.pt resets best-tracking and
        # can overwrite best.pt with a worse model — critical on Colab, which
        # resumes after every disconnect.)
        improved = eer < best
        if improved:
            best, bad_epochs = eer, 0
        else:
            bad_epochs += 1
        save_ckpt(last, model, criterion, opt, sched, scaler, epoch, best, cfg)
        if improved:
            save_ckpt(run / "best.pt", model, criterion, opt, sched, scaler, epoch, best, cfg)
            print(f"  -> new best, saved {run/'best.pt'}")
        elif bad_epochs >= tcfg.get("patience", 999):    # early stopping (Tutorial 8 spirit)
            print(f"early stop: no dev improvement for {bad_epochs} epochs")
            break

    if not (run / "best.pt").exists():
        # dev EER never improved past the initial 100% -> training diverged
        # (e.g. NaN loss). Fail loudly here instead of letting eval.py crash
        # two steps later on a misleading FileNotFoundError.
        raise RuntimeError(
            f"{run}: no checkpoint was ever saved (dev EER never improved). "
            f"Check {run/'log.csv'} for NaN losses — training likely diverged.")

    save_metrics(run, dev_eer_seen=best * 100,
                 trainable_params=count_trainable(model),
                 epochs_ran=epoch + 1, subset=bool(args.subset))
    print(f"Best dev EER (seen): {best*100:.2f}%  ->  {run/'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="override data.root in the config")
    ap.add_argument("--resume", action="store_true", help="resume from last.pt")
    ap.add_argument("--subset", action="store_true",
                    help="train on the fast subset protocols (scripts/make_subset.py)")
    main(ap.parse_args())
