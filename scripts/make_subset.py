"""
make_subset.py — Write balanced fast-iteration subset protocols.

Keeps ALL bona fide utterances (they are the minority class) and stratified-
samples spoof utterances per attack, writing standard-format protocol files to
<root>/LA/subset_protocols/{train,dev}.txt. Loaders are unchanged — training
just points at the smaller protocol via `--subset`.

Final reported numbers always come from the FULL protocols; the subset exists
only to make debugging and hyperparameter sanity checks cheap.

Usage:
    python scripts/make_subset.py --root D:\asvspoof --train-spoof 8000 --dev-spoof 4000
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import la_paths  # noqa: E402


def subsample(protocol_path: Path, n_spoof: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    bona, by_attack = [], defaultdict(list)
    for line in protocol_path.read_text().splitlines():
        if not line.strip():
            continue
        attack, key = line.split()[3], line.split()[4]
        (bona if key == "bonafide" else by_attack[attack]).append(line)
    per_attack = max(1, n_spoof // max(len(by_attack), 1))
    spoof = []
    for a, lines in sorted(by_attack.items()):
        spoof += rng.sample(lines, min(per_attack, len(lines)))
    out = bona + spoof
    rng.shuffle(out)
    print(f"{protocol_path.name}: kept {len(bona)} bona fide + {len(spoof)} spoof "
          f"({per_attack}/attack x {len(by_attack)} attacks)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--train-spoof", type=int, default=8000)
    ap.add_argument("--dev-spoof", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    paths = la_paths(args.root)
    out_dir = paths["subset_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.train_spoof), ("dev", args.dev_spoof)):
        lines = subsample(paths[f"{split}_protocol"], n, args.seed)
        (out_dir / f"{split}.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}\\train.txt, dev.txt — use with:  python -m src.train --subset ...")


if __name__ == "__main__":
    main()
