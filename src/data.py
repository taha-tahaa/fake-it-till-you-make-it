"""
data.py — ASVspoof 2019 LA dataset, protocol parsing, and dataloaders.

The LA protocol labels each utterance with its attack id:
    train/dev -> attacks A01..A06 ("seen")
    eval      -> attacks A07..A19 ("unseen") + bona fide
We parse those files to get the seen/unseen split for free — the core of the
generalization experiment.

Expected directory layout (official LA.zip from Edinburgh DataShare 10283/3336,
downloaded by scripts/download_data.py):

    <root>/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
    <root>/LA/ASVspoof2019_LA_{train,dev,eval}/flac/*.flac
    <root>/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt

Verify an installed dataset with:
    python -m src.data --root <root> --verify
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import worker_init_fn

SEEN_ATTACKS = {f"A0{i}" for i in range(1, 7)}            # A01..A06
UNSEEN_ATTACKS = {f"A{i:02d}" for i in range(7, 20)}      # A07..A19


def la_paths(root: str) -> dict:
    """All official LA paths derived from a single data root."""
    la = Path(root) / "LA"
    proto = la / "ASVspoof2019_LA_cm_protocols"
    return {
        "train_protocol": proto / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol":   proto / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol":  proto / "ASVspoof2019.LA.cm.eval.trl.txt",
        "train_audio": la / "ASVspoof2019_LA_train" / "flac",
        "dev_audio":   la / "ASVspoof2019_LA_dev" / "flac",
        "eval_audio":  la / "ASVspoof2019_LA_eval" / "flac",
        "asv_scores":  la / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
        "subset_dir":  la / "subset_protocols",   # written by scripts/make_subset.py
    }


@dataclass
class Utterance:
    path: str
    label: int          # 1 = bona fide, 0 = spoof
    attack: str         # "-" for bona fide, else A01..A19


def parse_protocol(protocol_file, audio_root) -> list[Utterance]:
    """Parse an ASVspoof2019 .trn/.trl protocol file into Utterance records.

    Each line looks like:  LA_0079 LA_T_1138215 - - bonafide
                           LA_0079 LA_T_1271820 - A01 spoof
    """
    items = []
    for line in Path(protocol_file).read_text().splitlines():
        if not line.strip():
            continue
        spk, utt, _, attack, key = line.split()
        items.append(Utterance(
            path=str(Path(audio_root) / f"{utt}.flac"),
            label=1 if key == "bonafide" else 0,
            attack=attack,
        ))
    return items


def get_split_items(cfg_data: dict, split: str) -> list[Utterance]:
    """Resolve a split ('train'/'dev'/'eval') to Utterance records, honoring the
    optional protocol overrides in the config (used for the fast subset)."""
    paths = la_paths(cfg_data["root"])
    protocol = cfg_data.get(f"{split}_protocol") or paths[f"{split}_protocol"]
    if cfg_data.get(f"{split}_protocol"):          # override is relative to root
        protocol = Path(cfg_data["root"]) / cfg_data[f"{split}_protocol"]
    return parse_protocol(protocol, paths[f"{split}_audio"])


class ASVspoofDataset(Dataset):
    """Returns (waveform, label, attack).

    train=True  -> random 4 s crop (cheap augmentation, varies each epoch)
    train=False -> deterministic CENTER crop, so dev/eval metrics are exactly
                   reproducible run-to-run (a random crop here would make model
                   comparisons noisy).
    Feature extraction lives in the model / features.py so front-ends are swappable.
    """

    def __init__(self, items: list[Utterance], sample_rate: int = 16000,
                 max_seconds: float = 4.0, augment=None, train: bool = False):
        self.items = items
        self.sr = sample_rate
        self.max_len = int(max_seconds * sample_rate)
        self.augment = augment
        self.train = train

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        # Import inside the method (not stored on self): on Windows the DataLoader
        # spawns workers by pickling the dataset, and a module attribute can't be
        # pickled ("cannot pickle 'module' object"). sys.modules makes this cheap.
        import torchaudio
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0)                                  # mono
        if wav.numel() < self.max_len:                     # pad short clips
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.numel()))
        elif self.train:                                   # random crop (train only)
            start = torch.randint(0, wav.numel() - self.max_len + 1, (1,)).item()
            wav = wav[start:start + self.max_len]
        else:                                              # center crop (dev/eval)
            start = (wav.numel() - self.max_len) // 2
            wav = wav[start:start + self.max_len]
        return wav

    def __getitem__(self, i):
        it = self.items[i]
        wav = self._load(it.path)
        if self.train and self.augment is not None:        # never augment dev/eval
            wav = self.augment(wav, self.sr)
        return wav, it.label, it.attack


def make_loader(items, batch_size=32, train=False, augment=None,
                num_workers=2, max_seconds=4.0, **kw):
    ds = ASVspoofDataset(items, max_seconds=max_seconds,
                         augment=augment if train else None, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True,
                      worker_init_fn=worker_init_fn,
                      persistent_workers=num_workers > 0, **kw)


# ------------------------------------------------------------------ verification
def verify(root: str) -> bool:
    """Check that every utterance referenced by each protocol exists on disk and
    report per-split counts. Fails loudly on missing files."""
    paths = la_paths(root)
    ok = True
    for split in ("train", "dev", "eval"):
        protocol, audio = paths[f"{split}_protocol"], paths[f"{split}_audio"]
        if not protocol.exists():
            print(f"[{split}] MISSING protocol: {protocol}")
            ok = False
            continue
        items = parse_protocol(protocol, audio)
        on_disk = {p.name for p in audio.glob("*.flac")} if audio.exists() else set()
        missing = [it for it in items if Path(it.path).name not in on_disk]
        n_bona = sum(it.label for it in items)
        print(f"[{split}] protocol: {len(items):6d} utts "
              f"({n_bona} bona fide / {len(items) - n_bona} spoof) | "
              f"on disk: {len(on_disk):6d} | missing: {len(missing)}")
        if missing:
            print(f"         e.g. {missing[0].path}")
            ok = False
    if not paths["asv_scores"].exists():
        print(f"[asv]   MISSING ASV score file (needed for min t-DCF): {paths['asv_scores']}")
        ok = False
    else:
        print(f"[asv]   ASV score file found.")
    print("VERIFY:", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="./data", help="data root (contains LA/)")
    ap.add_argument("--verify", action="store_true", help="check dataset integrity")
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(0 if verify(args.root) else 1)
    ap.print_help()
