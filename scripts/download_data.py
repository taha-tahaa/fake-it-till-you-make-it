r"""
download_data.py — Fetch ASVspoof 2019 LA and lay it out for src/data.py.

Source: the Kaggle mirror `awsaf49/asvpoof-2019-dataset` via the `kagglehub`
library, which hosts the full official LA (+PA) tree — flac, cm protocols, AND
the ASV score files needed for min t-DCF. We use Kaggle because the official
Edinburgh DataShare server refuses programmatic downloads (HTTP 403 /
connection reset), both on Colab and off-campus.

One-time Kaggle auth (free, ~1 minute) — new token-based flow:
  1. Sign in at https://www.kaggle.com/settings -> API -> "Create New Token".
     (If your account issues the newer bearer token starting with "KGAT_",
     follow steps 2-3 below instead of downloading kaggle.json.)
  2. Save it to ~/.kaggle/access_token:
       Windows : C:\Users\<you>\.kaggle\access_token   (file contains just the token)
       Colab   : set the KAGGLE_API_TOKEN env var, or upload the file (see notebook)
  3. Nothing else to configure — kagglehub reads it automatically.
  (Classic kaggle.json with username+key also still works if you have one.)

Usage (identical on Windows / Colab / Linux):
    python scripts/download_data.py --out C:\asvspoof     # local (needs ~50 GB free)
    python scripts/download_data.py --out ./data          # Colab

Disk: the archive is ~23.6 GB; kagglehub caches it under ~/.cache/kagglehub and
extracts alongside, so budget ~50 GB free. Re-running resumes/reuses the cache
and is a no-op once <out>/LA exists.

Manual fallback (no Kaggle account): download LA.zip by hand from
https://datashare.ed.ac.uk/handle/10283/3336 in a browser, unzip it so that
<out>/LA/ASVspoof2019_LA_cm_protocols exists, then run this script — it will
skip the download and just verify.
"""
import argparse
import shutil
import sys
from pathlib import Path


def find_la_dir(base: Path) -> Path | None:
    """Locate the folder that directly contains the cm protocols (i.e. the 'LA'
    dir), wherever the archive happened to nest it."""
    hits = list(base.rglob("ASVspoof2019_LA_cm_protocols"))
    return hits[0].parent if hits else None


def normalize_layout(src: Path, out: Path):
    """Ensure <out>/LA is the folder holding ASVspoof2019_LA_*. kagglehub's
    cache copy may nest it under an extra 'archive' or dataset-name folder."""
    target = out / "LA"
    if (target / "ASVspoof2019_LA_cm_protocols").exists():
        return
    la = find_la_dir(src)
    if la is None:
        raise FileNotFoundError(
            f"Could not find ASVspoof2019_LA_cm_protocols anywhere under {src}. "
            "The archive layout was unexpected.")
    if target.exists():
        shutil.rmtree(target)
    print(f"copying dataset into place: {la}  ->  {target}  (one-time; a few minutes)")
    shutil.copytree(la, target)


def kaggle_download(out: Path, max_retries: int = 8) -> Path:
    try:
        import kagglehub
    except ImportError:
        import subprocess
        print("installing the kagglehub package ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub

    from kagglehub.config import get_kaggle_credentials
    if get_kaggle_credentials() is None:
        print("ERROR: no Kaggle credentials found.\n"
              "       Create a token at https://www.kaggle.com/settings (API -> "
              "Create New Token).\n"
              "       New bearer-token accounts: save it to ~/.kaggle/access_token "
              "(just the token, no quotes).\n"
              "       Classic accounts: save kaggle.json to ~/.kaggle/kaggle.json.\n"
              "       See this script's docstring for exact paths.", file=sys.stderr)
        sys.exit(2)

    print("Downloading awsaf49/asvpoof-2019-dataset (~23.6 GB) — this takes a while.")
    # Home connections drop mid-transfer; kagglehub caches partial bytes and
    # resumes via a Range request, so we just retry the call on network errors.
    import time
    for attempt in range(1, max_retries + 1):
        try:
            return Path(kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset"))
        except Exception as e:
            name = type(e).__name__
            transient = any(k in name for k in
                            ("ChunkedEncoding", "Protocol", "Connection", "Timeout",
                             "IncompleteRead", "SSL"))
            if not transient or attempt == max_retries:
                raise
            wait = min(30, 5 * attempt)
            print(f"\n[attempt {attempt}] network error ({name}); resuming in {wait}s ...")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./data", help="data root (LA/ will live inside)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if "onedrive" in str(out).lower():
        print("WARNING: the data root is inside OneDrive — tens of GB would sync to the "
              "cloud.\n         Prefer e.g. --out C:\\asvspoof (pass --data-root to train/eval).")

    if (out / "LA" / "ASVspoof2019_LA_cm_protocols").exists():
        print(f"{out/'LA'} already present — skipping download.")
    else:
        cache_path = kaggle_download(out)
        normalize_layout(cache_path, out)

    # Verify against the protocols (counts + missing files + ASV scores).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import verify
    sys.exit(0 if verify(str(out)) else 1)


if __name__ == "__main__":
    main()
