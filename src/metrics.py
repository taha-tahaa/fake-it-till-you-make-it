"""
metrics.py — Official ASVspoof-style evaluation metrics.

* EER (Equal Error Rate)  — primary metric; the operating point where the
  false-acceptance rate (spoof accepted as bona fide) equals the false-rejection
  rate (bona fide rejected). Lower is better.
* min t-DCF               — the official ASVspoof 2019 tandem detection cost,
  computed against the organizer-provided ASV scores. Port of the official
  t-DCF v1 implementation (Kinnunen et al., "t-DCF: a Detection Cost Function
  for the Tandem Assessment of Spoofing Countermeasures and ASV").
* ROC-AUC / accuracy      — secondary, for completeness in the report.

Convention everywhere: higher score == more "bona fide"; label 1 = bona fide,
label 0 = spoof.
"""
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- DET
def det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """Vectorized DET sweep (O(n log n), fine for the 71k-utterance eval set).

    Returns (frr, far, thresholds): threshold i accepts iff score >= thresholds[i].
    Mirrors the official ASVspoof implementation's compute_det_curve.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    idx = np.argsort(all_scores, kind="mergesort")
    labels = labels[idx]

    # Cumulative counts of targets/nontargets below each threshold.
    tar_below = np.cumsum(labels)
    non_above = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_below)

    frr = np.concatenate(([0.0], tar_below / target_scores.size))       # false reject
    far = np.concatenate(([1.0], non_above / nontarget_scores.size))    # false accept
    thresholds = np.concatenate(([all_scores[idx[0]] - 1e-3], all_scores[idx]))
    return frr, far, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER as a fraction in [0, 1]; multiply by 100 for a percentage."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    frr, far, _ = det_curve(scores[labels == 1], scores[labels == 0])
    i = np.nanargmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2.0)


def compute_auc_acc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and accuracy at the 0.5 (sigmoid) / 0.0 (cosine) natural threshold."""
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(labels, scores))
    thr = 0.5 if scores.min() >= 0.0 and scores.max() <= 1.0 else 0.0
    acc = float(((scores >= thr).astype(int) == labels).mean())
    return auc, acc


# ----------------------------------------------------------------------- t-DCF
# Official ASVspoof 2019 cost model (fixed by the challenge organizers).
ASV_COST_MODEL = dict(
    Pspoof=0.05,        # prior of a spoofing attack
    Ptar=0.9405,        # = (1 - Pspoof) * 0.99, prior of target speaker
    Pnon=0.0095,        # = (1 - Pspoof) * 0.01, prior of nontarget speaker
    Cmiss_asv=1, Cfa_asv=10,   # ASV miss / false-accept costs
    Cmiss_cm=1, Cfa_cm=10,     # countermeasure (our model) miss / false-accept costs
)


def load_asv_scores(path: str):
    """Parse the organizer ASV score file (ASVspoof2019.LA.asv.eval.gi.trl.scores.txt).
    Lines: '<source> <key> <score>' with key in {target, nontarget, spoof}."""
    keys, scores = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        keys.append(parts[-2])
        scores.append(float(parts[-1]))
    keys, scores = np.array(keys), np.array(scores)
    return (scores[keys == "target"], scores[keys == "nontarget"], scores[keys == "spoof"])


def compute_min_tdcf(cm_scores: np.ndarray, cm_labels: np.ndarray,
                     asv_score_file: str, cost_model: dict = ASV_COST_MODEL) -> float:
    """Minimum normalized tandem-DCF (official ASVspoof 2019 protocol, t-DCF v1).

    The ASV system is fixed at its EER threshold on target/nontarget trials; we
    sweep only the countermeasure threshold and report the minimum.
    """
    tar_asv, non_asv, spoof_asv = load_asv_scores(asv_score_file)

    # Fix the ASV operating point at its own EER threshold.
    frr, far, thr = det_curve(tar_asv, non_asv)
    i = np.nanargmin(np.abs(frr - far))
    asv_threshold = thr[i]
    Pmiss_asv = float((tar_asv < asv_threshold).mean())
    Pfa_asv = float((non_asv >= asv_threshold).mean())
    Pmiss_spoof_asv = float((spoof_asv < asv_threshold).mean())

    # Constants of the unit-cost tandem model (eq. 10-11 of the t-DCF paper).
    C1 = (cost_model["Ptar"] * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
          - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv)
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)
    if C1 < 0 or C2 < 0:
        raise ValueError("Negative t-DCF weight — check the ASV score file.")

    cm_scores = np.asarray(cm_scores, dtype=np.float64)
    cm_labels = np.asarray(cm_labels, dtype=np.int64)
    frr_cm, far_cm, _ = det_curve(cm_scores[cm_labels == 1], cm_scores[cm_labels == 0])

    tdcf = C1 * frr_cm + C2 * far_cm            # sweep the CM threshold
    tdcf_norm = tdcf / min(C1, C2)              # normalize to [~0, 1+]
    return float(np.min(tdcf_norm))


if __name__ == "__main__":
    # Sanity: perfectly separable -> EER 0; random -> ~0.5.
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(1, 0.1, 500), rng.normal(-1, 0.1, 500)]
    y = np.r_[np.ones(500), np.zeros(500)]
    print("separable EER:", round(compute_eer(s, y) * 100, 3), "%")
    s = rng.normal(0, 1, 1000)
    print("random EER:", round(compute_eer(s, y) * 100, 2), "%")
