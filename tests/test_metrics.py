"""Sanity tests for the metrics that every claim in the report rests on."""
import numpy as np
import pytest

from src.metrics import compute_eer, compute_min_tdcf, det_curve


def test_eer_separable_is_zero():
    scores = np.r_[np.full(100, 2.0), np.full(100, -2.0)]
    labels = np.r_[np.ones(100), np.zeros(100)]
    assert compute_eer(scores, labels) == pytest.approx(0.0, abs=1e-9)


def test_eer_random_is_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 20000)
    labels = np.r_[np.ones(10000), np.zeros(10000)]
    assert compute_eer(scores, labels) == pytest.approx(0.5, abs=0.02)


def test_eer_inverted_scores_is_one_minus():
    rng = np.random.default_rng(1)
    scores = np.r_[rng.normal(1, 1, 500), rng.normal(-1, 1, 500)]
    labels = np.r_[np.ones(500), np.zeros(500)]
    e = compute_eer(scores, labels)
    e_inv = compute_eer(-scores, labels)
    assert e + e_inv == pytest.approx(1.0, abs=0.02)


def test_det_curve_monotone():
    rng = np.random.default_rng(2)
    frr, far, thr = det_curve(rng.normal(1, 1, 300), rng.normal(-1, 1, 300))
    assert (np.diff(frr) >= 0).all()      # FRR grows with threshold
    assert (np.diff(far) <= 0).all()      # FAR shrinks with threshold
    assert (np.diff(thr) >= 0).all()


@pytest.fixture
def asv_file(tmp_path):
    """Synthetic ASV score file in the official 3-column format."""
    rng = np.random.default_rng(3)
    lines = []
    for key, mu in (("target", 3.0), ("nontarget", -3.0), ("spoof", 0.0)):
        for s in rng.normal(mu, 1.0, 500):
            lines.append(f"src {key} {s:.4f}")
    f = tmp_path / "asv.txt"
    f.write_text("\n".join(lines))
    return str(f)


def test_min_tdcf_perfect_cm_near_zero(asv_file):
    scores = np.r_[np.full(200, 5.0), np.full(200, -5.0)]
    labels = np.r_[np.ones(200), np.zeros(200)]
    assert compute_min_tdcf(scores, labels, asv_file) == pytest.approx(0.0, abs=1e-6)


def test_min_tdcf_random_cm_worse_than_perfect(asv_file):
    rng = np.random.default_rng(4)
    labels = np.r_[np.ones(500), np.zeros(500)]
    random_tdcf = compute_min_tdcf(rng.normal(0, 1, 1000), labels, asv_file)
    assert 0.0 < random_tdcf <= 1.0 + 1e-9   # min t-DCF is normalized; 1.0 = useless CM
