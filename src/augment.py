"""
augment.py — RawBoost data augmentation for anti-spoofing.

Faithful port of the official RawBoost implementation:
  H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans,
  "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic
   Speaker Verification Anti-Spoofing", ICASSP 2022.
  Original code: https://github.com/TakHemlata/SSL_Anti-spoofing (MIT license).

We use the "series (1)+(2)" configuration — linear & non-linear convolutive
noise followed by impulsive signal-dependent noise — which was the best setting
for the LA (codec/transmission) condition in the paper.

External material note (course honesty): RawBoost is NOT course material; it is
credited in the report as an external augmentation whose effect we ablate.
"""
import numpy as np
import torch
from scipy import signal as ss


def _rand(lo, hi, integer=False):
    y = np.random.uniform(lo, hi)
    return int(y) if integer else y


def _norm_wav(x, always=False):
    peak = np.amax(np.abs(x))
    if peak > 0 and (always or peak > 1):
        x = x / peak
    return x


def _gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw,
                      min_coeff, max_coeff, min_g, max_g, fs):
    """Random multi-band FIR filter built from cascaded band-stop windows."""
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rand(min_f, max_f)
        bw = _rand(min_bw, max_bw)
        c = _rand(min_coeff, max_coeff, integer=True)
        if c % 2 == 0:                       # firwin needs an odd tap count
            c += 1
        f1 = max(fc - bw / 2, 1 / 1000)
        f2 = min(fc + bw / 2, fs / 2 - 1 / 1000)
        b = np.convolve(ss.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)
    g = _rand(min_g, max_g)
    _, h = ss.freqz(b, 1, fs=fs)
    return pow(10, g / 20) * b / np.amax(np.abs(h))


def _filter_fir(x, b):
    n = b.shape[0] + 1
    y = ss.lfilter(b, 1, np.pad(x, (0, n)))
    return y[n // 2: y.shape[0] - n // 2]


def lnl_convolutive_noise(x, fs, n_f=5, n_bands=5, min_f=20, max_f=8000,
                          min_bw=100, max_bw=1000, min_coeff=10, max_coeff=100,
                          min_g=0, max_g=0, min_bias=5, max_bias=20):
    """RawBoost algo (1): linear + non-linear convolutive noise. Powers of the
    signal are passed through independent random filters and summed, simulating
    channel/codec nonlinearity."""
    y = np.zeros_like(x)
    for i in range(n_f):
        if i == 1:                            # attenuate the non-linear terms
            min_g, max_g = min_g - min_bias, max_g - max_bias
        b = _gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw,
                              min_coeff, max_coeff, min_g, max_g, fs)
        y = y + _filter_fir(np.power(x, i + 1), b)
    return _norm_wav(y - np.mean(y))


def isd_additive_noise(x, p_max=10, g_sd=2):
    """RawBoost algo (2): impulsive signal-dependent noise on a random subset of
    samples (device/quantization artifacts)."""
    beta = _rand(0, p_max)
    y = x.copy()
    n = int(x.shape[0] * beta / 100)
    idx = np.random.permutation(x.shape[0])[:n]
    f_r = (2 * np.random.rand(n) - 1) * (2 * np.random.rand(n) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm_wav(y)


def ssi_additive_noise(x, fs, snr_min=10, snr_max=40, **notch_kw):
    """RawBoost algo (3): stationary signal-independent colored additive noise."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(notch_kw.get("n_bands", 5), 20, 8000, 100, 1000,
                          10, 100, 0, 0, fs)
    noise = _norm_wav(_filter_fir(noise, b), always=True)
    snr = _rand(snr_min, snr_max)
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(x) / 10 ** (0.05 * snr)
    return x + noise


class Augmenter:
    """Callable used by the Dataset (train split only).

    mode:
      "rawboost" — series algo (1)+(2), the paper's best LA setting
      "none"     — identity (clean-training ablation)
    p: probability of applying the augmentation to a given clip.
    """

    def __init__(self, mode: str = "rawboost", p: float = 0.5):
        self.mode, self.p = mode, p

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if self.mode == "none" or self.p <= 0 or torch.rand(1).item() >= self.p:
            return wav
        x = wav.numpy().astype(np.float64)
        x = lnl_convolutive_noise(x, sr)
        x = isd_additive_noise(x)
        return torch.from_numpy(x.astype(np.float32))
