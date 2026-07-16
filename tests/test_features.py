"""Front-end numerical stability under AMP.

Regression test for a real bug: under torch.autocast fp16, clamp_min(1e-9)
underflows to a literal 0.0 (fp16's smallest representable magnitude is
~6.1e-5), so log(0) = -inf and the loss goes NaN a few batches later. Silent
speech (near-zero amplitude, common in real recordings) triggers this. The
front-ends must force fp32 internally regardless of the caller's autocast state.
"""
import torch

from src.features import LFCC, LogMel

CUDA = torch.cuda.is_available()


def _check_finite(module, wav):
    device = "cuda" if CUDA else "cpu"
    module = module.to(device)
    wav = wav.to(device)
    with torch.autocast(device_type=device, enabled=CUDA):   # AMP only exists on CUDA
        out = module(wav)
    assert torch.isfinite(out).all(), "front-end produced non-finite values under AMP"


def test_logmel_finite_on_silence_under_amp():
    _check_finite(LogMel(), torch.zeros(2, 16000))


def test_lfcc_finite_on_silence_under_amp():
    _check_finite(LFCC(), torch.zeros(2, 16000))


def test_logmel_finite_on_normal_audio_under_amp():
    torch.manual_seed(0)
    _check_finite(LogMel(), 0.1 * torch.randn(2, 16000))
