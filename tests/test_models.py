"""Model shape/param-count tests. The SSL test needs `transformers` + a network
download, so it is skipped unless RUN_SLOW=1 (run it once in the dlproj env / Colab)."""
import os

import pytest
import torch

from src.models import LCNN, build_model, count_trainable


def test_lcnn_forward_shapes():
    model = LCNN(frontend="lfcc")
    wav = torch.randn(2, 16000)                    # 2 clips x 1 s
    logits, emb = model(wav)
    assert logits.shape == (2, 1)
    assert emb.shape[0] == 2 and emb.ndim == 2


def test_lcnn_param_count_reasonable():
    n = count_trainable(LCNN())
    assert 100_000 < n < 5_000_000                 # "light" CNN, honestly reported


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model({"model": {"name": "nope"}})


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                    reason="downloads wav2vec2 weights; set RUN_SLOW=1 to run")
def test_ssl_lora_trains_under_2_percent():
    cfg = {"model": {"name": "ssl", "backbone": "facebook/wav2vec2-base",
                     "peft": "lora", "r": 8, "alpha": 16}}
    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable(model)
    assert trainable / total < 0.02                # the PEFT claim we make in the report
    logits, emb = model(torch.randn(1, 16000))
    assert logits.shape == (1, 1)
