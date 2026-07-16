"""OC-Softmax regression tests.

A sign error here silently inverts the detector (bona fide scored as spoof),
producing a plausible-looking but wrong EER. These tests pin the direction.
"""
import torch

from src.losses import OCSoftmax, WeightedBCE


def _fixed_oc(dim=8):
    oc = OCSoftmax(feat_dim=dim)
    with torch.no_grad():                      # pin the center along axis 0
        oc.center.zero_()
        oc.center[0, 0] = 1.0
    return oc


def test_ocsoftmax_score_ranks_bonafide_above_spoof():
    oc = _fixed_oc()
    bona = torch.zeros(4, 8); bona[:, 0] = 1.0      # aligned with center
    spoof = torch.zeros(4, 8); spoof[:, 0] = -1.0   # anti-aligned
    emb = torch.cat([bona, spoof])
    scores = oc.score(None, emb)
    assert scores[:4].mean() > scores[4:].mean()    # bona fide more "bona fide"


def test_ocsoftmax_loss_lower_when_configuration_correct():
    oc = _fixed_oc()
    bona = torch.zeros(4, 8); bona[:, 0] = 1.0
    spoof = torch.zeros(4, 8); spoof[:, 0] = -1.0
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

    correct = oc(None, torch.cat([bona, spoof]), labels)     # bona aligned, spoof anti
    inverted = oc(None, torch.cat([spoof, bona]), labels)    # bona anti, spoof aligned
    assert correct.item() < inverted.item()


def test_ocsoftmax_training_step_reduces_loss():
    torch.manual_seed(0)
    oc = OCSoftmax(feat_dim=8)
    emb = torch.randn(16, 8, requires_grad=True)
    labels = torch.tensor([1, 0] * 8)
    opt = torch.optim.SGD(list(oc.parameters()) + [emb], lr=0.5)
    first = oc(None, emb, labels).item()
    for _ in range(20):
        opt.zero_grad()
        loss = oc(None, emb, labels)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_weighted_bce_score_is_probability():
    bce = WeightedBCE(pos_weight=3.0)
    logits = torch.tensor([[10.0], [-10.0]])
    s = WeightedBCE.score(logits, None)
    assert s[0] > 0.99 and s[1] < 0.01             # sigmoid of large +/- logits
