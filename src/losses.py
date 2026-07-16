"""
losses.py — Training objectives.

- Weighted BCE handles the ~1:9 bona fide/spoof class imbalance.
- OC-Softmax (Zhang et al., "One-Class Learning Towards Synthetic Voice
  Spoofing Detection", 2021 — external material, credited) tightens the bona
  fide cluster and pushes spoofs beyond a margin; known to improve
  generalization to UNSEEN attacks, which is exactly our research question.

Scoring convention: higher score == more bona fide.
  BCE models score with sigmoid(logit); OC-Softmax models score with the cosine
  similarity of the embedding to the learned bona fide center.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits, embeddings, targets):
        # embeddings unused; shared signature with OCSoftmax
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets.float(), pos_weight=self.pos_weight)

    @staticmethod
    def score(logits, embeddings):
        """P(bona fide) in [0,1]."""
        return torch.sigmoid(logits.squeeze(-1))


class OCSoftmax(nn.Module):
    """One-class softmax over a learnable bona fide center (has trainable params —
    include criterion.parameters() in the optimizer and save with the checkpoint)."""

    def __init__(self, feat_dim: int, m_real: float = 0.9, m_fake: float = 0.2,
                 alpha: float = 20.0):
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, feat_dim))
        self.m_real, self.m_fake, self.alpha = m_real, m_fake, alpha

    def _cos(self, embeddings):
        w = F.normalize(self.center, dim=1)
        x = F.normalize(embeddings, dim=1)
        return x @ w.t()                          # (B, 1) cosine to bona fide center

    def forward(self, logits, embeddings, targets):
        # logits unused; shared signature with WeightedBCE.
        # softplus(z) penalizes z>0, so we build z to be positive on violations:
        #   bona fide (target=1): z = alpha*(m_real - cos)  -> pushes cos up past m_real
        #   spoof     (target=0): z = alpha*(cos - m_fake)  -> pushes cos below m_fake
        cos = self._cos(embeddings)
        margin = torch.where(targets.bool(), self.m_real, self.m_fake).unsqueeze(1)
        scores = self.alpha * (margin - cos)
        scores[~targets.bool()] *= -1.0           # negate for SPOOF, not bona fide
        return F.softplus(scores).mean()

    def score(self, logits, embeddings):
        """Cosine to the bona fide center; higher == more bona fide."""
        return self._cos(embeddings).squeeze(-1)


def build_criterion(cfg: dict, model, n_pos: int, n_neg: int) -> nn.Module:
    """Factory: 'bce' (default) or 'ocsoftmax' from cfg['train']['loss']."""
    kind = cfg["train"].get("loss", "bce")
    if kind == "bce":
        return WeightedBCE(pos_weight=n_neg / max(n_pos, 1))
    if kind == "ocsoftmax":
        emb_dim = model.head.in_features          # both models expose .head
        return OCSoftmax(feat_dim=emb_dim)
    raise ValueError(f"unknown loss {kind!r}")
