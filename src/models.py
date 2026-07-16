"""
models.py — Model zoo.

  * LCNN        — Light CNN with Max-Feature-Map activation (classic anti-spoofing
                  baseline; Tutorial 6 material: CNNs + BatchNorm)
  * SSLDetector — wav2vec 2.0 (frozen, external pretrained weights) adapted with
                  LoRA / DoRA / linear-probe + attentive pooling + linear head
                  (Tutorial 9 material: transfer learning, LoRA & DoRA, SSL)

Every model returns (logits, embedding); the embedding feeds OC-Softmax and the
t-SNE visualization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LogMel, LFCC, Wav2Vec2Frontend


# ----------------------------- Baseline: LCNN ------------------------------------
class MFM(nn.Module):
    """Max-Feature-Map: split channels in half and take the elementwise max —
    a learned, competitive activation that makes LCNNs compact and selective."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    """Light CNN over an LFCC / log-mel "image". ~0.41 M trainable params —
    comparable to the LCNNs used as official ASVspoof baselines.

    Channels: input is the single-channel spectrogram; the four blocks have
    post-MFM widths 48 -> 64 -> 96 -> 128 (each conv emits 2x, MFM halves it).
    """

    def __init__(self, frontend="lfcc", emb_dim=128):
        super().__init__()
        self.front = LFCC() if frontend == "lfcc" else LogMel()
        c = [1, 48, 64, 96, 128]
        layers = []
        for i in range(len(c) - 1):
            layers += [nn.Conv2d(c[i], c[i + 1] * 2, 3, padding=1), MFM(),
                       nn.MaxPool2d(2), nn.BatchNorm2d(c[i + 1])]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(c[-1], emb_dim)     # embedding used for t-SNE / OC-Softmax
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, wav):
        x = self.front(wav)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        emb = self.proj(x)
        return self.head(F.relu(emb)), emb        # logits (B,1), embedding (B,D)


# ----------------------------- Proposed: wav2vec2 + PEFT -------------------------
class AttentivePool(nn.Module):
    """Attention-weighted mean over time (Tutorial 7: attention mechanism)."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x):                                  # x: (B, T, D)
        a = torch.softmax(self.w(x).squeeze(-1), dim=1)    # (B, T)
        return (a.unsqueeze(-1) * x).sum(1)                # (B, D)


class SSLDetector(nn.Module):
    """Frozen wav2vec 2.0 + parameter-efficient adaptation.

    peft:
      "lora"  — LoRA adapters on attention projections (the main model)
      "dora"  — DoRA variant (magnitude/direction decomposition, Tutorial 9)
      "probe" — nothing trainable inside the backbone (linear-probe lower bound)
    """

    def __init__(self, backbone="facebook/wav2vec2-base",
                 peft="lora", r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.front = Wav2Vec2Frontend(backbone, freeze=True)
        dim = self.front.backbone.config.hidden_size
        self._add_peft(peft, r, alpha)
        self.pool = AttentivePool(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)

    def _add_peft(self, peft, r, alpha):
        if peft == "probe":
            return                                          # backbone stays fully frozen
        if peft in ("lora", "dora"):
            import types
            from peft import LoraConfig, get_peft_model
            # peft calls enable_input_require_grads() when wrapping, which needs
            # get_input_embeddings(). Wav2Vec2Model is an audio model with no token
            # embeddings and raises NotImplementedError, so we point it at the conv
            # feature encoder (the actual model input stage). LoRA gradients flow
            # through the adapter params regardless; this only stops the crash.
            bb = self.front.backbone
            bb.get_input_embeddings = types.MethodType(
                lambda self: self.feature_extractor, bb)
            cfg = LoraConfig(
                r=r, lora_alpha=alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                use_dora=(peft == "dora"),
                bias="none", task_type=None)
            self.front.backbone = get_peft_model(self.front.backbone, cfg)
            return
        raise ValueError(f"unknown peft mode {peft!r}")

    def forward(self, wav):
        feats = self.front(wav)                # (B, T, D)
        emb = self.pool(feats)                 # (B, D)
        logits = self.head(self.drop(emb))     # (B, 1)
        return logits, emb


def build_model(cfg: dict) -> nn.Module:
    """Factory used by train.py / eval.py from a YAML config."""
    m = cfg["model"]
    if m["name"] == "lcnn":
        return LCNN(frontend=m.get("frontend", "lfcc"))
    if m["name"] == "ssl":
        return SSLDetector(
            backbone=m.get("backbone", "facebook/wav2vec2-base"),
            peft=m.get("peft", "lora"),
            r=m.get("r", 8),
            alpha=m.get("alpha", 16))
    raise ValueError(f"unknown model {m['name']}")


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
