"""
features.py — Interchangeable front-ends.

Three front-ends so we can run the front-end ablation in the report:
  * LFCC      — classic anti-spoofing cepstral feature
  * log-mel   — standard spectrogram
  * wav2vec2  — self-supervised representation (frozen backbone, external weights)
"""
import torch
import torch.nn as nn


class LogMel(nn.Module):
    def __init__(self, sr=16000, n_mels=80, n_fft=512, hop=160):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)

    def forward(self, wav):                      # wav: (B, T)
        # Force fp32: under AMP autocast this module would run in fp16, where
        # clamp_min(1e-9) underflows to a literal 0.0 (fp16's smallest
        # representable magnitude is ~6.1e-5) and log(0) = -inf -> NaN loss.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = self.mel(wav.float()).clamp_min(1e-5).log()  # (B, n_mels, frames)
        return x.unsqueeze(1)                    # (B, 1, n_mels, frames) for a CNN


class LFCC(nn.Module):
    def __init__(self, sr=16000, n_lfcc=60, n_fft=512, hop=160):
        super().__init__()
        import torchaudio
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sr, n_lfcc=n_lfcc,
            speckwargs={"n_fft": n_fft, "hop_length": hop})

    def forward(self, wav):
        # Same fp32-under-AMP guard as LogMel; LFCC's internal log-magnitude
        # step is exposed to the identical fp16-underflow risk.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = self.lfcc(wav.float())           # (B, n_lfcc, frames)
        return x.unsqueeze(1)


class Wav2Vec2Frontend(nn.Module):
    """Frozen wav2vec 2.0 backbone. PEFT adapters (added in models.py) are the only
    trainable parameters inside the transformer.

    Note: wav2vec2 checkpoints expect zero-mean / unit-variance waveforms (the
    HF processor's do_normalize) — we apply that here per utterance. Skipping it
    silently degrades the representation.
    """
    def __init__(self, name="facebook/wav2vec2-base", freeze=True):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained(name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, wav):                      # wav: (B, T) at 16 kHz
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.var(dim=1, keepdim=True) + 1e-7).sqrt()
        return self.backbone(wav).last_hidden_state   # (B, frames, hidden)
