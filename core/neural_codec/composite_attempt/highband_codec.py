"""
High-band specialist codec.

Encoder produces audio whose spectral support is hard-restricted to [3-6.5 kHz]
during the forward pass. Decoder operates only on the same band. Trained through
real Opus straight-through, so the model learns what specifically survives the
narrow-band high-frequency channel that Opus VoIP allocates few bits to.

If this works at low BER, we can stack it under the SEQ adversarial speech-band
codec to get composite "speech-textured 0-3 kHz + tone-modulated 3-6.5 kHz" with
combined reliable rate well above either band alone.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import math
import sys
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_codec import Encoder, Decoder, opus_surrogate, real_opus_batch, stft_loss, DEVICE, SR, SYMBOL_MS
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SYMBOL_N = SR * SYMBOL_MS // 1000  # 480
HIGH_BAND_LO = 3000.0
HIGH_BAND_HI = 6500.0


def hard_bandpass_torch(x: torch.Tensor, lo_hz: float, hi_hz: float) -> torch.Tensor:
    """Differentiable bandpass via FFT mask."""
    n = x.size(-1)
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, 1.0/SR).to(x.device)
    mask = ((freqs >= lo_hz) & (freqs <= hi_hz)).float()
    return torch.fft.irfft(X * mask, n=n, dim=-1)


class HighBandEncoder(Encoder):
    """Same architecture as IID Encoder; output is forced to live in [HIGH_BAND_LO, HIGH_BAND_HI]."""
    def forward(self, bits):
        x = super().forward(bits)
        return hard_bandpass_torch(x, HIGH_BAND_LO, HIGH_BAND_HI)


class HighBandDecoder(Decoder):
    """Same as IID Decoder; receives only high-band audio (caller is responsible)."""
    pass


def make_step(enc, dec, opt, n_bits, perc_w=0.005, snr_db=18.0):
    def step(batch_size, surrogate=True):
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        audio = enc(bits)
        if surrogate:
            audio_codec = opus_surrogate(audio, snr_db=snr_db)
        else:
            with torch.no_grad():
                audio_rt = real_opus_batch(audio)
            audio_codec = audio + (audio_rt - audio).detach()
        # Decoder sees only the high band of the (possibly distorted) audio
        audio_codec_band = hard_bandpass_torch(audio_codec, HIGH_BAND_LO, HIGH_BAND_HI)
        logits = dec(audio_codec_band)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        perc = stft_loss(audio)
        loss = bit_loss + perc_w * perc
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            ber = (preds != bits).float().mean().item()
        return dict(loss=loss.item(), bit_loss=bit_loss.item(), ber=ber)
    return step


def eval_real(enc, dec, n_bits, n_batches=4, batch_size=64):
    enc.eval(); dec.eval()
    bers = []
    with torch.no_grad():
        for _ in range(n_batches):
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            audio = enc(bits)
            audio_rt = real_opus_batch(audio)
            audio_band = hard_bandpass_torch(audio_rt, HIGH_BAND_LO, HIGH_BAND_HI)
            logits = dec(audio_band)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    enc.train(); dec.train()
    return float(np.mean(bers))


def train(n_bits=64, n_steps=2000, batch_size=128, lr=1e-3, perc_w=0.005,
          real_opus_every=25, real_opus_warmup=100, real_opus_bs=48,
          eval_every=250, ckpt_path=None):
    raw_bps = n_bits * 1000 / SYMBOL_MS
    print(f"# device={DEVICE}  n_bits={n_bits}  raw={raw_bps:.0f} bps")
    print(f"# high band: [{HIGH_BAND_LO:.0f}, {HIGH_BAND_HI:.0f}] Hz")
    print(f"# steps={n_steps}  bs={batch_size}  ro_every={real_opus_every} (warmup {real_opus_warmup})")

    enc = HighBandEncoder(n_bits=n_bits).to(DEVICE)
    dec = HighBandDecoder(n_bits=n_bits).to(DEVICE)
    n_p = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"# params: {n_p/1e6:.2f}M")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, betas=(0.5, 0.99))
    step_fn = make_step(enc, dec, opt, n_bits, perc_w=perc_w)

    print(f"\n{'step':>5} {'bit_loss':>8} {'sur_BER':>8} {'roBER':>7} {'evalBER':>8}")
    last_ro = float("nan")
    for step in range(1, n_steps+1):
        m = step_fn(batch_size, surrogate=True)
        if real_opus_every and step > real_opus_warmup and step % real_opus_every == 0:
            mr = step_fn(real_opus_bs, surrogate=False)
            last_ro = mr["ber"]
        if step == 1 or step % 100 == 0 or step == n_steps:
            line = (f"{step:>5} {m['bit_loss']:>8.4f} {m['ber']*100:>6.2f}% "
                    f"{last_ro*100:>5.2f}%" if not math.isnan(last_ro)
                    else f"{step:>5} {m['bit_loss']:>8.4f} {m['ber']*100:>6.2f}% {'-':>6}")
            if step % eval_every == 0 or step == n_steps:
                ev = eval_real(enc, dec, n_bits)
                line += f"   {ev*100:>6.2f}%"
            print(line, flush=True)

    if ckpt_path:
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                    "n_bits": n_bits,
                    "high_band_lo": HIGH_BAND_LO, "high_band_hi": HIGH_BAND_HI},
                   ckpt_path)
        print(f"# saved {ckpt_path}")
    return enc, dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/composite_attempt/ckpt_highband.pt"))
    args = ap.parse_args()
    train(n_bits=args.n_bits, n_steps=args.steps, batch_size=args.batch,
          lr=args.lr, eval_every=args.eval_every, ckpt_path=args.ckpt)
