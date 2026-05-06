"""
Steganographic embedding into real speech.

  cover_speech_30ms (real human speech)  +  bits  -->  modified_speech_30ms
                                                              |
                                                       real libopus 24k
                                                              |
                                                  modified_speech_after_codec
                                                              |
                                                          decoder
                                                              |
                                                            bits

The cover IS real human speech sampled from the corpus. The encoder produces a
small additive perturbation that encodes the bits. The output sounds like the
input cover (because perturbation is small) AND survives Opus (because Opus is
designed to preserve speech) AND the decoder can extract the bits.

Loss: bit BCE + alpha * |modified - cover|_1 (perceptual closeness)
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import math
import sys
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_codec import opus_surrogate, real_opus_batch, DEVICE, SR, SYMBOL_MS
from adversarial_realism import RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000  # 480 samples


class StegEncoder(nn.Module):
    """Takes (cover_audio, bits) -> modified_audio.

    Architecture: bits get embedded to a feature vector. Cover audio passes
    through a 1D conv encoder. The two are concatenated channel-wise. Then a
    1D conv decoder produces the perturbation, added to cover, clamped.
    """
    def __init__(self, n_bits: int, hidden: int = 96):
        super().__init__()
        self.n_bits = n_bits
        # Bit embedding broadcast to time
        self.bit_fc = nn.Linear(n_bits, hidden * 60)  # to (B, hidden, 60)
        # Cover audio encoder: downsamples by 8x to (B, hidden, 60)
        self.cover_enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),  # 480 -> 240
            nn.Conv1d(64, hidden, 4, 2, 1), nn.GELU(),  # 240 -> 120
            nn.Conv1d(hidden, hidden, 4, 2, 1), nn.GELU(),  # 120 -> 60
        )
        # Decoder: upsample joint features back to 480 samples (the perturbation)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden*2, hidden, 4, 2, 1), nn.GELU(),  # 60 -> 120
            nn.ConvTranspose1d(hidden, 64, 4, 2, 1),       nn.GELU(),  # 120 -> 240
            nn.ConvTranspose1d(64, 32, 4, 2, 1),           nn.GELU(),  # 240 -> 480
            nn.Conv1d(32, 1, 7, 1, 3),                     nn.Tanh(),
        )

    def forward(self, cover: torch.Tensor, bits: torch.Tensor, perturbation_scale: float = 0.3):
        # cover: (B, 480), bits: (B, n_bits)
        c_feat = self.cover_enc(cover.unsqueeze(1))  # (B, hidden, 60)
        b_feat = self.bit_fc(bits.float() * 2 - 1).view(c_feat.size(0), -1, 60)  # (B, hidden, 60)
        joint = torch.cat([c_feat, b_feat], dim=1)
        delta = self.dec(joint).squeeze(1)  # (B, 480), in [-1, 1]
        # Psychoacoustic masking: perturbation magnitude scales with local cover amplitude.
        # Loud cover regions tolerate more perturbation; quiet regions get little.
        # mask = smoothed |cover|, so we don't poke holes in silence.
        with torch.no_grad():
            mask_raw = cover.abs()
            # 5-sample box filter for smoothing
            mask = F.avg_pool1d(mask_raw.unsqueeze(1), kernel_size=11, stride=1, padding=5).squeeze(1)
            mask = mask.clamp(min=0.005)  # tiny floor so silence still carries some bits
        modified = cover + perturbation_scale * delta * mask
        return modified.clamp(-1, 1), delta


class StegDecoder(nn.Module):
    """Takes (post-codec audio) -> bits.

    Doesn't take the cover as input — has to recover bits from just the
    received audio. (Symmetric to a real comm channel.)
    """
    def __init__(self, n_bits: int, hidden: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),     # 480 -> 240
            nn.Conv1d(64, 96, 4, 2, 1), nn.GELU(),     # 240 -> 120
            nn.Conv1d(96, hidden, 4, 2, 1), nn.GELU(), # 120 -> 60
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 60, 256),
            nn.GELU(),
            nn.Linear(256, n_bits),
        )

    def forward(self, audio):
        x = audio.unsqueeze(1)
        x = self.conv(x).flatten(1)
        return self.fc(x)


def train(
    n_bits: int = 32,
    n_steps: int = 2500,
    batch_size: int = 96,
    lr: float = 1e-3,
    perturbation_scale: float = 0.05,
    alpha_perceptual: float = 5.0,  # weight on |modified - cover|_1
    snr_db: float = 18.0,
    real_opus_every: int = 25,
    real_opus_warmup: int = 100,
    real_opus_bs: int = 48,
    eval_every: int = 250,
    ckpt_path: str | None = None,
    init_ckpt: str | None = None,
    alpha_ramp_to: float | None = None,
    real_speech_dir: str = str(REPO_ROOT / "tests/data/raw"),
):
    raw_bps = n_bits * 1000 / SYMBOL_MS
    print(f"# device={DEVICE}  n_bits={n_bits}  raw={raw_bps:.0f} bps")
    print(f"# perturbation_scale={perturbation_scale}  alpha_L1={alpha_perceptual}")
    print(f"# steps={n_steps}  bs={batch_size}  ro_every={real_opus_every}")

    enc = StegEncoder(n_bits).to(DEVICE)
    dec = StegDecoder(n_bits).to(DEVICE)
    if init_ckpt and Path(init_ckpt).exists():
        st = torch.load(init_ckpt, map_location=DEVICE)
        enc.load_state_dict(st["encoder"]); dec.load_state_dict(st["decoder"])
        print(f"# loaded init checkpoint: {init_ckpt}")
    n_p = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"# params: {n_p/1e6:.2f}M")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, betas=(0.5, 0.99))

    paths = [Path(real_speech_dir)/n for n in ("synth_readaloud_01.wav", "lex_jensen.wav")]
    sampler = RealSpeechSampler(paths)

    print(f"\n{'step':>5} {'bit_loss':>8} {'L1':>7} {'snr_dB':>7} {'sur_BER':>8} {'roBER':>7} {'evalBER':>8}")
    last_ro = float("nan")
    for step in range(1, n_steps + 1):
        # Optional alpha ramp from `alpha_perceptual` to `alpha_ramp_to` over first half of training
        if alpha_ramp_to is not None:
            ramp_progress = min(1.0, step / (n_steps * 0.5))
            alpha_now = alpha_perceptual + ramp_progress * (alpha_ramp_to - alpha_perceptual)
        else:
            alpha_now = alpha_perceptual
        cover = sampler.sample(batch_size)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        modified, delta = enc(cover, bits, perturbation_scale=perturbation_scale)
        # Surrogate channel
        rx = opus_surrogate(modified, snr_db=snr_db)
        logits = dec(rx)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        l1 = F.l1_loss(modified, cover)
        # SNR of perturbation relative to cover
        cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
        pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
        snr_db_meas = 10 * torch.log10(cover_pwr / pert_pwr).mean().item()
        loss = bit_loss + alpha_now * l1
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            sur_ber = (preds != bits).float().mean().item()

        # Real-Opus straight-through
        if real_opus_every and step > real_opus_warmup and step % real_opus_every == 0:
            cover = sampler.sample(real_opus_bs)
            bits = torch.randint(0, 2, (real_opus_bs, n_bits), device=DEVICE).float()
            modified, _ = enc(cover, bits, perturbation_scale=perturbation_scale)
            with torch.no_grad():
                rx_real = real_opus_batch(modified)
            rx_st = modified + (rx_real - modified).detach()
            logits = dec(rx_st)
            bit_loss_ro = F.binary_cross_entropy_with_logits(logits, bits)
            l1_ro = F.l1_loss(modified, cover)
            loss_ro = bit_loss_ro + alpha_now * l1_ro
            opt.zero_grad(); loss_ro.backward(); opt.step()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                last_ro = (preds != bits).float().mean().item()

        if step == 1 or step % 100 == 0 or step == n_steps:
            line = (f"{step:>5} {bit_loss.item():>8.4f} {l1.item():>7.4f} {snr_db_meas:>6.1f} "
                    f"{sur_ber*100:>6.2f}% "
                    f"{last_ro*100:>5.2f}%" if not math.isnan(last_ro)
                    else f"{step:>5} {bit_loss.item():>8.4f} {l1.item():>7.4f} {snr_db_meas:>6.1f} "
                         f"{sur_ber*100:>6.2f}% {'-':>6}")
            if step % eval_every == 0 or step == n_steps:
                ev = eval_real(enc, dec, sampler, n_bits, perturbation_scale)
                line += f"   {ev['ber']*100:>5.2f}% (snr={ev['snr_db']:.1f}dB)"
            print(line, flush=True)

    if ckpt_path:
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                    "n_bits": n_bits, "perturbation_scale": perturbation_scale}, ckpt_path)
        print(f"# saved {ckpt_path}")
    return enc, dec, sampler


def eval_real(enc, dec, sampler, n_bits, perturbation_scale, n_batches=4, batch_size=64):
    enc.eval(); dec.eval()
    bers = []; snrs = []
    with torch.no_grad():
        for _ in range(n_batches):
            cover = sampler.sample(batch_size)
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            modified, _ = enc(cover, bits, perturbation_scale=perturbation_scale)
            rx = real_opus_batch(modified)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
            cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
            pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
            snrs.append((10 * torch.log10(cover_pwr / pert_pwr)).mean().item())
    enc.train(); dec.train()
    return dict(ber=float(np.mean(bers)), snr_db=float(np.mean(snrs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perturbation_scale", type=float, default=0.05)
    ap.add_argument("--alpha_perceptual", type=float, default=5.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/stego/ckpt_stego.pt"))
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--alpha_ramp_to", type=float, default=None)
    args = ap.parse_args()
    train(n_bits=args.n_bits, n_steps=args.steps, batch_size=args.batch,
          lr=args.lr, perturbation_scale=args.perturbation_scale,
          alpha_perceptual=args.alpha_perceptual,
          alpha_ramp_to=args.alpha_ramp_to,
          init_ckpt=args.init_ckpt,
          eval_every=args.eval_every, ckpt_path=args.ckpt)
