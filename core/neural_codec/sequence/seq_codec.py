"""
Sequence-level neural codec.

Instead of generating each 30 ms symbol independently (the IID model in neural_codec.py),
generate a 120 ms block of 4 symbols at once. The encoder is free to make formants/voicing
flow across the 4 sub-windows, producing audio with syllable-level structure.

Same channel model: real libopus 24k VoIP. Same loss ingredients: BCE on bits, STFT
perceptual, surrogate + real-Opus straight-through. Optionally an adversarial term.

Bit rate is the same as IID at the same `n_bits_per_symbol`: 4 * n_bits / 120ms.
What changes is *quality of the audio*: cross-symbol coherence is now possible.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import math
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_codec import opus_surrogate, real_opus_batch, stft_loss, DEVICE, SR, SYMBOL_MS
from adversarial_realism import Discriminator, RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000  # 480
N_CHUNKS = 4
BLOCK_N = SYMBOL_N * N_CHUNKS      # 1920 samples = 120 ms


class SeqEncoder(nn.Module):
    """bits (B, 4*n_bits) -> audio (B, 1920) in [-1, 1].
    Same depth as the IID encoder (4 upsample stages) but starting from a longer
    base spatial dimension (240 vs 60) so the receptive field covers the whole
    120 ms block without needing extra layers.
    """
    def __init__(self, n_bits_per_chunk: int, hidden: int = 128, base_len: int = 240):
        super().__init__()
        self.n_bits_total = n_bits_per_chunk * N_CHUNKS
        self.base_len = base_len
        self.hidden = hidden
        self.fc = nn.Linear(self.n_bits_total, hidden * base_len)
        # 240 -> 480 -> 960 -> 1920 — three 2x upsamples
        self.up = nn.Sequential(
            nn.ConvTranspose1d(hidden, 96, 4, 2, 1), nn.GELU(),  # 240 -> 480
            nn.ConvTranspose1d(96, 48, 4, 2, 1),     nn.GELU(),  # 480 -> 960
            nn.ConvTranspose1d(48, 24, 4, 2, 1),     nn.GELU(),  # 960 -> 1920
            nn.Conv1d(24, 1, 7, 1, 3),               nn.Tanh(),
        )

    def forward(self, bits):  # bits: (B, n_bits_total)
        x = self.fc(bits.float() * 2 - 1)
        x = x.view(x.size(0), self.hidden, self.base_len)
        x = self.up(x).squeeze(1)  # B x 1920
        # Soft window over the whole 120 ms block — only the boundaries
        ramp = max(8, BLOCK_N // 200)  # ~10 samples
        win = torch.ones(BLOCK_N, device=x.device)
        win[:ramp] = torch.linspace(0, 1, ramp, device=x.device)
        win[-ramp:] = torch.linspace(1, 0, ramp, device=x.device)
        return x * win


class SeqDecoder(nn.Module):
    """audio (B, 1920) -> bits logits (B, 4*n_bits)."""
    def __init__(self, n_bits_per_chunk: int, hidden: int = 128):
        super().__init__()
        self.n_bits_total = n_bits_per_chunk * N_CHUNKS
        self.hidden = hidden
        # Mirror of encoder: 1920 -> 960 -> 480 -> 240 (then read off)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 24, 7, 1, 3), nn.GELU(),
            nn.Conv1d(24, 48, 4, 2, 1), nn.GELU(),    # 1920 -> 960
            nn.Conv1d(48, 96, 4, 2, 1), nn.GELU(),    # 960  -> 480
            nn.Conv1d(96, hidden, 4, 2, 1), nn.GELU(),# 480  -> 240
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 240, 512),
            nn.GELU(),
            nn.Linear(512, self.n_bits_total),
        )

    def forward(self, audio):
        x = audio.unsqueeze(1)
        x = self.conv(x).flatten(1)
        return self.fc(x)


# ---------- discriminator over 120 ms windows ----------
class SeqDiscriminator(nn.Module):
    """Bigger receptive field than the 30 ms disc — can judge syllable-rate structure."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(1, hidden, 15, 1, 7)),     nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden, hidden*2, 5, 2, 2)), nn.LeakyReLU(0.2),  # 1920 -> 960
            sn(nn.Conv1d(hidden*2, hidden*2, 5, 2, 2)), nn.LeakyReLU(0.2),  # 960 -> 480
            sn(nn.Conv1d(hidden*2, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),  # 480 -> 240
            sn(nn.Conv1d(hidden*4, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),  # 240 -> 120
            sn(nn.Conv1d(hidden*4, hidden*8, 5, 2, 2)), nn.LeakyReLU(0.2),  # 120 -> 60
            sn(nn.Conv1d(hidden*8, 1, 60, 1, 0)),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze()


class SeqRealSpeechSampler(RealSpeechSampler):
    """120 ms chunks of real speech for discriminator training."""
    def __init__(self, paths):
        # Reuse parent then re-tile into 120 ms blocks
        super().__init__(paths)
        # Take consecutive 30 ms chunks and stack into 120 ms blocks
        n_blocks = len(self.clips) // N_CHUNKS
        self.blocks = self.clips[:n_blocks*N_CHUNKS].reshape(n_blocks, N_CHUNKS, SYMBOL_N).reshape(n_blocks, BLOCK_N)
        print(f"# real-speech 120 ms blocks: {len(self.blocks)}")

    def sample_block(self, batch_size: int) -> torch.Tensor:
        idx = np.random.randint(0, len(self.blocks), size=batch_size)
        return torch.from_numpy(self.blocks[idx]).to(DEVICE)


# ---------- training ----------
def train(
    n_bits_per_chunk: int = 32,   # 4 * 32 = 128 bits per 120 ms = 1067 bps raw
    n_steps: int = 2500,
    batch_size: int = 96,
    lr: float = 5e-4,
    perc_weight: float = 0.005,
    snr_db: float = 18.0,
    real_opus_every: int = 25,
    real_opus_warmup: int = 100,
    real_opus_bs: int = 48,
    lambda_adv: float = 0.0,
    adv_warmup: int = 0,
    eval_every: int = 250,
    ckpt_path: str | None = None,
    real_speech_dir: str = str(REPO_ROOT / "tests/data/raw"),
):
    n_bits_total = n_bits_per_chunk * N_CHUNKS
    raw_bps = n_bits_total / (BLOCK_N / SR)
    print(f"# device={DEVICE}  n_bits_per_chunk={n_bits_per_chunk}  block={BLOCK_N//SYMBOL_N} symbols/{BLOCK_N/SR*1000:.0f}ms")
    print(f"# raw_bitrate={raw_bps:.0f} bps  steps={n_steps}  bs={batch_size}  perc={perc_weight} snr={snr_db}")
    if lambda_adv > 0:
        print(f"# adversarial: lambda_adv={lambda_adv} warmup={adv_warmup}")

    enc = SeqEncoder(n_bits_per_chunk).to(DEVICE)
    dec = SeqDecoder(n_bits_per_chunk).to(DEVICE)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"# params (E+D): {n_params/1e6:.2f}M")
    opt_g = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, betas=(0.5, 0.99))

    sampler = None; disc = None; opt_d = None
    if lambda_adv > 0:
        paths = [Path(real_speech_dir)/n for n in ("synth_readaloud_01.wav", "lex_jensen.wav")]
        sampler = SeqRealSpeechSampler(paths)
        disc = SeqDiscriminator().to(DEVICE)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.99))

    print(f"\n{'step':>5} {'bit_loss':>8} {'perc':>6} {'sur_BER':>8} {'roBER':>7} {'evalBER':>8}")
    last_ro_step_ber = float("nan")
    for step in range(1, n_steps + 1):
        # ---- D step (if adv) ----
        if disc is not None:
            with torch.no_grad():
                bits = torch.randint(0, 2, (batch_size, n_bits_total), device=DEVICE).float()
                fake_audio = enc(bits).detach()
            real_audio = sampler.sample_block(batch_size)
            d_real = disc(real_audio); d_fake = disc(fake_audio)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            real_audio_gp = real_audio.detach().requires_grad_(True)
            d_real_gp = disc(real_audio_gp)
            grads = torch.autograd.grad(d_real_gp.sum(), real_audio_gp, create_graph=True)[0]
            d_loss = d_loss + 1.0 * (grads ** 2).sum(dim=-1).mean()
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # ---- G step ----
        bits = torch.randint(0, 2, (batch_size, n_bits_total), device=DEVICE).float()
        audio = enc(bits)
        audio_codec = opus_surrogate(audio, snr_db=snr_db)
        logits = dec(audio_codec)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        perc = stft_loss(audio)
        loss = bit_loss + perc_weight * perc
        if disc is not None:
            adv_w = lambda_adv * min(1.0, max(0.0, (step - adv_warmup) / max(1, adv_warmup)))
            adv_g = -disc(audio).mean()
            loss = loss + adv_w * adv_g
        opt_g.zero_grad(); loss.backward(); opt_g.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            sur_ber = (preds != bits).float().mean().item()

        # ---- real-Opus straight-through G step ----
        if real_opus_every and step > real_opus_warmup and step % real_opus_every == 0:
            bits = torch.randint(0, 2, (real_opus_bs, n_bits_total), device=DEVICE).float()
            audio_g = enc(bits)
            with torch.no_grad():
                audio_rt = real_opus_batch(audio_g)
            audio_st = audio_g + (audio_rt - audio_g).detach()
            logits = dec(audio_st)
            bit_loss_ro = F.binary_cross_entropy_with_logits(logits, bits)
            opt_g.zero_grad(); bit_loss_ro.backward(); opt_g.step()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                last_ro_step_ber = (preds != bits).float().mean().item()

        if step == 1 or step % 100 == 0 or step == n_steps:
            line = (f"{step:>5} {bit_loss.item():>8.4f} {perc.item():>6.3f} "
                    f"{sur_ber*100:>6.2f}% "
                    f"{last_ro_step_ber*100:>5.2f}%" if not math.isnan(last_ro_step_ber)
                    else f"{step:>5} {bit_loss.item():>8.4f} {perc.item():>6.3f} "
                         f"{sur_ber*100:>6.2f}% {'-':>6}")
            if step % eval_every == 0 or step == n_steps:
                ev = eval_real_opus(enc, dec, n_bits_total)
                line += f"  {ev*100:>6.2f}%"
            print(line, flush=True)

    if ckpt_path:
        save = {"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "n_bits_total": n_bits_total, "n_chunks": N_CHUNKS,
                "n_bits_per_chunk": n_bits_per_chunk, "block_n": BLOCK_N}
        if disc is not None: save["discriminator"] = disc.state_dict()
        torch.save(save, ckpt_path)
        print(f"# saved to {ckpt_path}")
    return enc, dec, disc


def eval_real_opus(enc, dec, n_bits_total, n_batches=4, batch_size=64):
    enc.eval(); dec.eval()
    all_ber = []
    with torch.no_grad():
        for _ in range(n_batches):
            bits = torch.randint(0, 2, (batch_size, n_bits_total), device=DEVICE).float()
            audio = enc(bits)
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_ber.append((preds != bits).float().mean().item())
    enc.train(); dec.train()
    return float(np.mean(all_ber))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits_per_chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_adv", type=float, default=0.0)
    ap.add_argument("--adv_warmup", type=int, default=300)
    ap.add_argument("--real_opus_every", type=int, default=25)
    ap.add_argument("--real_opus_warmup", type=int, default=100)
    ap.add_argument("--real_opus_bs", type=int, default=48)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core" / "neural_codec" / "sequence" / "ckpt_seq.pt"))
    args = ap.parse_args()
    train(
        n_bits_per_chunk=args.n_bits_per_chunk,
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        lambda_adv=args.lambda_adv,
        adv_warmup=args.adv_warmup,
        real_opus_every=args.real_opus_every,
        real_opus_warmup=args.real_opus_warmup,
        real_opus_bs=args.real_opus_bs,
        eval_every=args.eval_every,
        ckpt_path=args.ckpt,
    )
