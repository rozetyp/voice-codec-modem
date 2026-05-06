"""
Adversarial-realism training: make the carrier audio sound like real human speech.

Fine-tune from a working bit-channel encoder/decoder by adding:
  1. a discriminator that classifies 30ms chunks as "real human speech" vs "neural codec output"
  2. an adversarial loss term that pushes the encoder to fool the discriminator
  3. while preserving bit recovery through real Opus

Goal: audio that the discriminator can't reliably tell from real speech, that still
delivers bits through libopus 24k VoIP. If we hit this, the carrier sounds like a person.
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
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_codec import (
    Encoder, Decoder, opus_surrogate, real_opus_batch, stft_loss,
    DEVICE, SR, SYMBOL_MS,
)
SYMBOL_N = SR * SYMBOL_MS // 1000  # 480 samples


# ---------- real-speech corpus loader ----------
class RealSpeechSampler:
    """Slice files in tests/data/raw/ into 30ms chunks; sample uniformly per call."""
    def __init__(self, paths, min_rms=0.01, exclude_silence=True):
        clips = []
        for p in paths:
            a, sr = sf.read(str(p))
            if a.ndim > 1: a = a.mean(axis=1)
            if sr != SR:
                # We only have 16k files in this corpus; trust + assert
                assert sr == SR, f"expected {SR}, got {sr} from {p}"
            a = a.astype(np.float32)
            # Normalize each clip to ~unit-RMS scale used by encoder output (~0.07-0.25)
            rms = np.sqrt(np.mean(a**2)) + 1e-9
            target_rms = 0.12  # match encoder output scale
            a = a * (target_rms / rms)
            n_chunks = len(a) // SYMBOL_N
            for i in range(n_chunks):
                seg = a[i*SYMBOL_N:(i+1)*SYMBOL_N]
                if exclude_silence and np.sqrt(np.mean(seg**2)) < min_rms:
                    continue
                clips.append(seg)
        self.clips = np.stack(clips, axis=0).astype(np.float32)
        print(f"# real-speech corpus: {len(self.clips)} chunks of {SYMBOL_N} samples ({SYMBOL_MS}ms)")

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = np.random.randint(0, len(self.clips), size=batch_size)
        x = self.clips[idx]
        return torch.from_numpy(x).to(DEVICE)


# ---------- discriminator ----------
class Discriminator(nn.Module):
    """1D CNN over 30ms windows. Returns one logit per chunk (real vs. fake)."""
    def __init__(self, hidden: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(1, hidden, 9, 1, 4)),       nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden, hidden*2, 5, 2, 2)),nn.LeakyReLU(0.2),  # 480 -> 240
            sn(nn.Conv1d(hidden*2, hidden*2, 5, 2, 2)), nn.LeakyReLU(0.2),  # 240 -> 120
            sn(nn.Conv1d(hidden*2, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),  # 120 -> 60
            sn(nn.Conv1d(hidden*4, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),  # 60 -> 30
            sn(nn.Conv1d(hidden*4, 1, 30, 1, 0)),    # 30 -> 1
        )

    def forward(self, x):  # x: B x N
        return self.net(x.unsqueeze(1)).squeeze()  # B


# ---------- training ----------
def gradient_penalty(D, real, fake):
    eps = torch.rand(real.size(0), 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    logits = D(interp)
    grads = torch.autograd.grad(
        outputs=logits, inputs=interp,
        grad_outputs=torch.ones_like(logits),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return ((grads.norm(2, dim=-1) - 1) ** 2).mean()


def train(
    n_bits: int = 64,
    init_ckpt: str | None = None,
    n_steps: int = 2000,
    batch_size: int = 128,
    lr_enc: float = 1e-4,
    lr_dec: float = 1e-4,
    lr_disc: float = 2e-4,
    lambda_adv: float = 0.3,
    lambda_perc: float = 0.005,
    lambda_gp: float = 1.0,
    real_opus_every: int = 25,
    real_opus_warmup: int = 0,
    real_opus_bs: int = 64,
    adv_warmup: int = 200,
    snr_db: float = 18.0,
    eval_every: int = 250,
    ckpt_path: str | None = None,
    real_speech_dir: str = str(REPO_ROOT / "tests/data/raw"),
):
    print(f"# device={DEVICE}  n_bits={n_bits} ({n_bits*1000//SYMBOL_MS} bps raw)")
    print(f"# init_ckpt={init_ckpt}  steps={n_steps}  bs={batch_size}")
    print(f"# lambda_adv={lambda_adv} (warmup {adv_warmup})  lambda_perc={lambda_perc} lambda_gp={lambda_gp}")

    enc = Encoder(n_bits=n_bits).to(DEVICE)
    dec = Decoder(n_bits=n_bits).to(DEVICE)
    if init_ckpt and Path(init_ckpt).exists():
        st = torch.load(init_ckpt, map_location=DEVICE)
        enc.load_state_dict(st["encoder"]); dec.load_state_dict(st["decoder"])
        print(f"# loaded encoder+decoder from {init_ckpt}")
    disc = Discriminator().to(DEVICE)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters()) + sum(p.numel() for p in disc.parameters())
    print(f"# params (E+D+disc): {n_params/1e6:.2f}M")

    opt_g = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                              lr=lr_enc, betas=(0.5, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_disc, betas=(0.5, 0.99))

    paths = [Path(real_speech_dir)/n for n in ("synth_readaloud_01.wav", "lex_jensen.wav")]
    sampler = RealSpeechSampler(paths)

    print(f"\n{'step':>5} {'bit_loss':>8} {'adv_g':>7} {'perc':>6} {'d_loss':>7} {'d_acc':>7} {'sur_BER':>7} {'roBER':>7} {'evalBER':>7} {'discRealAcc':>10}")

    last_ro_step_ber = float("nan")
    for step in range(1, n_steps + 1):
        # ---- D step ----
        with torch.no_grad():
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            fake_audio = enc(bits).detach()
        real_audio = sampler.sample(batch_size)
        d_real = disc(real_audio)
        d_fake = disc(fake_audio)
        # Hinge loss
        d_loss_real = F.relu(1.0 - d_real).mean()
        d_loss_fake = F.relu(1.0 + d_fake).mean()
        d_loss = d_loss_real + d_loss_fake
        # Gradient penalty (R1 on real)
        if lambda_gp > 0:
            real_audio_gp = real_audio.detach().requires_grad_(True)
            d_real_gp = disc(real_audio_gp)
            grads = torch.autograd.grad(d_real_gp.sum(), real_audio_gp, create_graph=True)[0]
            r1 = (grads ** 2).sum(dim=-1).mean()
            d_loss = d_loss + lambda_gp * r1
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        with torch.no_grad():
            d_acc = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()) * 0.5

        # ---- G step ----
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        audio = enc(bits)
        # Channel: surrogate for backprop
        audio_codec = opus_surrogate(audio, snr_db=snr_db)
        logits = dec(audio_codec)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        perc = stft_loss(audio)
        # Adversarial term: encoder wants disc(fake) high
        d_fake_for_g = disc(audio)
        adv_g = -d_fake_for_g.mean()  # hinge for G
        adv_w = lambda_adv * min(1.0, max(0.0, (step - adv_warmup) / max(1, adv_warmup)))
        g_loss = bit_loss + adv_w * adv_g + lambda_perc * perc
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            sur_ber = (preds != bits).float().mean().item()

        # ---- occasional real-Opus straight-through G step ----
        if real_opus_every and step > real_opus_warmup and step % real_opus_every == 0:
            bits = torch.randint(0, 2, (real_opus_bs, n_bits), device=DEVICE).float()
            audio_g = enc(bits)
            with torch.no_grad():
                audio_rt = real_opus_batch(audio_g)
            audio_st = audio_g + (audio_rt - audio_g).detach()
            logits = dec(audio_st)
            bit_loss_ro = F.binary_cross_entropy_with_logits(logits, bits)
            d_fake_for_g_ro = disc(audio_g)
            adv_g_ro = -d_fake_for_g_ro.mean()
            g_loss_ro = bit_loss_ro + adv_w * adv_g_ro
            opt_g.zero_grad(); g_loss_ro.backward(); opt_g.step()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                last_ro_step_ber = (preds != bits).float().mean().item()

        # ---- log ----
        if step == 1 or step % 50 == 0 or step == n_steps:
            line = (f"{step:>5} {bit_loss.item():>8.4f} {adv_g.item():>+7.3f} "
                    f"{perc.item():>6.3f} {d_loss.item():>7.3f} {d_acc.item()*100:>6.1f}% "
                    f"{sur_ber*100:>6.2f}% "
                    f"{last_ro_step_ber*100:>5.2f}%" if not math.isnan(last_ro_step_ber)
                    else f"{step:>5} {bit_loss.item():>8.4f} {adv_g.item():>+7.3f} "
                         f"{perc.item():>6.3f} {d_loss.item():>7.3f} {d_acc.item()*100:>6.1f}% "
                         f"{sur_ber*100:>6.2f}% {'-':>6}")
            if step % eval_every == 0 or step == n_steps:
                ev = eval_full(enc, dec, disc, sampler, n_bits)
                line += f"   {ev['eval_ber']*100:>5.2f}%   {ev['disc_real_acc']*100:>6.2f}%"
            print(line, flush=True)

    if ckpt_path:
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                    "discriminator": disc.state_dict(), "n_bits": n_bits}, ckpt_path)
        print(f"# saved to {ckpt_path}")
    return enc, dec, disc, sampler


def eval_full(enc, dec, disc, sampler, n_bits, n_batches=4, batch_size=64):
    enc.eval(); dec.eval(); disc.eval()
    bits_all = []; preds_all = []
    real_correct = 0; real_total = 0; fake_correct = 0; fake_total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            audio = enc(bits)
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bits_all.append(bits); preds_all.append(preds)
            d_real_logits = disc(sampler.sample(batch_size))
            d_fake_logits = disc(audio)
            real_correct += (d_real_logits > 0).sum().item(); real_total += d_real_logits.numel()
            fake_correct += (d_fake_logits < 0).sum().item(); fake_total += d_fake_logits.numel()
    bits_all = torch.cat(bits_all); preds_all = torch.cat(preds_all)
    eval_ber = (preds_all != bits_all).float().mean().item()
    enc.train(); dec.train(); disc.train()
    return dict(eval_ber=eval_ber,
                disc_real_acc=real_correct / real_total,
                disc_fake_acc=fake_correct / fake_total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits", type=int, default=64)
    ap.add_argument("--init_ckpt", default=str(REPO_ROOT / "core/neural_codec/ckpt_n64_mixed.pt"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lambda_adv", type=float, default=0.3)
    ap.add_argument("--adv_warmup", type=int, default=200)
    ap.add_argument("--real_opus_every", type=int, default=25)
    ap.add_argument("--real_opus_warmup", type=int, default=0)
    ap.add_argument("--real_opus_bs", type=int, default=64)
    ap.add_argument("--lr_enc", type=float, default=1e-4)
    ap.add_argument("--lr_disc", type=float, default=2e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/adversarial/ckpt_n64_adv.pt"))
    args = ap.parse_args()
    train(
        n_bits=args.n_bits,
        init_ckpt=args.init_ckpt,
        n_steps=args.steps,
        batch_size=args.batch,
        lambda_adv=args.lambda_adv,
        adv_warmup=args.adv_warmup,
        real_opus_every=args.real_opus_every,
        real_opus_warmup=args.real_opus_warmup,
        real_opus_bs=args.real_opus_bs,
        lr_enc=args.lr_enc,
        lr_disc=args.lr_disc,
        eval_every=args.eval_every,
        ckpt_path=args.ckpt,
    )
