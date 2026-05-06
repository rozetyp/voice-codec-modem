"""
Steganographic embedding into real speech, with AMR-NB 12.2 kbps in the channel.

Channel = ffmpeg libopencore_amrnb -b:a 12.2k -ar 8000 (then re-up to 16 kHz).
This is the *actual* cellular voice channel for the original threat model.

Hypothesis: real-speech cover survives AMR-NB by design (it's literally what
the codec was built to preserve). Sub-perceptual perturbations might ride
along in the bit allocation AMR-NB devotes to that speech.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import math
import subprocess
import sys
import tempfile
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

from neural_codec import opus_surrogate, stft_loss, DEVICE, SR, SYMBOL_MS
from adversarial_realism import RealSpeechSampler
from stego_codec import StegEncoder, StegDecoder

SYMBOL_N = SR * SYMBOL_MS // 1000  # 480 samples
WORK = Path(tempfile.gettempdir()) / "stego_amrnb"
WORK.mkdir(exist_ok=True)


def amrnb_round_trip(audio_np: np.ndarray, bitrate_kbps: float = 12.2) -> np.ndarray:
    """Round-trip 1D audio through AMR-NB.
    AMR-NB requires 8 kHz mono. ffmpeg handles resample both directions."""
    tag = str(np.random.randint(1, 1<<30))
    inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.amr"; out = WORK/f"out_{tag}.wav"
    sf.write(inp, audio_np.astype(np.float32), SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                    "-c:a","libopencore_amrnb","-ar","8000","-ac","1",
                    "-b:a", f"{bitrate_kbps}k", str(opx)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                    "-ar",str(SR),"-ac","1",str(out)], check=True)
    a, _ = sf.read(out)
    inp.unlink(missing_ok=True); opx.unlink(missing_ok=True); out.unlink(missing_ok=True)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio_np):
        a = np.pad(a, (0, len(audio_np) - len(a)))
    elif len(a) > len(audio_np):
        a = a[:len(audio_np)]
    return a.astype(np.float32)


def amrnb_round_trip_batch(audio_t: torch.Tensor, bitrate_kbps: float = 12.2) -> torch.Tensor:
    audio_np = audio_t.detach().cpu().numpy()
    B, N = audio_np.shape
    flat = audio_np.reshape(-1)
    rt = amrnb_round_trip(flat, bitrate_kbps=bitrate_kbps)
    rt = rt[: B*N].reshape(B, N)
    return torch.from_numpy(rt).to(audio_t.device)


def amrnb_surrogate(audio: torch.Tensor, snr_db: float = 12.0):
    """Differentiable approximation of AMR-NB.
    Pipeline: lowpass at 4 kHz (AMR-NB nyquist), perceptual emphasis on 200-3400 Hz
    (telephony band), heavy quantization noise. Calibrated to be HARSHER than the
    Opus surrogate so the model has to learn robust solutions."""
    sr = SR; n = audio.size(-1)
    fft = torch.fft.rfft(audio, dim=-1)
    freqs = torch.fft.rfftfreq(n, 1.0/sr).to(audio.device)
    # Hard lowpass at 4 kHz with smooth taper from 3.4-4 kHz
    mask = torch.clamp(1.0 - (freqs - 3400.0) / 600.0, 0.0, 1.0)
    # Telephony band emphasis: signal between 300-3400 is preserved more
    speech = ((freqs > 300) & (freqs < 3400)).float() * 0.3
    weight = mask * (1.0 + speech)
    fft = fft * weight
    out = torch.fft.irfft(fft, n=n, dim=-1)
    # ACELP-class quantization noise (stronger than Opus)
    rms = audio.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-9
    noise = torch.randn_like(out) * rms * (10 ** (-snr_db / 20))
    noise = noise + torch.randn_like(out) * 0.005  # noise floor
    out = out + noise
    return out.clamp(-1, 1)


def train(
    n_bits: int = 16,
    n_steps: int = 2000,
    batch_size: int = 96,
    lr: float = 1e-3,
    perturbation_scale: float = 0.5,
    alpha_perceptual: float = 0.0,
    alpha_ramp_to: float | None = None,
    snr_db: float = 12.0,
    real_amrnb_every: int = 25,
    real_amrnb_warmup: int = 200,
    real_amrnb_bs: int = 32,
    eval_every: int = 250,
    ckpt_path: str | None = None,
    init_ckpt: str | None = None,
    real_speech_dir: str = str(REPO_ROOT / "tests/data/raw"),
):
    raw_bps = n_bits * 1000 / SYMBOL_MS
    print(f"# device={DEVICE}  n_bits={n_bits}  raw={raw_bps:.0f} bps")
    print(f"# channel: AMR-NB 12.2 kbps (8 kHz)  pert_scale={perturbation_scale}  alpha={alpha_perceptual}->{alpha_ramp_to}")
    print(f"# steps={n_steps}  bs={batch_size}  ro_every={real_amrnb_every}")

    enc = StegEncoder(n_bits).to(DEVICE)
    dec = StegDecoder(n_bits).to(DEVICE)
    if init_ckpt and Path(init_ckpt).exists():
        st = torch.load(init_ckpt, map_location=DEVICE)
        # Only load if n_bits matches
        if st.get("n_bits") == n_bits:
            enc.load_state_dict(st["encoder"]); dec.load_state_dict(st["decoder"])
            print(f"# loaded init checkpoint: {init_ckpt}")
        else:
            print(f"# init_ckpt has n_bits={st.get('n_bits')}, current is {n_bits} — starting fresh")

    n_p = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"# params: {n_p/1e6:.2f}M")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=lr, betas=(0.5, 0.99))

    paths = [Path(real_speech_dir)/n for n in ("synth_readaloud_01.wav", "lex_jensen.wav")]
    sampler = RealSpeechSampler(paths)

    print(f"\n{'step':>5} {'bit_loss':>8} {'L1':>7} {'snr_dB':>7} {'sur_BER':>8} {'roBER':>7} {'evalBER':>8}")
    last_ro = float("nan")
    for step in range(1, n_steps + 1):
        if alpha_ramp_to is not None:
            ramp = min(1.0, step / (n_steps * 0.5))
            alpha_now = alpha_perceptual + ramp * (alpha_ramp_to - alpha_perceptual)
        else:
            alpha_now = alpha_perceptual

        cover = sampler.sample(batch_size)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        modified, _ = enc(cover, bits, perturbation_scale=perturbation_scale)
        rx = amrnb_surrogate(modified, snr_db=snr_db)
        logits = dec(rx)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        l1 = F.l1_loss(modified, cover)
        cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
        pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
        snr_db_meas = (10 * torch.log10(cover_pwr / pert_pwr)).mean().item()
        loss = bit_loss + alpha_now * l1
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            sur_ber = (preds != bits).float().mean().item()

        # Real-AMR-NB straight-through
        if real_amrnb_every and step > real_amrnb_warmup and step % real_amrnb_every == 0:
            cover = sampler.sample(real_amrnb_bs)
            bits = torch.randint(0, 2, (real_amrnb_bs, n_bits), device=DEVICE).float()
            modified, _ = enc(cover, bits, perturbation_scale=perturbation_scale)
            with torch.no_grad():
                rx_real = amrnb_round_trip_batch(modified)
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
                    "n_bits": n_bits, "perturbation_scale": perturbation_scale,
                    "channel": "amrnb_12.2k"}, ckpt_path)
        print(f"# saved {ckpt_path}")
    return enc, dec, sampler


def eval_real(enc, dec, sampler, n_bits, perturbation_scale, n_batches=2, batch_size=64):
    enc.eval(); dec.eval()
    bers = []; snrs = []
    with torch.no_grad():
        for _ in range(n_batches):
            cover = sampler.sample(batch_size)
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            modified, _ = enc(cover, bits, perturbation_scale=perturbation_scale)
            rx = amrnb_round_trip_batch(modified)
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
    ap.add_argument("--n_bits", type=int, default=16)  # half of Opus stego — channel is harsher
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perturbation_scale", type=float, default=0.5)
    ap.add_argument("--alpha_perceptual", type=float, default=0.0)
    ap.add_argument("--alpha_ramp_to", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core" / "neural_codec" / "cellular" / "ckpt_stego_amrnb.pt"))
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--real_amrnb_every", type=int, default=25)
    ap.add_argument("--real_amrnb_warmup", type=int, default=200)
    ap.add_argument("--real_amrnb_bs", type=int, default=32)
    args = ap.parse_args()
    train(n_bits=args.n_bits, n_steps=args.steps, batch_size=args.batch,
          lr=args.lr, perturbation_scale=args.perturbation_scale,
          alpha_perceptual=args.alpha_perceptual, alpha_ramp_to=args.alpha_ramp_to,
          real_amrnb_every=args.real_amrnb_every,
          real_amrnb_warmup=args.real_amrnb_warmup,
          real_amrnb_bs=args.real_amrnb_bs,
          eval_every=args.eval_every, ckpt_path=args.ckpt, init_ckpt=args.init_ckpt)
