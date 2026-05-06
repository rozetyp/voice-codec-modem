"""
Neural codec end-to-end through Opus.

  bits  ->  Encoder (NN)  ->  audio  ->  Opus  ->  audio'  ->  Decoder (NN)  ->  bits'
                                          ^
                                  (real libopus at eval;
                              differentiable surrogate at train)

Loss: BCE on bit recovery + multi-scale STFT loss (perceptual realism, light weight).
Surrogate: lowpass + perceptual A-weighting + Gaussian noise calibrated to ~18 dB SNR.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import math
import subprocess
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SR = 16000
SYMBOL_MS = 30
SYMBOL_N = SR * SYMBOL_MS // 1000  # 480 samples


# ---------- model ----------
class Encoder(nn.Module):
    """bits (B, n_bits) -> audio (B, SYMBOL_N) in [-1, 1]."""
    def __init__(self, n_bits: int, hidden: int = 192, base_len: int = 60):
        super().__init__()
        self.n_bits = n_bits
        self.base_len = base_len  # 60 -> 120 -> 240 -> 480
        self.fc = nn.Linear(n_bits, hidden * base_len)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(hidden, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),     nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),      nn.GELU(),
            nn.Conv1d(32, 1, 7, 1, 3),                nn.Tanh(),
        )

    def forward(self, bits):
        x = self.fc(bits.float() * 2 - 1)
        x = x.view(x.size(0), -1, self.base_len)
        x = self.up(x).squeeze(1)  # B x SYMBOL_N
        # Soft attack/release window so symbol edges are codec-friendly
        ramp = max(8, SYMBOL_N // 60)
        win = torch.ones(SYMBOL_N, device=x.device)
        win[:ramp] = torch.linspace(0, 1, ramp, device=x.device)
        win[-ramp:] = torch.linspace(1, 0, ramp, device=x.device)
        return x * win


class Decoder(nn.Module):
    """audio (B, SYMBOL_N) -> bit logits (B, n_bits)."""
    def __init__(self, n_bits: int, hidden: int = 192):
        super().__init__()
        self.n_bits = n_bits
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, 1, 3),  nn.GELU(),
            nn.Conv1d(32, 64, 4, 2, 1),  nn.GELU(),  # 480 -> 240
            nn.Conv1d(64, 128, 4, 2, 1), nn.GELU(),  # 240 -> 120
            nn.Conv1d(128, hidden, 4, 2, 1), nn.GELU(),  # 120 -> 60
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


# ---------- differentiable Opus surrogate ----------
def opus_surrogate(audio: torch.Tensor, snr_db: float = 18.0, lp_hz: float = 7000.0,
                   randomize: bool = True):
    """Approximate Opus 24k VoIP. Domain-randomized (random LP cutoff, SNR, group delay)
    so the model can't overfit to specific surrogate quirks.
    """
    sr = SR
    n = audio.size(-1)
    if randomize and audio.requires_grad:
        # Random per-batch perturbations (all differentiable wrt audio)
        snr_db = snr_db + (torch.rand(1, device=audio.device).item() - 0.5) * 8.0   # ±4 dB
        lp_hz = lp_hz + (torch.rand(1, device=audio.device).item() - 0.5) * 1500.0  # ±750 Hz
    fft = torch.fft.rfft(audio, dim=-1)
    freqs = torch.fft.rfftfreq(n, 1.0/sr).to(audio.device)
    # Smooth lowpass mask
    mask = torch.clamp(1.0 - (freqs - 6000.0) / max(1.0, lp_hz - 6000.0), 0.0, 1.0)
    # Perceptual emphasis (Opus models speech band; signals outside 200-3800 Hz get less faithful)
    speech = ((freqs > 150) & (freqs < 3800)).float() * 0.2
    weight = mask * (1.0 + speech)
    # Small random phase jitter to simulate Opus's group-delay/CELT framing
    if randomize and audio.requires_grad:
        phase_jitter = (torch.randn_like(freqs) * 0.05).to(torch.complex64)
        weight_c = weight.to(torch.complex64) * torch.exp(1j * phase_jitter)
        fft = fft * weight_c
    else:
        fft = fft * weight
    out = torch.fft.irfft(fft, n=n, dim=-1)
    # Quantization-style noise scaled by signal RMS
    rms = audio.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-9
    noise = torch.randn_like(out) * rms * (10 ** (-snr_db / 20))
    # Add a small floor of unmasked-by-RMS noise (Opus has nonzero noise floor)
    noise = noise + torch.randn_like(out) * 0.003
    out = out + noise
    return out.clamp(-1, 1)


# ---------- multi-scale STFT loss ----------
def stft_loss(x: torch.Tensor, target_log_mag_floor: float = -6.0):
    """Encourage harmonic-structured (speech-like) spectra by minimizing log-mag floor.
    Penalizes very-quiet bands (silences) only mildly so we don't kill expressiveness.
    """
    loss = 0.0
    for n_fft in (128, 256, 512):
        spec = torch.stft(x, n_fft=n_fft, hop_length=n_fft//4,
                          window=torch.hann_window(n_fft, device=x.device),
                          return_complex=True)
        log_mag = torch.log(spec.abs() + 1e-6)
        loss = loss + F.relu(target_log_mag_floor - log_mag).mean()
    return loss


# ---------- real-Opus eval (ffmpeg) ----------
WORK = Path(tempfile.gettempdir()) / "neural_codec"
WORK.mkdir(exist_ok=True)
def real_opus_rt(audio_np: np.ndarray, bitrate_kbps: int = 24) -> np.ndarray:
    """audio_np: (N,) float32 in [-1,1]. Returns same-length float32."""
    tag = str(np.random.randint(1, 1<<30))
    inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
    sf.write(inp, audio_np.astype(np.float32), SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                    "-c:a","libopus","-b:a",f"{bitrate_kbps}k",
                    "-application","voip",str(opx)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                    "-ar",str(SR),"-ac","1",str(out)], check=True)
    a, _ = sf.read(out)
    inp.unlink(missing_ok=True); opx.unlink(missing_ok=True); out.unlink(missing_ok=True)
    if len(a) < len(audio_np):
        a = np.pad(a, (0, len(audio_np) - len(a)))
    elif len(a) > len(audio_np):
        a = a[:len(audio_np)]
    return a.astype(np.float32)


def real_opus_batch(audio_t: torch.Tensor, batch_concat: bool = True) -> torch.Tensor:
    """audio_t: (B, SYMBOL_N) on any device. Returns (B, SYMBOL_N) on same device.
    If batch_concat: pack the whole batch into one ffmpeg call to save shellout overhead.
    """
    audio_np = audio_t.detach().cpu().numpy()
    B, N = audio_np.shape
    if batch_concat:
        flat = audio_np.reshape(-1)  # (B*N,)
        rt = real_opus_rt(flat)
        rt = rt[: B*N].reshape(B, N)
    else:
        rt = np.stack([real_opus_rt(a) for a in audio_np], axis=0)
    return torch.from_numpy(rt).to(audio_t.device)


# ---------- training ----------
def make_train_step(encoder, decoder, opt, n_bits, perc_weight=0.005, snr_db=18.0):
    def step(batch_size: int, surrogate: bool = True):
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        audio = encoder(bits)
        if surrogate:
            audio_codec = opus_surrogate(audio, snr_db=snr_db)
        else:
            # Real Opus + straight-through estimator: forward through libopus,
            # but pass gradients straight through the encoder->decoder boundary.
            with torch.no_grad():
                audio_codec_no_grad = real_opus_batch(audio)
            # Straight-through: the forward op is real Opus, the backward op is identity
            audio_codec = audio + (audio_codec_no_grad - audio).detach()
        logits = decoder(audio_codec)
        loss_bit = F.binary_cross_entropy_with_logits(logits, bits)
        loss_perc = stft_loss(audio)
        loss = loss_bit + perc_weight * loss_perc
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            ber = (preds != bits).float().mean().item()
        return dict(loss=loss.item(), loss_bit=loss_bit.item(),
                    loss_perc=loss_perc.item(), ber=ber)
    return step


def eval_real_opus(encoder, decoder, n_bits, n_batches=4, batch_size=64):
    encoder.eval(); decoder.eval()
    bits_all = []; preds_all = []
    with torch.no_grad():
        for _ in range(n_batches):
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            audio = encoder(bits)
            audio_codec = real_opus_batch(audio)
            logits = decoder(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bits_all.append(bits); preds_all.append(preds)
    bits_all = torch.cat(bits_all); preds_all = torch.cat(preds_all)
    ber = (preds_all != bits_all).float().mean().item()
    encoder.train(); decoder.train()
    return ber


# ---------- run ----------
def train(n_bits: int = 32, n_steps: int = 4000, batch_size: int = 256,
          eval_every: int = 500, lr: float = 1e-3, perc_weight: float = 0.005,
          snr_db: float = 18.0, ckpt_path: str | None = None,
          real_opus_every: int = 0, real_opus_batch_size: int = 64,
          real_opus_warmup: int = 0):
    """real_opus_every: if >0, run a real-Opus straight-through step every N surrogate steps.
       real_opus_warmup: train this many surrogate-only steps before mixing in real Opus."""
    print(f"# device={DEVICE}  n_bits={n_bits} ({n_bits*1000//SYMBOL_MS} bps raw)  "
          f"batch={batch_size}  steps={n_steps}  perc_w={perc_weight}  snr_db={snr_db}")
    if real_opus_every:
        print(f"# real_opus_every={real_opus_every}  (warmup={real_opus_warmup}, "
              f"ro_batch={real_opus_batch_size})")
    enc = Encoder(n_bits=n_bits).to(DEVICE)
    dec = Decoder(n_bits=n_bits).to(DEVICE)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"# params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    step_fn = make_train_step(enc, dec, opt, n_bits, perc_weight=perc_weight, snr_db=snr_db)
    print(f"{'step':>6} {'loss':>7} {'bit_loss':>8} {'perc':>7} {'surrogate_BER':>14} "
          f"{'realOpusStepBER':>15} {'real_opus_eval':>14}")
    last_ro_step_ber = float("nan")
    for step in range(1, n_steps + 1):
        m = step_fn(batch_size, surrogate=True)
        # Mixed: occasional real-Opus straight-through step
        if (real_opus_every and step > real_opus_warmup
                and step % real_opus_every == 0):
            mr = step_fn(real_opus_batch_size, surrogate=False)
            last_ro_step_ber = mr["ber"]
        if step == 1 or step % 100 == 0 or step == n_steps:
            line = (f"{step:>6} {m['loss']:>7.4f} {m['loss_bit']:>8.4f} "
                    f"{m['loss_perc']:>7.4f} {m['ber']*100:>12.2f}% "
                    f"{last_ro_step_ber*100:>13.2f}%" if not math.isnan(last_ro_step_ber)
                    else f"{step:>6} {m['loss']:>7.4f} {m['loss_bit']:>8.4f} "
                         f"{m['loss_perc']:>7.4f} {m['ber']*100:>12.2f}% {'-':>14}")
            if step % eval_every == 0 or step == n_steps:
                ro_ber = eval_real_opus(enc, dec, n_bits)
                line += f" {ro_ber*100:>12.2f}%"
            print(line, flush=True)
    if ckpt_path:
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                    "n_bits": n_bits}, ckpt_path)
        print(f"# saved checkpoint to {ckpt_path}")
    return enc, dec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bits", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--perc_weight", type=float, default=0.005)
    ap.add_argument("--snr_db", type=float, default=18.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--real_opus_every", type=int, default=0,
                    help="If >0, do a real-Opus straight-through step every N surrogate steps.")
    ap.add_argument("--real_opus_batch", type=int, default=64)
    ap.add_argument("--real_opus_warmup", type=int, default=0)
    args = ap.parse_args()
    train(n_bits=args.n_bits, n_steps=args.steps, batch_size=args.batch,
          lr=args.lr, perc_weight=args.perc_weight, snr_db=args.snr_db,
          eval_every=args.eval_every, ckpt_path=args.ckpt,
          real_opus_every=args.real_opus_every,
          real_opus_batch_size=args.real_opus_batch,
          real_opus_warmup=args.real_opus_warmup)
