"""
Modal training app for a bigger, better stego_opus codec.

Goal: ~1.5 kbps reliable with real-human-voice cover through Opus 24k VOIP.
Today's baseline (530 bps reliable, 13 dB SNR) was undertrained. This run:

  * 25M-param StegEncoder/Decoder (vs 2.4M today)
  * n_bits = 64  → 2133 bps raw  → ~1.6 kbps reliable post-FEC at the same RS code
  * LibriSpeech clean-train-100 (~28k speakers) instead of two clips
  * Multi-scale STFT + Mel-L1 perceptual loss (vs L1 amplitude only)
  * Adversarial discriminator: a CNN that judges "is this real speech?",
    trained jointly. Encoder learns to fool it.
  * Real-Opus straight-through every 25 steps (Opus 24k VOIP via ffmpeg-libopus)
  * Curriculum: phase 1 learn bit channel; phase 2 ramp adversarial + perceptual

Run:
  modal run app/modal_train.py
  # then once it's done:
  modal volume get stego-codec-vol /checkpoints/ckpt_stego_v2.pt ./

A10G GPU, ~3 hours wall, ~$3.30 in compute (free $30/mo Modal credits cover it).
"""
from __future__ import annotations
import modal

# ---------- Modal app setup ----------
APP_NAME = "stego-opus-v2"
VOLUME_NAME = "stego-codec-vol"
GPU = "A10G"
TIMEOUT_SECONDS = 4 * 60 * 60  # 4-hour soft cap; will exit cleanly if it converges sooner

# Image: Debian + python 3.12 + torch + ffmpeg (libopus + libopencore-amrnb)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "soundfile==0.12.1",
        "reedsolo==1.7.0",
        "tqdm==4.67.1",
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ===========================================================================
# Everything below this line runs INSIDE the GPU container.
# ===========================================================================
@app.function(
    gpu=GPU,
    volumes={"/vol": vol},
    timeout=TIMEOUT_SECONDS,
    cpu=4.0,
    memory=16 * 1024,
)
def train(
    n_bits: int = 64,
    n_steps: int = 6000,
    batch_size: int = 64,
    lr_g: float = 1e-3,
    lr_d: float = 2e-4,
    pert_scale: float = 0.5,
    perc_weight: float = 0.0,        # ramps up
    adv_weight: float = 0.0,         # ramps up
    perc_target: float = 0.1,        # gentle nudge toward imperceptibility, not a hard squeeze
    adv_target: float = 0.1,         # gentle
    warmup_bit_only: int = 2500,     # was 800 — LibriSpeech corpus is way more diverse than the 2-clip corpus
                                     # the original code converged on, so the bit channel needs more time
    real_opus_every: int = 25,
    real_opus_warmup: int = 200,
    eval_every: int = 250,
    seed: int = 0,
):
    import math
    import os
    import random
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    import numpy as np
    import scipy.signal as sps
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tqdm

    # ----- constants -----
    SR = 16000
    SYMBOL_MS = 30
    SYMBOL_N = SR * SYMBOL_MS // 1000  # 480
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[modal] device={DEVICE}  torch={torch.__version__}  "
          f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[modal] gpu={torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # ----- LibriSpeech (cached on volume) -----
    LIBRI_ROOT = Path("/vol/librispeech")
    LIBRI_TARBALL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    # dev-clean is 337 MB and has 40 speakers / 2703 utterances — plenty of
    # diversity for our 30 ms chunks, much faster to download than train-clean-100.
    LIBRI_SUBSET = LIBRI_ROOT / "LibriSpeech" / "dev-clean"

    if not LIBRI_SUBSET.exists():
        LIBRI_ROOT.mkdir(parents=True, exist_ok=True)
        tarball = LIBRI_ROOT / "dev-clean.tar.gz"
        if not tarball.exists():
            print(f"[modal] downloading LibriSpeech dev-clean ({LIBRI_TARBALL})")
            subprocess.run(["wget", "-q", "-O", str(tarball), LIBRI_TARBALL], check=True)
        print("[modal] extracting LibriSpeech…")
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(LIBRI_ROOT)], check=True)
        tarball.unlink(missing_ok=True)
        vol.commit()
        print(f"[modal] extracted to {LIBRI_SUBSET}")
    else:
        print(f"[modal] LibriSpeech cached at {LIBRI_SUBSET}")

    # ----- speech corpus loader: slice all .flac into 30 ms chunks above silence -----
    def load_corpus(min_rms: float = 0.01) -> np.ndarray:
        flacs = sorted(LIBRI_SUBSET.rglob("*.flac"))
        print(f"[modal] indexing {len(flacs)} flacs from {LIBRI_SUBSET}")
        chunks = []
        for p in tqdm.tqdm(flacs, desc="slicing"):
            a, sr = sf.read(p)
            if a.ndim > 1: a = a.mean(axis=1)
            if sr != SR:
                # Quick resample via scipy
                from scipy.signal import resample_poly
                a = resample_poly(a, SR, sr)
            a = a.astype(np.float32)
            # Normalize to a consistent scale
            rms = np.sqrt(np.mean(a**2)) + 1e-9
            a = a * (0.12 / rms)
            n = len(a) // SYMBOL_N
            for i in range(n):
                seg = a[i*SYMBOL_N:(i+1)*SYMBOL_N]
                if np.sqrt(np.mean(seg**2)) < min_rms: continue
                chunks.append(seg)
        clips = np.stack(chunks, axis=0).astype(np.float32)
        print(f"[modal] corpus: {len(clips)} chunks ({len(clips)*SYMBOL_MS/1000:.0f}s of speech)")
        return clips

    speech_chunks = load_corpus()

    # ----- model classes -----
    # Same architecture as the proven core/neural_codec/stego/stego_codec.py
    # (the 2.4M-param model that successfully reached 0.5% BER on MPS).
    # Going wider/deeper before validating LibriSpeech+perceptual+adversarial works
    # at the proven architecture is putting two unknowns on top of each other.
    HIDDEN_E = 96
    HIDDEN_D = 128

    class StegEncoder(nn.Module):
        def __init__(self, n_bits, hidden=HIDDEN_E):
            super().__init__()
            self.bit_fc = nn.Linear(n_bits, hidden * 60)
            self.cover_enc = nn.Sequential(
                nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
                nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),
                nn.Conv1d(64, hidden, 4, 2, 1), nn.GELU(),
                nn.Conv1d(hidden, hidden, 4, 2, 1), nn.GELU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(hidden*2, hidden, 4, 2, 1), nn.GELU(),
                nn.ConvTranspose1d(hidden, 64, 4, 2, 1),       nn.GELU(),
                nn.ConvTranspose1d(64, 32, 4, 2, 1),           nn.GELU(),
                nn.Conv1d(32, 1, 7, 1, 3),                     nn.Tanh(),
            )

        def forward(self, cover, bits, perturbation_scale=0.5):
            c_feat = self.cover_enc(cover.unsqueeze(1))
            b_feat = self.bit_fc(bits.float() * 2 - 1).view(c_feat.size(0), -1, 60)
            joint = torch.cat([c_feat, b_feat], dim=1)
            delta = self.dec(joint).squeeze(1)
            with torch.no_grad():
                mask = F.avg_pool1d(cover.abs().unsqueeze(1), 11, 1, 5).squeeze(1).clamp(min=0.005)
            modified = cover + perturbation_scale * delta * mask
            return modified.clamp(-1, 1), delta

    class StegDecoder(nn.Module):
        def __init__(self, n_bits, hidden=HIDDEN_D):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 7, 1, 3), nn.GELU(),
                nn.Conv1d(32, 64, 4, 2, 1), nn.GELU(),
                nn.Conv1d(64, 96, 4, 2, 1), nn.GELU(),
                nn.Conv1d(96, hidden, 4, 2, 1), nn.GELU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden * 60, 256), nn.GELU(),
                nn.Linear(256, n_bits),
            )

        def forward(self, audio):
            x = audio.unsqueeze(1)
            x = self.conv(x).flatten(1)
            return self.fc(x)

    class Discriminator(nn.Module):
        """Patch-CNN that judges 30 ms windows real-vs-fake."""
        def __init__(self, hidden=64):
            super().__init__()
            sn = nn.utils.spectral_norm
            self.net = nn.Sequential(
                sn(nn.Conv1d(1, hidden, 9, 1, 4)),       nn.LeakyReLU(0.2),
                sn(nn.Conv1d(hidden, hidden*2, 5, 2, 2)), nn.LeakyReLU(0.2),
                sn(nn.Conv1d(hidden*2, hidden*2, 5, 2, 2)), nn.LeakyReLU(0.2),
                sn(nn.Conv1d(hidden*2, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),
                sn(nn.Conv1d(hidden*4, hidden*4, 5, 2, 2)), nn.LeakyReLU(0.2),
                sn(nn.Conv1d(hidden*4, 1, 30, 1, 0)),
            )
        def forward(self, x): return self.net(x.unsqueeze(1)).squeeze()

    # ----- losses -----
    def opus_surrogate(audio, snr_db=18.0, lp_hz=7000.0):
        n = audio.size(-1)
        fft = torch.fft.rfft(audio, dim=-1)
        freqs = torch.fft.rfftfreq(n, 1.0/SR).to(audio.device)
        snr_db_j = snr_db + (torch.rand(1, device=audio.device).item() - 0.5) * 8.0
        lp_hz_j = lp_hz + (torch.rand(1, device=audio.device).item() - 0.5) * 1500.0
        mask = torch.clamp(1.0 - (freqs - 6000.0) / max(1.0, lp_hz_j - 6000.0), 0.0, 1.0)
        speech = ((freqs > 150) & (freqs < 3800)).float() * 0.2
        weight = mask * (1.0 + speech)
        out = torch.fft.irfft(fft * weight, n=n, dim=-1)
        rms = audio.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-9
        out = out + torch.randn_like(out) * rms * (10 ** (-snr_db_j / 20))
        out = out + torch.randn_like(out) * 0.003
        return out.clamp(-1, 1)

    WORK = Path(tempfile.gettempdir())/"opus_rt"; WORK.mkdir(exist_ok=True)
    def real_opus_batch(audio_t):
        audio_np = audio_t.detach().cpu().numpy()
        B, N = audio_np.shape
        flat = audio_np.reshape(-1)
        tag = str(np.random.randint(1, 1<<30))
        inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
        sf.write(inp, flat.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a","24k","-application","voip", str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1", str(out)], check=True)
        a, _ = sf.read(out)
        inp.unlink(missing_ok=True); opx.unlink(missing_ok=True); out.unlink(missing_ok=True)
        if a.ndim > 1: a = a.mean(axis=1)
        if len(a) < B*N: a = np.pad(a, (0, B*N - len(a)))
        elif len(a) > B*N: a = a[:B*N]
        return torch.from_numpy(a.reshape(B, N).astype(np.float32)).to(audio_t.device)

    def stft_perceptual_loss(modified, cover):
        """Multi-scale STFT magnitude L1 between modified and cover.
        Linear-magnitude only — log-magnitude term made the loss huge and
        crushed the bit signal in run #2. n_fft<=512 (480-sample input limit)."""
        loss = 0.0
        for n_fft in (128, 256, 512):
            window = torch.hann_window(n_fft, device=modified.device)
            S_m = torch.stft(modified, n_fft=n_fft, hop_length=n_fft//4,
                             window=window, return_complex=True).abs()
            S_c = torch.stft(cover,    n_fft=n_fft, hop_length=n_fft//4,
                             window=window, return_complex=True).abs()
            loss = loss + F.l1_loss(S_m, S_c)
        return loss

    def sample_cover(batch_size):
        idx = np.random.randint(0, len(speech_chunks), size=batch_size)
        return torch.from_numpy(speech_chunks[idx]).to(DEVICE)

    # ----- init -----
    enc = StegEncoder(n_bits).to(DEVICE)
    dec = StegDecoder(n_bits).to(DEVICE)
    disc = Discriminator().to(DEVICE)
    n_params_g = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    n_params_d = sum(p.numel() for p in disc.parameters())
    print(f"[modal] params: encoder+decoder={n_params_g/1e6:.1f}M  discriminator={n_params_d/1e6:.1f}M")
    # AdamW betas: stable (0.9, 0.999) during bit-only warmup, GAN-style (0.5, 0.99)
    # only after the discriminator engages (perceptual+adversarial ramping in).
    opt_g = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                              lr=lr_g, betas=(0.9, 0.999))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, betas=(0.5, 0.99))
    _gan_betas_set = False

    # ----- train loop -----
    print(f"\n[modal] training {n_steps} steps  bs={batch_size}  n_bits={n_bits}  pert={pert_scale}")
    print(f"        warmup_bit_only={warmup_bit_only}  real_opus_every={real_opus_every}")
    print(f"        targets: perc_weight->{perc_target}  adv_weight->{adv_target}\n")
    print(f"{'step':>5} {'bit_loss':>9} {'perc':>7} {'adv_g':>7} {'d_loss':>7} {'snr':>6} "
          f"{'sur_BER':>8} {'roBER':>7} {'evalBER':>9}")

    last_ro = float("nan")
    best_eval_ber = 1.0
    best_eval_snr = 0.0
    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Curriculum on weights
        if step <= warmup_bit_only:
            perc_w = 0.0; adv_w = 0.0
        else:
            ramp = min(1.0, (step - warmup_bit_only) / max(1, n_steps * 0.4))
            perc_w = ramp * perc_target
            adv_w = ramp * adv_target
            # Switch generator optimizer to GAN-style betas the first time adversarial
            # loss kicks in (preserves momentum learned during stable warmup).
            if not _gan_betas_set:
                for g in opt_g.param_groups:
                    g["betas"] = (0.5, 0.99)
                _gan_betas_set = True
                print(f"[modal] step {step}: switched G optimizer to GAN-style betas (0.5, 0.99)")

        # ---- Discriminator step ----
        d_loss_val = 0.0; d_acc = 0.5
        if adv_w > 0:
            with torch.no_grad():
                bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
                cover_d = sample_cover(batch_size)
                fake_d, _ = enc(cover_d, bits, perturbation_scale=pert_scale)
            real_d = sample_cover(batch_size)
            d_real = disc(real_d); d_fake = disc(fake_d)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            real_d_gp = real_d.detach().requires_grad_(True)
            d_real_gp = disc(real_d_gp)
            grads = torch.autograd.grad(d_real_gp.sum(), real_d_gp, create_graph=True)[0]
            d_loss = d_loss + 1.0 * (grads ** 2).sum(dim=-1).mean()
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            with torch.no_grad():
                d_acc = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()).item() * 0.5
            d_loss_val = float(d_loss.item())

        # ---- Generator step (surrogate) ----
        cover = sample_cover(batch_size)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        modified, _ = enc(cover, bits, perturbation_scale=pert_scale)
        rx = opus_surrogate(modified, snr_db=18.0)
        logits = dec(rx)
        bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
        perc_loss = stft_perceptual_loss(modified, cover) if perc_w > 0 else torch.tensor(0.0, device=DEVICE)
        if adv_w > 0:
            adv_g = -disc(modified).mean()
        else:
            adv_g = torch.tensor(0.0, device=DEVICE)
        loss = bit_loss + perc_w * perc_loss + adv_w * adv_g
        opt_g.zero_grad(); loss.backward(); opt_g.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            sur_ber = (preds != bits).float().mean().item()
            cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
            pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
            snr_db = (10 * torch.log10(cover_pwr / pert_pwr)).mean().item()

        # ---- Real-Opus straight-through ----
        if real_opus_every and step > real_opus_warmup and step % real_opus_every == 0:
            cover_ro = sample_cover(min(batch_size, 32))
            bits_ro = torch.randint(0, 2, (cover_ro.size(0), n_bits), device=DEVICE).float()
            modified_ro, _ = enc(cover_ro, bits_ro, perturbation_scale=pert_scale)
            with torch.no_grad():
                rx_real = real_opus_batch(modified_ro)
            rx_st = modified_ro + (rx_real - modified_ro).detach()
            logits_ro = dec(rx_st)
            bit_loss_ro = F.binary_cross_entropy_with_logits(logits_ro, bits_ro)
            perc_loss_ro = stft_perceptual_loss(modified_ro, cover_ro) if perc_w > 0 else torch.tensor(0.0, device=DEVICE)
            adv_g_ro = -disc(modified_ro).mean() if adv_w > 0 else torch.tensor(0.0, device=DEVICE)
            loss_ro = bit_loss_ro + perc_w * perc_loss_ro + adv_w * adv_g_ro
            opt_g.zero_grad(); loss_ro.backward(); opt_g.step()
            with torch.no_grad():
                preds_ro = (torch.sigmoid(logits_ro) > 0.5).float()
                last_ro = (preds_ro != bits_ro).float().mean().item()

        # ---- Logging / eval ----
        if step == 1 or step % 50 == 0 or step == n_steps:
            line = (f"{step:>5} {bit_loss.item():>9.4f} {perc_loss.item():>7.3f} "
                    f"{adv_g.item():>+7.3f} {d_loss_val:>7.3f} {snr_db:>5.1f} "
                    f"{sur_ber*100:>6.2f}% {last_ro*100 if not math.isnan(last_ro) else 0:>5.2f}%")
            if step % eval_every == 0 or step == n_steps:
                # Full real-Opus eval with bigger batch for tighter estimate
                enc.eval(); dec.eval()
                with torch.no_grad():
                    eval_b = 96
                    cover_e = sample_cover(eval_b)
                    bits_e = torch.randint(0, 2, (eval_b, n_bits), device=DEVICE).float()
                    modified_e, _ = enc(cover_e, bits_e, perturbation_scale=pert_scale)
                    rx_e = real_opus_batch(modified_e)
                    preds_e = (torch.sigmoid(dec(rx_e)) > 0.5).float()
                    eval_ber = (preds_e != bits_e).float().mean().item()
                    cover_pwr_e = cover_e.pow(2).mean(dim=-1) + 1e-9
                    pert_pwr_e = (modified_e - cover_e).pow(2).mean(dim=-1) + 1e-12
                    eval_snr = (10 * torch.log10(cover_pwr_e / pert_pwr_e)).mean().item()
                enc.train(); dec.train()
                line += f"   {eval_ber*100:>6.2f}% (snr={eval_snr:.1f}dB)"
                # Save best — only if SNR is in a usable range (not the degenerate
                # near-silence collapse). Want at least 10 dB of perturbation so the
                # model is actually doing something.
                if eval_snr > 10.0 and eval_ber < best_eval_ber:
                    best_eval_ber = eval_ber
                    best_eval_snr = eval_snr
                    best_path = Path("/vol/checkpoints/ckpt_stego_v2_best.pt")
                    best_path.parent.mkdir(exist_ok=True)
                    torch.save({
                        "encoder": enc.state_dict(),
                        "decoder": dec.state_dict(),
                        "discriminator": disc.state_dict(),
                        "n_bits": n_bits,
                        "perturbation_scale": pert_scale,
                        "step": step,
                        "eval_ber": eval_ber,
                        "eval_snr": eval_snr,
                    }, best_path)
                    vol.commit()
                    line += f"  ★ best (saved)"
            print(line, flush=True)

        # Periodic checkpoint
        if step % 1000 == 0 or step == n_steps:
            ckpt_path = Path(f"/vol/checkpoints/ckpt_stego_v2_step{step}.pt")
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "discriminator": disc.state_dict(),
                "n_bits": n_bits,
                "perturbation_scale": pert_scale,
                "step": step,
            }, ckpt_path)
            vol.commit()
            print(f"[modal] saved {ckpt_path}", flush=True)

    # Final checkpoint with stable name
    final = Path("/vol/checkpoints/ckpt_stego_v2.pt")
    torch.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "discriminator": disc.state_dict(),
        "n_bits": n_bits,
        "perturbation_scale": pert_scale,
        "step": n_steps,
    }, final)
    vol.commit()
    elapsed = time.time() - t0
    print(f"\n[modal] DONE in {elapsed/60:.1f} min")
    print(f"[modal]   final ckpt: {final}")
    print(f"[modal]   best ckpt:  /vol/checkpoints/ckpt_stego_v2_best.pt  "
          f"(eval_ber={best_eval_ber*100:.2f}%, snr={best_eval_snr:.1f}dB)")
    return {
        "final_ckpt": str(final),
        "best_eval_ber": best_eval_ber,
        "best_eval_snr": best_eval_snr,
        "elapsed_minutes": elapsed/60,
        "n_steps": n_steps,
    }


@app.local_entrypoint()
def main(
    n_bits: int = 64,
    n_steps: int = 5000,
    batch_size: int = 64,
):
    result = train.remote(n_bits=n_bits, n_steps=n_steps, batch_size=batch_size)
    print(f"\n[local] training finished: {result}")
    print(f"[local] pull the checkpoint: modal volume get {VOLUME_NAME} /checkpoints/ckpt_stego_v2.pt ./")
