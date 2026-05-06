"""
Quantify how speech-like the encoder output is, and compare to real speech and to
the baseline (non-adversarial) checkpoint.

Metrics:
  1. Mel-spectrogram L1 distance to real speech (lower = more speech-like)
  2. Spectral centroid distribution (KS distance to real speech)
  3. Zero-crossing-rate distribution (KS distance to real speech)
  4. Trained discriminator's confidence (if the disc is loaded)

Also runs the discriminator from a baseline-vs-adversarial cross-eval: load the trained
discriminator, ask it to classify each model's output. Lower fool rate = more speech-like.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

import numpy as np
import soundfile as sf
import scipy.signal as sps
from scipy.stats import ks_2samp
import torch
import torch.nn.functional as F

from neural_codec import Encoder, Decoder, real_opus_batch, DEVICE, SR, SYMBOL_MS
from adversarial_realism import Discriminator, RealSpeechSampler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SYMBOL_N = SR * SYMBOL_MS // 1000


def encoder_output(ckpt_path: str, n_batches: int = 8, batch_size: int = 64):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = Encoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    chunks = []
    with torch.no_grad():
        for _ in range(n_batches):
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            chunks.append(enc(bits).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32), n_bits


def mel_spec(audio_np, n_mels=40, n_fft=256, hop=64):
    """Quick log-mel spectrogram."""
    f, t, S = sps.stft(audio_np, fs=SR, nperseg=n_fft, noverlap=n_fft-hop,
                       boundary=None, padded=False, axis=-1)
    P = np.abs(S) ** 2  # (B, F, T)
    # Linear mel filterbank (rough but consistent)
    f_mel = np.linspace(0, SR/2, n_mels + 2)
    mel = np.zeros((n_mels, P.shape[1]))
    for i in range(n_mels):
        lo, hi = f_mel[i], f_mel[i+2]
        mid = f_mel[i+1]
        for j, fj in enumerate(f):
            if lo <= fj <= mid: mel[i, j] = (fj - lo) / (mid - lo + 1e-9)
            elif mid < fj <= hi: mel[i, j] = (hi - fj) / (hi - mid + 1e-9)
    M = np.einsum('mf,bft->bmt', mel, P) + 1e-8
    return np.log(M)


def spectral_centroid(audio_np, n_fft=256):
    """Per-window spectral centroid (Hz). Returns (B*T,) array."""
    f, t, S = sps.stft(audio_np, fs=SR, nperseg=n_fft, noverlap=0, boundary=None, padded=False, axis=-1)
    P = np.abs(S)
    cent = np.einsum('f,bft->bt', f, P) / (P.sum(axis=1) + 1e-9)
    return cent.reshape(-1)


def zcr(audio_np):
    """Per-chunk zero-crossing rate."""
    s = np.sign(audio_np); s[s == 0] = 1
    return (np.abs(np.diff(s, axis=-1)) / 2).mean(axis=-1)


def evaluate(ckpts: dict, real_paths: list[Path], disc_ckpt: str | None = None):
    sampler = RealSpeechSampler(real_paths)
    real = sampler.clips[np.random.choice(len(sampler.clips), 512, replace=False)]
    print(f"# real corpus reference: 512 chunks")
    real_mel = mel_spec(real)
    real_cent = spectral_centroid(real)
    real_zcr = zcr(real)

    disc = None
    if disc_ckpt and Path(disc_ckpt).exists():
        st = torch.load(disc_ckpt, map_location=DEVICE)
        if "discriminator" in st:
            disc = Discriminator().to(DEVICE)
            disc.load_state_dict(st["discriminator"]); disc.eval()
            print(f"# loaded discriminator from {disc_ckpt}")

    print(f"\n{'ckpt':<35} {'n_bits':>6} {'mel_L1':>7} {'cent_KS':>8} {'zcr_KS':>7} {'disc_realProb':>13}")
    for label, path in ckpts.items():
        audio, n_bits = encoder_output(path)
        # Match RMS to real (so we don't unfairly penalize loudness)
        rms_real = np.sqrt(np.mean(real**2))
        rms_fake = np.sqrt(np.mean(audio**2))
        audio_n = audio * (rms_real / (rms_fake + 1e-9))

        f_mel = mel_spec(audio_n)
        f_cent = spectral_centroid(audio_n)
        f_zcr = zcr(audio_n)

        mel_l1 = float(np.mean(np.abs(f_mel.mean(axis=(0,2)) - real_mel.mean(axis=(0,2)))))
        cent_ks, _ = ks_2samp(f_cent, real_cent)
        zcr_ks, _ = ks_2samp(f_zcr, real_zcr)

        if disc is not None:
            with torch.no_grad():
                d_logits = disc(torch.from_numpy(audio).to(DEVICE))
                p_real = torch.sigmoid(d_logits).mean().item()
            disc_str = f"{p_real*100:>11.2f}%"
        else:
            disc_str = "    -"
        print(f"  {label:<33} {n_bits:>4}  {mel_l1:>6.3f}  {cent_ks:>6.3f}  {zcr_ks:>6.3f}  {disc_str}")

    # Save sample audio for listening: 2 seconds = 2000ms / 30ms = 66 chunks
    out_dir = Path(str(REPO_ROOT / "core" / "neural_codec" / "adversarial" / "realism_samples"))
    out_dir.mkdir(exist_ok=True)
    for label, path in ckpts.items():
        audio, n_bits = encoder_output(path, n_batches=2, batch_size=66)
        clip = audio.reshape(-1)[:SR*2]  # 2s
        # Match real speech RMS for fair listening
        clip = clip * (np.sqrt(np.mean(real**2)) / (np.sqrt(np.mean(clip**2)) + 1e-9))
        clip = np.clip(clip, -1, 1).astype(np.float32)
        path = out_dir / f"sample_{label.replace('/', '_').replace('.', '_')}.wav"
        sf.write(path, clip, SR)
        print(f"# wrote {path}")
    # Also save 2 sec of real speech for reference
    real_clip = real.reshape(-1)[:SR*2]
    sf.write(out_dir / "sample_real_speech.wav",
             np.clip(real_clip, -1, 1).astype(np.float32), SR)
    print(f"# wrote {out_dir}/sample_real_speech.wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(REPO_ROOT / "core/neural_codec/ckpt_n64_mixed.pt"))
    ap.add_argument("--adv", default=str(REPO_ROOT / "core/neural_codec/adversarial/ckpt_n64_adv.pt"))
    args = ap.parse_args()
    paths = [
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav")),
        Path(str(REPO_ROOT / "tests/data/raw/lex_jensen.wav")),
    ]
    ckpts = {}
    if Path(args.baseline).exists(): ckpts["baseline (n64_mixed)"] = args.baseline
    if Path(args.adv).exists(): ckpts["adversarial (n64_adv)"] = args.adv
    evaluate(ckpts, paths, disc_ckpt=args.adv)
