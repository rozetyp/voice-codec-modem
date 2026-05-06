"""Realism eval for the sequence-level codec.

Compares the seq model's 120 ms output (which can have cross-symbol structure) to:
 - real human speech (positive class)
 - the IID adversarial model's output (best speech-like baseline so far)
 - the IID baseline (modem-tone control)
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import scipy.signal as sps
from scipy.stats import ks_2samp
import soundfile as sf
import torch

from neural_codec import Encoder as IIDEncoder, DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, BLOCK_N, N_CHUNKS
from realism_eval import mel_spec, spectral_centroid, zcr
from adversarial_realism import RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000


def encode_iid(ckpt_path, n_blocks=64):
    """Generate 1920-sample blocks from an IID encoder by stitching 4 × 30 ms outputs."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc = IIDEncoder(n_bits=ckpt["n_bits"]).to(DEVICE); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    with torch.no_grad():
        bits = torch.randint(0, 2, (n_blocks * N_CHUNKS, ckpt["n_bits"]), device=DEVICE).float()
        chunks = enc(bits).cpu().numpy()  # (n_blocks*4, 480)
    # Concatenate consecutive 4 chunks per block
    return chunks.reshape(n_blocks, N_CHUNKS, SYMBOL_N).reshape(n_blocks, BLOCK_N).astype(np.float32)


def encode_seq(ckpt_path, n_blocks=64):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits_per_chunk = ckpt["n_bits_per_chunk"]
    enc = SeqEncoder(n_bits_per_chunk).to(DEVICE); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    with torch.no_grad():
        bits = torch.randint(0, 2, (n_blocks, n_bits_per_chunk * N_CHUNKS), device=DEVICE).float()
        return enc(bits).cpu().numpy().astype(np.float32)


def boundary_discontinuity(blocks, chunk_n=SYMBOL_N):
    """For each 120ms block, measure energy of the discontinuity at the 30ms chunk
    boundaries. Real speech and seq-model output should have low boundary energy;
    IID-stitched output may have audible seams."""
    out = []
    for b in blocks:
        for i in range(1, N_CHUNKS):
            j = i * chunk_n
            # |x[j] - x[j-1]| relative to local RMS
            local = b[max(0, j-50):j+50]
            rms = np.sqrt(np.mean(local**2)) + 1e-9
            d = abs(b[j] - b[j-1]) / rms
            out.append(d)
    return np.array(out)


def run(checkpoints, real_paths):
    sampler = RealSpeechSampler(real_paths)
    # Stack 30 ms chunks into 120 ms blocks for fair comparison
    n_blocks = 256
    real_chunks = sampler.clips[np.random.choice(len(sampler.clips), n_blocks * N_CHUNKS, replace=False)]
    real = real_chunks.reshape(n_blocks, N_CHUNKS, SYMBOL_N).reshape(n_blocks, BLOCK_N).astype(np.float32)
    real_mel = mel_spec(real); real_cent = spectral_centroid(real); real_zcr = zcr(real)
    real_disc = boundary_discontinuity(real)
    print(f"# real reference: {n_blocks} x 120ms blocks; boundary jump (median) = {np.median(real_disc):.4f}")

    print(f"\n{'label':<32} {'mel_L1':>7} {'cent_KS':>8} {'zcr_KS':>7} {'bndJump_median':>15}")
    for label, kind, path in checkpoints:
        if not Path(path).exists():
            print(f"  {label:<30} (missing {path})"); continue
        if kind == "iid":
            blocks = encode_iid(path, n_blocks=n_blocks)
        elif kind == "seq":
            blocks = encode_seq(path, n_blocks=n_blocks)
        else:
            raise ValueError(kind)
        # Match RMS to real
        rms_real = np.sqrt(np.mean(real**2))
        rms_fake = np.sqrt(np.mean(blocks**2))
        blocks_n = blocks * (rms_real / (rms_fake + 1e-9))
        f_mel = mel_spec(blocks_n); f_cent = spectral_centroid(blocks_n); f_zcr = zcr(blocks_n)
        mel_l1 = float(np.mean(np.abs(f_mel.mean(axis=(0,2)) - real_mel.mean(axis=(0,2)))))
        cent_ks, _ = ks_2samp(f_cent, real_cent)
        zcr_ks, _ = ks_2samp(f_zcr, real_zcr)
        bnd = boundary_discontinuity(blocks)
        print(f"  {label:<30} {mel_l1:>6.3f}  {cent_ks:>6.3f}  {zcr_ks:>6.3f}  {np.median(bnd):>14.4f}")
        # Save 2 sec audio
        out_dir = Path(str(REPO_ROOT / "core" / "neural_codec" / "sequence" / "seq_realism_samples")); out_dir.mkdir(exist_ok=True)
        clip = blocks_n.reshape(-1)[:SR*2]
        clip = clip * (np.sqrt(np.mean(real**2)) / (np.sqrt(np.mean(clip**2)) + 1e-9))
        sf.write(out_dir / f"sample_{label.replace(' ', '_').replace('/', '_').replace('.', '_')}.wav",
                 np.clip(clip, -1, 1).astype(np.float32), SR)
    # Real reference clip
    out_dir = Path(str(REPO_ROOT / "core" / "neural_codec" / "sequence" / "seq_realism_samples")); out_dir.mkdir(exist_ok=True)
    real_clip = real.reshape(-1)[:SR*2]
    sf.write(out_dir / "sample_real_speech.wav",
             np.clip(real_clip, -1, 1).astype(np.float32), SR)
    print(f"\n# samples saved to {out_dir}/")


if __name__ == "__main__":
    paths = [
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav")),
        Path(str(REPO_ROOT / "tests/data/raw/lex_jensen.wav")),
    ]
    checkpoints = [
        ("IID baseline n=32",        "iid", str(REPO_ROOT / "core/neural_codec/ckpt_n32_mixed.pt")),
        ("IID adv n=64",             "iid", str(REPO_ROOT / "core/neural_codec/adversarial/ckpt_n64_adv.pt")),
        ("IID adv n=128",            "iid", str(REPO_ROOT / "core/neural_codec/adversarial/ckpt_n128_adv.pt")),
        ("SEQ no-adv (this run)",    "seq", str(REPO_ROOT / "core" / "neural_codec" / "sequence" / "ckpt_seq_v2.pt")),
        ("SEQ adversarial (this run)","seq", str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt")),
    ]
    run(checkpoints, paths)
