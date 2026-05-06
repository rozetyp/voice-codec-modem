"""Save listenable audio: cover / modified / through-Opus / decoded / mismatch."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import soundfile as sf
import torch

from neural_codec import real_opus_batch, DEVICE, SR, SYMBOL_MS
from stego_codec import StegEncoder, StegDecoder
from adversarial_realism import RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/stego/ckpt_stego_p2.pt"))
    ap.add_argument("--n_blocks", type=int, default=66)  # ~2 sec at 30ms per block
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "stego" / "listen_samples"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    pert_scale = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits).to(DEVICE); dec = StegDecoder(n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()

    sampler = RealSpeechSampler([
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav"))
    ])

    torch.manual_seed(0)
    cover = sampler.sample(args.n_blocks)
    bits = torch.randint(0, 2, (args.n_blocks, n_bits), device=DEVICE).float()

    with torch.no_grad():
        modified, _ = enc(cover, bits, perturbation_scale=pert_scale)
        rx = real_opus_batch(modified)
        cover_through_opus = real_opus_batch(cover)  # for comparison
        logits = dec(rx)
        preds = (torch.sigmoid(logits) > 0.5).float()
        ber = (preds != bits).float().mean().item()

    cover_np = cover.cpu().numpy().reshape(-1).astype(np.float32)
    modified_np = modified.cpu().numpy().reshape(-1).astype(np.float32)
    rx_np = rx.cpu().numpy().reshape(-1).astype(np.float32)
    cover_opus_np = cover_through_opus.cpu().numpy().reshape(-1).astype(np.float32)
    delta_np = modified_np - cover_np

    # Compute SNR of perturbation vs cover
    cover_pwr = np.mean(cover_np ** 2)
    delta_pwr = np.mean(delta_np ** 2)
    snr_db = 10 * np.log10(cover_pwr / (delta_pwr + 1e-12))

    print(f"# n_blocks={args.n_blocks} ({args.n_blocks * SYMBOL_MS / 1000:.2f}s)  n_bits/block={n_bits}")
    print(f"# bits transmitted: {args.n_blocks * n_bits}")
    print(f"# real-Opus BER: {ber*100:.2f}%")
    print(f"# perturbation SNR: {snr_db:.1f} dB")

    sf.write(out_dir / "01_cover_real_speech.wav", np.clip(cover_np, -1, 1), SR)
    sf.write(out_dir / "02_cover_through_opus_only.wav", np.clip(cover_opus_np, -1, 1), SR)
    sf.write(out_dir / "03_modified_with_data.wav", np.clip(modified_np, -1, 1), SR)
    sf.write(out_dir / "04_modified_through_opus.wav", np.clip(rx_np, -1, 1), SR)
    sf.write(out_dir / "05_perturbation_only.wav",
             np.clip(delta_np / max(np.abs(delta_np).max(), 1e-9), -1, 1), SR)

    print(f"# wrote {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<48}  {f.stat().st_size:>10} bytes")


if __name__ == "__main__":
    main()
