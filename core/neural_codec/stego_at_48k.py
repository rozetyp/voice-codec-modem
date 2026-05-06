"""Test stego embedding at 48 kbps Opus AUDIO mode."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import numpy as np
import torch

from neural_codec import DEVICE, SR, SYMBOL_MS
from stego_codec import StegEncoder, StegDecoder
from adversarial_realism import RealSpeechSampler
from opus_mode_compare import opus_round_trip_batch


def main():
    ckpt_path = str(REPO_ROOT / "core/neural_codec/stego/ckpt_stego_p3.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    pert_scale = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits).to(DEVICE); dec = StegDecoder(n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()

    sampler = RealSpeechSampler([
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav"))
    ])

    print(f"# stego ckpt: {ckpt_path}  n_bits={n_bits}  perturbation_scale={pert_scale}")
    print(f"# raw bitrate: {n_bits * 1000 / SYMBOL_MS:.0f} bps")
    print()
    print(f"{'Opus config':<20}  {'real_BER':>9}  {'snr_dB':>7}")
    for label, application, bitrate in [
        ("24k VOIP", "voip", 24),
        ("24k AUDIO", "audio", 24),
        ("48k AUDIO", "audio", 48),
    ]:
        bers = []; snrs = []
        for seed in range(4):
            torch.manual_seed(seed)
            cover = sampler.sample(64)
            bits = torch.randint(0, 2, (64, n_bits), device=DEVICE).float()
            with torch.no_grad():
                modified, _ = enc(cover, bits, perturbation_scale=pert_scale)
                rx = opus_round_trip_batch(modified, application=application, bitrate_kbps=bitrate)
                logits = dec(rx)
                preds = (torch.sigmoid(logits) > 0.5).float()
                bers.append((preds != bits).float().mean().item())
                cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
                pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
                snrs.append((10 * torch.log10(cover_pwr / pert_pwr)).mean().item())
        print(f"  {label:<18}   {np.mean(bers)*100:>6.2f}%   {np.mean(snrs):>5.1f}")


if __name__ == "__main__":
    main()
