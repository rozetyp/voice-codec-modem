"""Save listenable audio: cover / modified / through-AMR-NB / decoded."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import soundfile as sf
import torch

from neural_codec import DEVICE, SR
from stego_codec import StegEncoder, StegDecoder
from stego_amrnb import amrnb_round_trip_batch
from adversarial_realism import RealSpeechSampler


def main():
    ckpt_path = str(REPO_ROOT / "core/neural_codec/cellular/ckpt_amrnb_real.pt")
    out_dir = Path(str(REPO_ROOT / "core" / "neural_codec" / "cellular" / "listen_samples")); out_dir.mkdir(exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]; pert = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits).to(DEVICE); dec = StegDecoder(n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    sampler = RealSpeechSampler([
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav")),
    ])
    torch.manual_seed(0)
    n_blocks = 100  # 3 seconds
    cover = sampler.sample(n_blocks)
    bits = torch.randint(0, 2, (n_blocks, n_bits), device=DEVICE).float()
    with torch.no_grad():
        modified, _ = enc(cover, bits, perturbation_scale=pert)
        rx = amrnb_round_trip_batch(modified)
        cover_through = amrnb_round_trip_batch(cover)
        logits = dec(rx)
        preds = (torch.sigmoid(logits) > 0.5).float()
        ber = (preds != bits).float().mean().item()

    cover_np = cover.cpu().numpy().reshape(-1).astype(np.float32)
    modified_np = modified.cpu().numpy().reshape(-1).astype(np.float32)
    rx_np = rx.cpu().numpy().reshape(-1).astype(np.float32)
    cover_through_np = cover_through.cpu().numpy().reshape(-1).astype(np.float32)
    delta = modified_np - cover_np
    cover_pwr = np.mean(cover_np**2); delta_pwr = np.mean(delta**2)
    snr_db = 10 * np.log10(cover_pwr / (delta_pwr + 1e-12))

    print(f"# {n_blocks * 30}ms = {n_blocks*30/1000:.1f}s of audio  ({n_blocks * n_bits} bits)")
    print(f"# real-AMR-NB BER (raw, no FEC): {ber*100:.2f}%")
    print(f"# perturbation SNR vs cover: {snr_db:.1f} dB")

    sf.write(out_dir / "01_cover_real_speech.wav", np.clip(cover_np, -1, 1), SR)
    sf.write(out_dir / "02_cover_through_amrnb_only.wav", np.clip(cover_through_np, -1, 1), SR)
    sf.write(out_dir / "03_modified_with_data.wav", np.clip(modified_np, -1, 1), SR)
    sf.write(out_dir / "04_modified_through_amrnb.wav", np.clip(rx_np, -1, 1), SR)
    sf.write(out_dir / "05_perturbation_only.wav", np.clip(delta / max(np.abs(delta).max(), 1e-9), -1, 1), SR)
    print(f"# wrote {out_dir}/")


if __name__ == "__main__":
    main()
