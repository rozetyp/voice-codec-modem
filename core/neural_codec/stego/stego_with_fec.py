"""Post-FEC payload test for the real-speech stego codec.

data bytes -> interleaved RS -> StegEncoder on real speech -> real ffmpeg libopus -> StegDecoder -> RS decode.
The cover is app/static/cover.wav, the clip every checkout has.

    stego_with_fec.py --application voip --kbps 24 --rs_data 127    # 531 bps
    stego_with_fec.py --application audio --kbps 48 --rs_data 191   # 799 bps
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for _d in ("", "adversarial", "sequence", "stego", "cellular", "composite_attempt"):
    sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec" / _d))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

import argparse

import numpy as np
import torch

from neural_codec import DEVICE, SYMBOL_MS
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes
from stego_codec import StegEncoder, StegDecoder
from adversarial_realism import RealSpeechSampler
from opus_mode_compare import opus_round_trip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/stego/ckpt_stego_p3.pt"))
    ap.add_argument("--application", default="voip", choices=["voip", "audio"])
    ap.add_argument("--kbps", type=int, default=24)
    ap.add_argument("--rs_data", type=int, default=127)
    ap.add_argument("--data_bytes", type=int, default=1900)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    n_bits = ckpt["n_bits"]
    pert_scale = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits).to(DEVICE); dec = StegDecoder(n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    sampler = RealSpeechSampler([REPO_ROOT / "app" / "static" / "cover.wav"])
    fec = InterleavedRS(n_data=args.rs_data, n_total=255, depth=8)

    eff_bps = n_bits * 1000 / SYMBOL_MS * args.rs_data / 255
    print(f"# Opus {args.kbps}k {args.application}  RS(255,{args.rs_data}) depth=8 -> {eff_bps:.0f} bps")
    tot_err = tot_bits = tot_fail = tot_blk = 0
    for seed in range(args.seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        data = bytes(np.random.default_rng(seed).integers(0, 256, args.data_bytes).tolist())
        coded, pad = fec.encode(data)
        tx = bytes_to_bits(coded)
        tx_full = np.concatenate([tx, np.zeros((-len(tx)) % n_bits, dtype=np.uint8)])
        symbols = torch.from_numpy(tx_full.reshape(-1, n_bits).astype(np.float32)).to(DEVICE)
        cover = sampler.sample(len(symbols))
        with torch.no_grad():
            modified, _ = enc(cover, symbols, perturbation_scale=pert_scale)
            rx = opus_round_trip_batch(modified, application=args.application, bitrate_kbps=args.kbps).to(DEVICE)
            preds = (torch.sigmoid(dec(rx)) > 0.5).cpu().numpy().astype(np.uint8).reshape(-1)[:len(tx)]
        raw_ber = float(np.mean(preds != tx))
        decoded, _, n_fail = fec.decode(bits_to_bytes(preds), pad)
        bx = np.array(bytes_to_bits(data)); by = np.array(bytes_to_bits(decoded[:args.data_bytes]))[:len(bx)]
        n_err = int((bx != by).sum()); n_blk = len(coded) // 255
        tot_err += n_err; tot_bits += len(bx); tot_fail += n_fail; tot_blk += n_blk
        print(f"   seed {seed}: raw {raw_ber*100:.2f}%  payload errors {n_err}/{len(bx)}  failed RS blocks {n_fail}/{n_blk}")
    print(f"TOTAL payload errors ({tot_err}/{tot_bits} bits)  failed RS blocks {tot_fail}/{tot_blk}")


if __name__ == "__main__":
    main()
