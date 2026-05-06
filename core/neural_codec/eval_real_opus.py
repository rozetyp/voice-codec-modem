"""
Heavy real-libopus evaluation of a trained neural codec checkpoint.
- Loads (encoder, decoder) state dicts
- Runs N batches through ffmpeg libopus 24k VoIP
- Reports BER + bits-transmitted, multi-seed
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from neural_codec import Encoder, Decoder, real_opus_batch, DEVICE, SR, SYMBOL_MS

def evaluate(ckpt_path: str, n_seeds: int = 5, batch_size: int = 256, bitrate_kbps: int = 24):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = Encoder(n_bits=n_bits).to(DEVICE)
    dec = Decoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    raw_bps = n_bits * 1000 / SYMBOL_MS
    print(f"# checkpoint: {ckpt_path}")
    print(f"# n_bits/symbol={n_bits}  raw_bitrate={raw_bps:.0f} bps  symbol_ms={SYMBOL_MS}")
    print(f"# eval: {n_seeds} seeds x {batch_size} symbols each (Opus {bitrate_kbps}k VoIP)")
    print(f"{'seed':>4}  {'bits_tx':>8}  {'bit_errs':>8}  {'BER':>8}")
    total_tx = 0; total_err = 0
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        with torch.no_grad():
            bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
            audio = enc(bits)
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float()
            n_err = int((preds != bits).sum().item())
            n_tx = batch_size * n_bits
            ber = n_err / n_tx
        print(f"  {seed:>2}    {n_tx:>6}    {n_err:>6}    {ber*100:>6.3f}%")
        total_tx += n_tx; total_err += n_err
    print(f"{'TOTAL':>4}    {total_tx:>6}    {total_err:>6}    {total_err/total_tx*100:>6.3f}%")
    print(f"# overall: {raw_bps:.0f} bps raw at {total_err/total_tx*100:.3f}% BER")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bitrate_kbps", type=int, default=24)
    args = ap.parse_args()
    evaluate(args.ckpt, n_seeds=args.seeds, batch_size=args.batch, bitrate_kbps=args.bitrate_kbps)
