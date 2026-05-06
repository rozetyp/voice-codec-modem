"""End-to-end FEC validation for the sequence-level codec.

Same RS-interleaved scheme as neural_with_fec, adapted for the seq codec which
processes 1920-sample blocks of n_bits_per_chunk*4 bits each.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

import numpy as np
import torch

from neural_codec import real_opus_batch, DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def measure(ckpt_path: str, n_data_bytes: int, n_seeds: int = 5,
            fec_n_total: int = 255, fec_n_data: int = 191, fec_depth: int = 8,
            batch_size: int = 128):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits_per_chunk = ckpt["n_bits_per_chunk"]
    n_bits_total = n_bits_per_chunk * N_CHUNKS
    enc = SeqEncoder(n_bits_per_chunk).to(DEVICE)
    dec = SeqDecoder(n_bits_per_chunk).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    fec = InterleavedRS(n_data=fec_n_data, n_total=fec_n_total, depth=fec_depth)

    raw_bps = n_bits_total / (BLOCK_N / SR)
    eff_bps = raw_bps * fec_n_data / fec_n_total
    print(f"# checkpoint: {ckpt_path}")
    print(f"# seq codec: {n_bits_per_chunk} bits/chunk x {N_CHUNKS} = {n_bits_total} bits/block")
    print(f"# raw_bitrate={raw_bps:.0f} bps  eff_bitrate={eff_bps:.0f} bps  "
          f"FEC=RS({fec_n_total},{fec_n_data}) depth={fec_depth}")
    print(f"{'seed':>4}  {'raw_BER':>8}  {'fail_blks':>9}  {'post_BER':>9}")

    raws = []; posts = []; fails = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        data = bytes(rng.integers(0, 256, n_data_bytes).tolist())
        coded, pad = fec.encode(data)
        tx_bits = bytes_to_bits(coded)
        pad_to_block = (-len(tx_bits)) % n_bits_total
        tx_bits_full = np.concatenate([tx_bits, np.zeros(pad_to_block, dtype=np.uint8)])
        n_blocks = len(tx_bits_full) // n_bits_total
        blocks = tx_bits_full.reshape(n_blocks, n_bits_total)

        rx_bits_chunks = []
        with torch.no_grad():
            for i in range(0, n_blocks, batch_size):
                sb = torch.from_numpy(blocks[i:i+batch_size].astype(np.float32)).to(DEVICE)
                audio = enc(sb)
                audio_codec = real_opus_batch(audio)
                logits = dec(audio_codec)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
                rx_bits_chunks.append(preds.reshape(-1))
        rx_bits = np.concatenate(rx_bits_chunks)[:len(tx_bits)]
        raw_ber = float(np.mean(rx_bits != tx_bits))
        rx_bytes = bits_to_bytes(rx_bits)
        decoded, n_corr, n_fail = fec.decode(rx_bytes, pad)
        decoded = decoded[:n_data_bytes]
        bx = bytes_to_bits(data); by = bytes_to_bits(decoded)[:len(bx)]
        post_ber = float(np.mean(np.array(bx) != np.array(by)))
        print(f"  {seed:>2}    {raw_ber*100:>6.3f}%  {n_fail:>7}     {post_ber*100:>7.4f}%")
        raws.append(raw_ber); posts.append(post_ber); fails.append(n_fail)
    print(f"  AVG   {np.mean(raws)*100:>6.3f}%  {np.mean(fails):>7.1f}     {np.mean(posts)*100:>7.4f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data_bytes", type=int, default=1900)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fec_n_data", type=int, default=191)
    ap.add_argument("--fec_n_total", type=int, default=255)
    ap.add_argument("--fec_depth", type=int, default=8)
    args = ap.parse_args()
    measure(args.ckpt, n_data_bytes=args.data_bytes, n_seeds=args.seeds,
            fec_n_data=args.fec_n_data, fec_n_total=args.fec_n_total, fec_depth=args.fec_depth)
