"""
Final reliable-rate validation: neural codec + RS(255,191) interleaved FEC.

Pipeline: data bytes -> RS encode -> bits -> neural encoder -> audio
       -> real libopus 24k VoIP -> neural decoder -> bits -> RS decode -> data bytes

The neural channel gives ~0.1% raw BER at 1067 bps; RS(255,191) corrects 32 byte errors
per 255-byte codeword; depth-8 interleaving spreads burst errors across codewords.
Effective rate: 1067 * 191/255 ≈ 800 bps reliable.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import reedsolo

from neural_codec import Encoder, Decoder, real_opus_batch, DEVICE, SR, SYMBOL_MS


class InterleavedRS:
    def __init__(self, n_data=191, n_total=255, depth=8):
        assert (n_total - n_data) % 2 == 0
        self.n_data, self.n_total, self.depth = n_data, n_total, depth
        self.rs = reedsolo.RSCodec(n_total - n_data, nsize=n_total)

    def encode(self, data: bytes):
        block = self.n_data * self.depth
        pad = (-len(data)) % block
        if pad: data = data + np.random.bytes(pad)  # random pad — no quantile-style decoder here so shouldn't matter, but be safe
        out = bytearray()
        for i in range(0, len(data), block):
            rows = [self.rs.encode(data[i+d*self.n_data : i+(d+1)*self.n_data]) for d in range(self.depth)]
            for col in range(self.n_total):
                for d in range(self.depth):
                    out.append(rows[d][col])
        return bytes(out), pad

    def decode(self, coded: bytes, pad: int):
        block = self.n_total * self.depth
        out = bytearray()
        n_corr = 0; n_fail = 0
        for i in range(0, len(coded) - block + 1, block):
            seg = coded[i:i+block]
            rows = [bytearray() for _ in range(self.depth)]
            idx = 0
            for col in range(self.n_total):
                for d in range(self.depth):
                    rows[d].append(seg[idx]); idx += 1
            for r in rows:
                try:
                    dec = self.rs.decode(bytes(r))[0]
                    out += dec; n_corr += 1
                except reedsolo.ReedSolomonError:
                    out += bytes(r[:self.n_data]); n_fail += 1
        if pad and len(out) >= pad: out = out[:-pad]
        return bytes(out), n_corr, n_fail


def bytes_to_bits(b): return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def measure_with_fec(ckpt_path: str, n_data_bytes: int, n_seeds: int = 5,
                     fec_n_total: int = 255, fec_n_data: int = 191, fec_depth: int = 8):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits_per_sym = ckpt["n_bits"]
    enc = Encoder(n_bits=n_bits_per_sym).to(DEVICE)
    dec = Decoder(n_bits=n_bits_per_sym).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    fec = InterleavedRS(n_data=fec_n_data, n_total=fec_n_total, depth=fec_depth)

    raw_bps = n_bits_per_sym * 1000 / SYMBOL_MS
    eff_bps = raw_bps * fec_n_data / fec_n_total
    print(f"# checkpoint: {ckpt_path}")
    print(f"# n_bits/sym={n_bits_per_sym}  raw_bitrate={raw_bps:.0f} bps")
    print(f"# FEC: RS({fec_n_total},{fec_n_data}) depth={fec_depth} -> "
          f"eff_bitrate={eff_bps:.0f} bps")
    print(f"{'seed':>4}  {'data_B':>7}  {'raw_BER':>8}  {'fail_blks':>9}  {'post_BER':>9}")

    total_data_bits = 0; total_post_errs = 0
    rng = np.random.default_rng(42)
    for seed in range(n_seeds):
        # Fresh data per seed
        data = rng.integers(0, 256, n_data_bytes, dtype=np.uint8).tobytes()
        coded, pad = fec.encode(data)
        # Bits per symbol packing
        tx_bits = bytes_to_bits(coded)  # length = len(coded) * 8
        pad_to_sym = (-len(tx_bits)) % n_bits_per_sym
        tx_bits_full = np.concatenate([tx_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
        n_sym = len(tx_bits_full) // n_bits_per_sym
        symbols = tx_bits_full.reshape(n_sym, n_bits_per_sym)

        with torch.no_grad():
            bits_t = torch.from_numpy(symbols.astype(np.float32)).to(DEVICE)
            audio = enc(bits_t)
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
        rx_bits = preds.reshape(-1)[:len(tx_bits)]
        raw_ber = float(np.mean(rx_bits != tx_bits))
        rx_bytes = bits_to_bytes(rx_bits)
        decoded, n_corr, n_fail = fec.decode(rx_bytes, pad)
        decoded = decoded[:n_data_bytes]
        # Post-FEC bit-level error count on the data payload
        bx = bytes_to_bits(data); by = bytes_to_bits(decoded)[:len(bx)]
        post_errs = int(np.sum(bx != by))
        post_ber = post_errs / len(bx)
        print(f"  {seed:>2}    {n_data_bytes:>5}    {raw_ber*100:>6.3f}%  {n_fail:>7}     "
              f"{post_ber*100:>7.4f}% ({post_errs} bit errs)")
        total_data_bits += len(bx); total_post_errs += post_errs

    overall_post_ber = total_post_errs / total_data_bits if total_data_bits else 1.0
    print(f"{'TOTAL':>4}            {'':>8}            "
          f"{overall_post_ber*100:>7.4f}% ({total_post_errs}/{total_data_bits} bits)")
    print(f"# headline: {eff_bps:.0f} bps reliable, {overall_post_ber*100:.4f}% post-FEC BER")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data_bytes", type=int, default=1900)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    measure_with_fec(args.ckpt, n_data_bytes=args.data_bytes, n_seeds=args.seeds)
