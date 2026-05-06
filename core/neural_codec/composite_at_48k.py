"""Composite test (SEQ speech + HighBand) under 48 kbps Opus AUDIO mode."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import numpy as np
import scipy.signal as sps
import torch

from neural_codec import DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS
from highband_codec import HighBandEncoder, HighBandDecoder, hard_bandpass_torch, HIGH_BAND_LO, HIGH_BAND_HI
from opus_mode_compare import opus_round_trip_batch
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes

SYMBOL_N = SR * SYMBOL_MS // 1000


def lowpass(x, hz, order=8):
    sos = sps.butter(order, hz/(SR/2), btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_ckpt", default=str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt"))
    ap.add_argument("--hb_ckpt", default=str(REPO_ROOT / "core/neural_codec/composite_attempt/ckpt_highband.pt"))
    ap.add_argument("--n_blocks", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    seq_ckpt = torch.load(args.seq_ckpt, map_location=DEVICE)
    n_bits_seq = seq_ckpt["n_bits_per_chunk"]; n_bits_seq_total = n_bits_seq * N_CHUNKS
    seq_enc = SeqEncoder(n_bits_seq).to(DEVICE); seq_dec = SeqDecoder(n_bits_seq).to(DEVICE)
    seq_enc.load_state_dict(seq_ckpt["encoder"]); seq_dec.load_state_dict(seq_ckpt["decoder"])
    seq_enc.eval(); seq_dec.eval()

    hb_ckpt = torch.load(args.hb_ckpt, map_location=DEVICE)
    n_bits_hb = hb_ckpt["n_bits"]
    hb_enc = HighBandEncoder(n_bits=n_bits_hb).to(DEVICE); hb_dec = HighBandDecoder(n_bits=n_bits_hb).to(DEVICE)
    hb_enc.load_state_dict(hb_ckpt["encoder"]); hb_dec.load_state_dict(hb_ckpt["decoder"])
    hb_enc.eval(); hb_dec.eval()

    raw_seq = n_bits_seq_total / (BLOCK_N/SR)
    raw_hb = n_bits_hb * 1000 / SYMBOL_MS
    print(f"# SEQ speech: {raw_seq:.0f} bps   HB tone: {raw_hb:.0f} bps   combined: {raw_seq+raw_hb:.0f} bps raw")
    print()
    print("# Testing under three Opus configs:")
    print(f"#  1. 24 kbps VOIP (the original threat model)")
    print(f"#  2. 24 kbps AUDIO (mode swap only)")
    print(f"#  3. 48 kbps AUDIO (real-world WebRTC/Discord-like)")
    print()
    for label, application, bitrate in [
        ("24k VOIP", "voip", 24),
        ("24k AUDIO", "audio", 24),
        ("48k AUDIO", "audio", 48),
    ]:
        print(f"# === {label} ===")
        print(f"  {'seed':>4}  {'seq_BER':>8}  {'hb_BER':>8}  {'seq_solo':>9}  {'hb_solo':>9}")
        seq_bers, hb_bers, seq_solo_bers, hb_solo_bers = [], [], [], []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            bits_seq_b = torch.randint(0, 2, (args.n_blocks, n_bits_seq_total), device=DEVICE).float()
            bits_hb_b = torch.randint(0, 2, (args.n_blocks * N_CHUNKS, n_bits_hb), device=DEVICE).float()

            with torch.no_grad():
                a_seq = seq_enc(bits_seq_b)
                a_hb_chunks = hb_enc(bits_hb_b)
                a_hb = a_hb_chunks.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(args.n_blocks, BLOCK_N)

            a_seq_np = a_seq.cpu().numpy()
            a_hb_np = a_hb.cpu().numpy()
            a_seq_lp = np.stack([lowpass(x, 2500.0) for x in a_seq_np])
            rms_speech = np.sqrt(np.mean(a_seq_lp**2)) + 1e-9
            rms_hb = np.sqrt(np.mean(a_hb_np**2)) + 1e-9
            a_hb_scaled = a_hb_np * (rms_speech / rms_hb)
            composite = np.clip(a_seq_lp + a_hb_scaled, -1, 1)

            composite_t = torch.from_numpy(composite.astype(np.float32)).to(DEVICE)
            composite_rt = opus_round_trip_batch(composite_t, application=application, bitrate_kbps=bitrate)
            composite_rt_np = composite_rt.cpu().numpy()

            rx_speech = np.stack([lowpass(x, 2800.0) for x in composite_rt_np])
            with torch.no_grad():
                seq_logits = seq_dec(torch.from_numpy(rx_speech.astype(np.float32)).to(DEVICE))
                seq_pred = (torch.sigmoid(seq_logits) > 0.5).float()
                seq_ber = (seq_pred != bits_seq_b).float().mean().item()

                rx_chunks = composite_rt_np.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N)
                rx_chunks_t = torch.from_numpy(rx_chunks.astype(np.float32)).to(DEVICE)
                rx_chunks_band = hard_bandpass_torch(rx_chunks_t, HIGH_BAND_LO, HIGH_BAND_HI)
                rx_chunks_band = rx_chunks_band * (rms_hb / rms_speech)
                hb_logits = hb_dec(rx_chunks_band)
                hb_pred = (torch.sigmoid(hb_logits) > 0.5).float()
                hb_ber = (hb_pred != bits_hb_b).float().mean().item()

                # Solo through Opus (no other-band interference)
                seq_solo_t = torch.from_numpy(a_seq_lp.astype(np.float32)).to(DEVICE)
                seq_solo_rt = opus_round_trip_batch(seq_solo_t, application=application, bitrate_kbps=bitrate)
                seq_solo_logits = seq_dec(seq_solo_rt)
                seq_solo_ber = ((torch.sigmoid(seq_solo_logits)>0.5).float() != bits_seq_b).float().mean().item()

                hb_solo_t = torch.from_numpy(a_hb_np.astype(np.float32)).to(DEVICE)
                hb_solo_rt = opus_round_trip_batch(hb_solo_t, application=application, bitrate_kbps=bitrate)
                hb_solo_chunks = hb_solo_rt.cpu().numpy().reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N)
                hb_solo_chunks_t = torch.from_numpy(hb_solo_chunks.astype(np.float32)).to(DEVICE)
                hb_solo_band = hard_bandpass_torch(hb_solo_chunks_t, HIGH_BAND_LO, HIGH_BAND_HI)
                hb_solo_logits = hb_dec(hb_solo_band)
                hb_solo_ber = ((torch.sigmoid(hb_solo_logits)>0.5).float() != bits_hb_b).float().mean().item()

            print(f"   {seed:>2}     {seq_ber*100:>6.2f}%   {hb_ber*100:>6.2f}%    {seq_solo_ber*100:>6.2f}%    {hb_solo_ber*100:>6.2f}%")
            seq_bers.append(seq_ber); hb_bers.append(hb_ber)
            seq_solo_bers.append(seq_solo_ber); hb_solo_bers.append(hb_solo_ber)

        avg_seq = np.mean(seq_bers); avg_hb = np.mean(hb_bers)
        print(f"   AVG   {avg_seq*100:>6.2f}%   {avg_hb*100:>6.2f}%    {np.mean(seq_solo_bers)*100:>6.2f}%    {np.mean(hb_solo_bers)*100:>6.2f}%")

        # FEC projection
        for n_data, n_total in [(191, 255), (159, 255), (127, 255), (95, 255), (63, 255)]:
            cap = (n_total - n_data) / 2 / n_total
            byte_seq = 1 - (1 - avg_seq) ** 8
            byte_hb = 1 - (1 - avg_hb) ** 8
            seq_ok = byte_seq < cap * 0.85
            hb_ok = byte_hb < cap * 0.85
            if seq_ok and hb_ok:
                eff_seq = raw_seq * n_data / n_total
                eff_hb = raw_hb * n_data / n_total
                print(f"     ✓ RS({n_total},{n_data}): combined reliable = {eff_seq + eff_hb:.0f} bps "
                      f"(speech {eff_seq:.0f} + HB {eff_hb:.0f})")
                break
        else:
            print(f"     ✗ no RS code clears both bands at this Opus config")
        print()


if __name__ == "__main__":
    main()
