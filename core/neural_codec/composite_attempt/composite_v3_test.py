"""Composite test: SEQ speech band + trained high-band specialist."""
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
import scipy.signal as sps
import soundfile as sf
import torch

from neural_codec import real_opus_batch, DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS
from highband_codec import HighBandEncoder, HighBandDecoder, hard_bandpass_torch, HIGH_BAND_LO, HIGH_BAND_HI
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes

SYMBOL_N = SR * SYMBOL_MS // 1000


def lowpass(x, hz, order=8):
    sos = sps.butter(order, hz/(SR/2), btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def show_band_pct(audio, label):
    f, P = sps.welch(audio, fs=SR, nperseg=512)
    bands = [(0, 1000), (1000, 3000), (3000, 5000), (5000, 7500)]
    e = [P[(f>=lo)&(f<hi)].sum() for lo, hi in bands]
    total = sum(e) + 1e-9; pct = [x/total*100 for x in e]
    print(f"  {label:<35} 0-1k:{pct[0]:5.1f}  1-3k:{pct[1]:5.1f}  3-5k:{pct[2]:5.1f}  5-7.5k:{pct[3]:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_ckpt", default=str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt"))
    ap.add_argument("--hb_ckpt", default=str(REPO_ROOT / "core/neural_codec/composite_attempt/ckpt_highband.pt"))
    ap.add_argument("--n_blocks", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "composite_attempt" / "composite_v3_out"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

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
    print(f"# SEQ speech-band: {raw_seq:.0f} bps   HB tone-band: {raw_hb:.0f} bps   combined: {raw_seq+raw_hb:.0f} bps raw")
    print()

    print(f"# === {args.seeds} seeds x {args.n_blocks} blocks ===")
    print(f"{'seed':>4}  {'seq_BER':>9}  {'hb_BER':>9}  {'seq_solo':>9}  {'hb_solo':>9}")
    seq_bers, hb_bers, seq_solo_bers, hb_solo_bers = [], [], [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        bits_seq_b = torch.randint(0, 2, (args.n_blocks, n_bits_seq_total), device=DEVICE).float()
        bits_hb_b = torch.randint(0, 2, (args.n_blocks * N_CHUNKS, n_bits_hb), device=DEVICE).float()

        with torch.no_grad():
            a_seq = seq_enc(bits_seq_b)  # (n_blocks, BLOCK_N)
            a_hb_chunks = hb_enc(bits_hb_b)  # already band-restricted; (n_blocks*N_CHUNKS, SYMBOL_N)
            a_hb = a_hb_chunks.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(args.n_blocks, BLOCK_N)

        a_seq_np = a_seq.cpu().numpy()
        a_hb_np = a_hb.cpu().numpy()
        # SEQ output: lowpass at 2.5kHz to leave a guard band
        a_seq_lp = np.stack([lowpass(x, 2500.0) for x in a_seq_np])
        # HB is already band-restricted
        # Match speech RMS so total perceives speech-dominant
        rms_speech = np.sqrt(np.mean(a_seq_lp**2)) + 1e-9
        rms_hb = np.sqrt(np.mean(a_hb_np**2)) + 1e-9
        a_hb_scaled = a_hb_np * (rms_speech / rms_hb)
        composite = a_seq_lp + a_hb_scaled
        composite = np.clip(composite, -1, 1)

        composite_t = torch.from_numpy(composite.astype(np.float32)).to(DEVICE)
        composite_rt = real_opus_batch(composite_t)
        composite_rt_np = composite_rt.cpu().numpy()

        # Decode each band
        rx_speech = np.stack([lowpass(x, 2800.0) for x in composite_rt_np])
        with torch.no_grad():
            seq_logits = seq_dec(torch.from_numpy(rx_speech.astype(np.float32)).to(DEVICE))
            seq_pred = (torch.sigmoid(seq_logits) > 0.5).float()
            seq_ber = (seq_pred != bits_seq_b).float().mean().item()

            # HB decoder: feed back the rx audio (it bandpasses internally? no, the encoder was band-restricted)
            # Need to apply the same hard bandpass as in HighBand training
            # Reshape into IID symbols first
            rx_chunks = composite_rt_np.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N)
            rx_chunks_t = torch.from_numpy(rx_chunks.astype(np.float32)).to(DEVICE)
            # Apply bandpass to keep only HB content (matches training pipeline)
            rx_chunks_band = hard_bandpass_torch(rx_chunks_t, HIGH_BAND_LO, HIGH_BAND_HI)
            # Undo the rms scaling so the HB decoder sees its training-scale input
            rx_chunks_band = rx_chunks_band * (rms_hb / rms_speech)
            hb_logits = hb_dec(rx_chunks_band)
            hb_pred = (torch.sigmoid(hb_logits) > 0.5).float()
            hb_ber = (hb_pred != bits_hb_b).float().mean().item()

            # Solo: each band alone through Opus
            seq_solo_t = torch.from_numpy(a_seq_lp.astype(np.float32)).to(DEVICE)
            seq_solo_rt = real_opus_batch(seq_solo_t)
            seq_solo_logits = seq_dec(seq_solo_rt)
            seq_solo_ber = ((torch.sigmoid(seq_solo_logits)>0.5).float() != bits_seq_b).float().mean().item()

            hb_solo_t = torch.from_numpy(a_hb_np.astype(np.float32)).to(DEVICE)
            hb_solo_rt = real_opus_batch(hb_solo_t)
            # Reshape into IID symbols for the decoder
            hb_solo_chunks = hb_solo_rt.cpu().numpy().reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N)
            hb_solo_chunks_t = torch.from_numpy(hb_solo_chunks.astype(np.float32)).to(DEVICE)
            hb_solo_band = hard_bandpass_torch(hb_solo_chunks_t, HIGH_BAND_LO, HIGH_BAND_HI)
            hb_solo_logits = hb_dec(hb_solo_band)
            hb_solo_ber = ((torch.sigmoid(hb_solo_logits)>0.5).float() != bits_hb_b).float().mean().item()

        print(f"  {seed:>2}    {seq_ber*100:>7.2f}%  {hb_ber*100:>7.2f}%   {seq_solo_ber*100:>7.2f}%   {hb_solo_ber*100:>7.2f}%")
        seq_bers.append(seq_ber); hb_bers.append(hb_ber)
        seq_solo_bers.append(seq_solo_ber); hb_solo_bers.append(hb_solo_ber)

        if seed == 0:
            sf.write(out_dir / "composite_carrier.wav",
                     np.clip(composite.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "composite_after_opus.wav",
                     np.clip(composite_rt_np.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "speech_band.wav",
                     np.clip(a_seq_lp.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "hb_band.wav",
                     np.clip(a_hb_scaled.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            show_band_pct(composite.reshape(-1), "composite carrier (1s)")
            show_band_pct(composite_rt_np.reshape(-1), "composite after Opus")

    print(f"\n# AVG composite: seq_BER={np.mean(seq_bers)*100:.2f}%  hb_BER={np.mean(hb_bers)*100:.2f}%")
    print(f"# AVG solo:      seq_BER={np.mean(seq_solo_bers)*100:.2f}%  hb_BER={np.mean(hb_solo_bers)*100:.2f}%")

    # Project to FEC reliable rates
    seq_ber_avg = np.mean(seq_bers)
    hb_ber_avg = np.mean(hb_bers)
    print()
    print("# FEC projections (composite):")
    for n_data, n_total in [(191, 255), (159, 255), (127, 255), (95, 255), (63, 255), (31, 255)]:
        eff_seq = raw_seq * n_data / n_total
        eff_hb = raw_hb * n_data / n_total
        # Approx byte BER from bit BER
        byte_ber_seq = 1 - (1 - seq_ber_avg) ** 8
        byte_ber_hb = 1 - (1 - hb_ber_avg) ** 8
        cap = (n_total - n_data) / 2 / n_total
        seq_ok = "✓" if byte_ber_seq < cap * 0.85 else "fails"
        hb_ok = "✓" if byte_ber_hb < cap * 0.85 else "fails"
        print(f"  RS({n_total},{n_data}): cap={cap*100:.1f}%  speech: byte_BER={byte_ber_seq*100:.1f}% {seq_ok} ({eff_seq:.0f} bps)   HB: byte_BER={byte_ber_hb*100:.1f}% {hb_ok} ({eff_hb:.0f} bps)")


if __name__ == "__main__":
    main()
