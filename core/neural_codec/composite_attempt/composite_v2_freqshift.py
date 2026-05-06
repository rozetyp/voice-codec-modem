"""
v2: instead of highpass-filtering the IID output (which strips its information),
frequency-translate it up to the high band via mixing with a 4 kHz carrier.
The IID's natural 0-3 kHz energy lands at 1-7 kHz; Opus preserves that band well.
"""
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

from neural_codec import Encoder as IIDEncoder, Decoder as IIDDecoder, real_opus_batch, DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS

SYMBOL_N = SR * SYMBOL_MS // 1000
F_CARRIER = 3500.0  # Hz, modulation carrier — push into 3.5-6.5 kHz where Opus is more generous
SPEECH_LP_HZ = 2500.0  # speech band cutoff, leave a 500 Hz guard before the IID upper sideband


def lowpass(x, hz, order=8):
    sos = sps.butter(order, hz/(SR/2), btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def bandpass(x, lo_hz, hi_hz, order=8):
    sos = sps.butter(order, [lo_hz/(SR/2), hi_hz/(SR/2)], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


UPPER_BP = (3000.0, 6500.0)  # bandpass for the upper sideband

def freq_shift_up(x_np: np.ndarray, f_c: float = F_CARRIER) -> np.ndarray:
    """SSB up-mix: multiply by cos(2π f_c t), then bandpass to keep upper sideband only."""
    n = x_np.shape[-1]
    t = np.arange(n) / SR
    carrier = np.cos(2*np.pi*f_c*t).astype(np.float32)
    mixed = (x_np * carrier).astype(np.float32) * 2.0
    return bandpass(mixed, *UPPER_BP)


def freq_shift_down(x_np: np.ndarray, f_c: float = F_CARRIER) -> np.ndarray:
    """Demodulate: bandpass to upper sideband, multiply by carrier, lowpass to baseband."""
    n = x_np.shape[-1]
    bp = bandpass(x_np, *UPPER_BP)
    t = np.arange(n) / SR
    carrier = np.cos(2*np.pi*f_c*t).astype(np.float32)
    mixed = bp * carrier * 2.0
    return lowpass(mixed, 3000.0)


def show_spectrum(audio, label):
    f, P = sps.welch(audio, fs=SR, nperseg=512)
    bands = [(0, 1000), (1000, 3000), (3000, 5000), (5000, 7500)]
    e = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi); e.append(P[m].sum())
    total = sum(e) + 1e-9; pct = [x/total*100 for x in e]
    print(f"  {label:<36} band%  0-1k:{pct[0]:5.1f}  1-3k:{pct[1]:5.1f}  3-5k:{pct[2]:5.1f}  5-7.5k:{pct[3]:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_ckpt", default=str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt"))
    ap.add_argument("--iid_ckpt", default=str(REPO_ROOT / "core/neural_codec/ckpt_n128_mixed.pt"))
    ap.add_argument("--n_blocks", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "composite_attempt" / "composite_v2_out"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    seq_ckpt = torch.load(args.seq_ckpt, map_location=DEVICE)
    n_bits_seq = seq_ckpt["n_bits_per_chunk"]; n_bits_seq_total = n_bits_seq * N_CHUNKS
    seq_enc = SeqEncoder(n_bits_seq).to(DEVICE); seq_dec = SeqDecoder(n_bits_seq).to(DEVICE)
    seq_enc.load_state_dict(seq_ckpt["encoder"]); seq_dec.load_state_dict(seq_ckpt["decoder"])
    seq_enc.eval(); seq_dec.eval()

    iid_ckpt = torch.load(args.iid_ckpt, map_location=DEVICE)
    n_bits_iid = iid_ckpt["n_bits"]
    iid_enc = IIDEncoder(n_bits=n_bits_iid).to(DEVICE); iid_dec = IIDDecoder(n_bits=n_bits_iid).to(DEVICE)
    iid_enc.load_state_dict(iid_ckpt["encoder"]); iid_dec.load_state_dict(iid_ckpt["decoder"])
    iid_enc.eval(); iid_dec.eval()

    raw_seq = n_bits_seq_total / (BLOCK_N/SR)
    raw_iid = n_bits_iid * 1000 / SYMBOL_MS
    print(f"# SEQ: {raw_seq:.0f} bps raw   IID: {raw_iid:.0f} bps raw   combined: {raw_seq+raw_iid:.0f} bps raw")
    print()

    # Spectra check
    print("# === Spectra ===")
    with torch.no_grad():
        bits_seq = torch.randint(0, 2, (4, n_bits_seq_total), device=DEVICE).float()
        bits_iid = torch.randint(0, 2, (16, n_bits_iid), device=DEVICE).float()
        a_seq = seq_enc(bits_seq).cpu().numpy().reshape(-1)
        a_iid = iid_enc(bits_iid).cpu().numpy().reshape(-1)
    a_seq_lp = lowpass(a_seq, SPEECH_LP_HZ)
    a_iid_shifted = freq_shift_up(a_iid)
    show_spectrum(a_seq, "SEQ raw")
    show_spectrum(a_iid, "IID raw")
    show_spectrum(a_seq_lp, "SEQ lowpassed @2.8kHz")
    show_spectrum(a_iid_shifted, "IID frequency-shifted up by 4kHz")

    # Sanity: can we demodulate without Opus?
    a_iid_recovered = freq_shift_down(a_iid_shifted)
    show_spectrum(a_iid_recovered, "IID after up-then-down (no Opus)")

    print(f"\n# === Composite test (freq-shifted IID), {args.seeds} seeds x {args.n_blocks} blocks ===")
    print(f"{'seed':>4}  {'seq_BER':>9}  {'iid_BER':>9}  {'iid_solo':>9}")
    seq_bers, iid_bers = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        bits_seq_b = torch.randint(0, 2, (args.n_blocks, n_bits_seq_total), device=DEVICE).float()
        bits_iid_b = torch.randint(0, 2, (args.n_blocks * N_CHUNKS, n_bits_iid), device=DEVICE).float()

        with torch.no_grad():
            a_seq = seq_enc(bits_seq_b)
            a_iid_chunks = iid_enc(bits_iid_b)
            a_iid = a_iid_chunks.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(args.n_blocks, BLOCK_N)

        a_seq_np = a_seq.cpu().numpy()
        a_iid_np = a_iid.cpu().numpy()
        a_seq_lp = np.stack([lowpass(x, SPEECH_LP_HZ) for x in a_seq_np])
        a_iid_shifted = np.stack([freq_shift_up(x) for x in a_iid_np])
        rms_speech = np.sqrt(np.mean(a_seq_lp**2)) + 1e-9
        rms_tone = np.sqrt(np.mean(a_iid_shifted**2)) + 1e-9
        # Match speech RMS so total is speech-dominant in loudness
        a_iid_shifted = a_iid_shifted * (rms_speech / rms_tone)
        composite = a_seq_lp + a_iid_shifted
        composite = np.clip(composite, -1, 1)

        composite_t = torch.from_numpy(composite.astype(np.float32)).to(DEVICE)
        composite_rt = real_opus_batch(composite_t)
        composite_rt_np = composite_rt.cpu().numpy()

        rx_speech_band = np.stack([lowpass(x, SPEECH_LP_HZ) for x in composite_rt_np])
        rx_tone_band = np.stack([freq_shift_down(x) for x in composite_rt_np])
        # Tone band needs to be re-scaled (RMS undo)
        rx_tone_band = rx_tone_band * (rms_tone / rms_speech)

        with torch.no_grad():
            seq_logits = seq_dec(torch.from_numpy(rx_speech_band.astype(np.float32)).to(DEVICE))
            seq_pred = (torch.sigmoid(seq_logits) > 0.5).float()
            seq_ber = (seq_pred != bits_seq_b).float().mean().item()

            rx_tone_t = torch.from_numpy(rx_tone_band.reshape(-1, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N).astype(np.float32)).to(DEVICE)
            iid_logits = iid_dec(rx_tone_t)
            iid_pred = (torch.sigmoid(iid_logits) > 0.5).float()
            iid_ber = (iid_pred != bits_iid_b).float().mean().item()

            # IID solo through Opus (after freq-shift)
            iid_solo_t = torch.from_numpy(a_iid_shifted.astype(np.float32)).to(DEVICE)
            iid_solo_rt = real_opus_batch(iid_solo_t)
            rx_solo = np.stack([freq_shift_down(x) for x in iid_solo_rt.cpu().numpy()])
            rx_solo = rx_solo * (rms_tone / rms_speech)  # match training scale
            rx_solo_t = torch.from_numpy(rx_solo.reshape(-1, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N).astype(np.float32)).to(DEVICE)
            iid_solo_logits = iid_dec(rx_solo_t)
            iid_solo_ber = ((torch.sigmoid(iid_solo_logits)>0.5).float() != bits_iid_b).float().mean().item()

        print(f"  {seed:>2}    {seq_ber*100:>7.2f}%  {iid_ber*100:>7.2f}%   {iid_solo_ber*100:>7.2f}%")
        seq_bers.append(seq_ber); iid_bers.append(iid_ber)

        if seed == 0:
            sf.write(out_dir / "composite_carrier.wav",
                     np.clip(composite.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "composite_after_opus.wav",
                     np.clip(composite_rt_np.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "speech_band_only.wav",
                     np.clip(a_seq_lp.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "iid_freqshifted.wav",
                     np.clip(a_iid_shifted.reshape(-1)[:SR*2], -1, 1).astype(np.float32), SR)

    print(f"\n# AVG seq_BER={np.mean(seq_bers)*100:.2f}%  iid_BER={np.mean(iid_bers)*100:.2f}%")
    print(f"# samples in {out_dir}/")


if __name__ == "__main__":
    main()
