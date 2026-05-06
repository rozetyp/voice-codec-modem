"""
TRIZ band-split composite carrier:
  - 0-3 kHz: speech-like content from the SEQ adversarial codec
  - 3.5-7 kHz: high-rate modem-tone content from the IID baseline n=128
  - Combined audio sounds speech-dominant; bit channel runs in both bands.

Naive test first: apply existing pre-trained models, band-filter their outputs,
sum, run through real Opus, decode each band with the respective model.
If both bands survive, we have ~3 kbps reliable + audibly speech-like.
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
SPEECH_LP_HZ = 3000.0
TONE_HP_HZ = 3500.0


def lowpass(x: np.ndarray, hz: float, sr: int = SR, order: int = 8):
    sos = sps.butter(order, hz / (sr/2), btype="lowpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def highpass(x: np.ndarray, hz: float, sr: int = SR, order: int = 8):
    sos = sps.butter(order, hz / (sr/2), btype="highpass", output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def show_spectrum(audio: np.ndarray, label: str):
    """Print energy in 4 bands so we can sanity-check."""
    f, P = sps.welch(audio, fs=SR, nperseg=512)
    bands = [(0, 1000), (1000, 3000), (3000, 5000), (5000, 7500)]
    e = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        e.append(P[m].sum())
    total = sum(e) + 1e-9
    pct = [x/total*100 for x in e]
    print(f"  {label:<32} band%  0-1k:{pct[0]:5.1f}  1-3k:{pct[1]:5.1f}  3-5k:{pct[2]:5.1f}  5-7.5k:{pct[3]:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_ckpt", default=str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt"))
    ap.add_argument("--iid_ckpt", default=str(REPO_ROOT / "core/neural_codec/ckpt_n128_mixed.pt"))
    ap.add_argument("--n_blocks", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "composite_attempt" / "composite_out"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # Load SEQ adv (speech-band)
    seq_ckpt = torch.load(args.seq_ckpt, map_location=DEVICE)
    n_bits_seq = seq_ckpt["n_bits_per_chunk"]; n_bits_seq_total = n_bits_seq * N_CHUNKS
    seq_enc = SeqEncoder(n_bits_seq).to(DEVICE); seq_dec = SeqDecoder(n_bits_seq).to(DEVICE)
    seq_enc.load_state_dict(seq_ckpt["encoder"]); seq_dec.load_state_dict(seq_ckpt["decoder"])
    seq_enc.eval(); seq_dec.eval()

    # Load IID baseline n=128 (will be band-limited to 3.5-7 kHz)
    iid_ckpt = torch.load(args.iid_ckpt, map_location=DEVICE)
    n_bits_iid = iid_ckpt["n_bits"]
    iid_enc = IIDEncoder(n_bits=n_bits_iid).to(DEVICE); iid_dec = IIDDecoder(n_bits=n_bits_iid).to(DEVICE)
    iid_enc.load_state_dict(iid_ckpt["encoder"]); iid_dec.load_state_dict(iid_ckpt["decoder"])
    iid_enc.eval(); iid_dec.eval()

    # Each SEQ block (120ms) aligns with 4 IID symbols (4*30ms = 120ms)
    print(f"# SEQ:  n_bits/chunk={n_bits_seq}  bits/block={n_bits_seq_total}  -> {n_bits_seq_total/(BLOCK_N/SR):.0f} bps raw")
    print(f"# IID:  n_bits/symbol={n_bits_iid} -> {n_bits_iid*1000//SYMBOL_MS} bps raw")
    print(f"# combined potential: {n_bits_seq_total/(BLOCK_N/SR) + n_bits_iid*1000/SYMBOL_MS:.0f} bps raw")
    print()

    print(f"# === Spectra check (one batch, no Opus) ===")
    with torch.no_grad():
        bits_seq = torch.randint(0, 2, (4, n_bits_seq_total), device=DEVICE).float()
        bits_iid = torch.randint(0, 2, (16, n_bits_iid), device=DEVICE).float()
        a_seq = seq_enc(bits_seq).cpu().numpy().reshape(-1)
        a_iid = iid_enc(bits_iid).cpu().numpy().reshape(-1)
    show_spectrum(a_seq, "SEQ encoder raw output")
    show_spectrum(a_iid, "IID encoder raw output")
    a_seq_lp = lowpass(a_seq, SPEECH_LP_HZ)
    a_iid_hp = highpass(a_iid, TONE_HP_HZ)
    show_spectrum(a_seq_lp, "SEQ output low-passed @3kHz")
    show_spectrum(a_iid_hp, "IID output high-passed @3.5kHz")

    print(f"\n# === Naive composite test, {args.seeds} seeds x {args.n_blocks} blocks ===")
    print(f"{'seed':>4}  {'seq_BER':>9}  {'iid_BER':>9}  {'speech_only':>11}  {'tone_only':>11}")
    seq_bers, iid_bers = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        # Generate aligned bit streams
        bits_seq_b = torch.randint(0, 2, (args.n_blocks, n_bits_seq_total), device=DEVICE).float()
        # 4 IID symbols per SEQ block
        bits_iid_b = torch.randint(0, 2, (args.n_blocks * N_CHUNKS, n_bits_iid), device=DEVICE).float()

        with torch.no_grad():
            a_seq = seq_enc(bits_seq_b)  # (n_blocks, BLOCK_N=1920)
            a_iid_chunks = iid_enc(bits_iid_b)  # (n_blocks*N_CHUNKS, SYMBOL_N=480)
            a_iid = a_iid_chunks.reshape(args.n_blocks, N_CHUNKS, SYMBOL_N).reshape(args.n_blocks, BLOCK_N)

        a_seq_np = a_seq.cpu().numpy()
        a_iid_np = a_iid.cpu().numpy()
        # Per-block band filter
        a_seq_lp = np.stack([lowpass(x, SPEECH_LP_HZ) for x in a_seq_np])
        a_iid_hp = np.stack([highpass(x, TONE_HP_HZ) for x in a_iid_np])
        # Equal RMS so neither dominates
        rms_speech = np.sqrt(np.mean(a_seq_lp**2)) + 1e-9
        rms_tone = np.sqrt(np.mean(a_iid_hp**2)) + 1e-9
        a_iid_hp = a_iid_hp * (rms_speech / rms_tone)  # match speech RMS so total perceives speech-dominant
        composite = a_seq_lp + a_iid_hp
        # Through real Opus (block by block)
        composite_t = torch.from_numpy(composite).to(DEVICE)
        composite_rt = real_opus_batch(composite_t)
        composite_rt_np = composite_rt.cpu().numpy()

        # Decode: split bands, run each model
        rx_speech_band = np.stack([lowpass(x, SPEECH_LP_HZ) for x in composite_rt_np])
        rx_tone_band = np.stack([highpass(x, TONE_HP_HZ) for x in composite_rt_np])

        with torch.no_grad():
            rx_speech_t = torch.from_numpy(rx_speech_band.astype(np.float32)).to(DEVICE)
            seq_logits = seq_dec(rx_speech_t)
            seq_pred = (torch.sigmoid(seq_logits) > 0.5).float()
            seq_ber = (seq_pred != bits_seq_b).float().mean().item()

            # Reshape tone band into IID symbols
            rx_tone_t = torch.from_numpy(rx_tone_band.reshape(-1, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N).astype(np.float32)).to(DEVICE)
            iid_logits = iid_dec(rx_tone_t)
            iid_pred = (torch.sigmoid(iid_logits) > 0.5).float()
            iid_ber = (iid_pred != bits_iid_b).float().mean().item()

        # Also test each in isolation through Opus (no other-band interference)
        with torch.no_grad():
            seq_solo_rt = real_opus_batch(torch.from_numpy(a_seq_lp.astype(np.float32)).to(DEVICE))
            seq_solo_logits = seq_dec(seq_solo_rt)
            seq_solo_ber = ((torch.sigmoid(seq_solo_logits)>0.5).float() != bits_seq_b).float().mean().item()

            iid_solo_rt = real_opus_batch(torch.from_numpy(a_iid_hp.astype(np.float32)).to(DEVICE))
            iid_solo_band = highpass(iid_solo_rt.cpu().numpy(), TONE_HP_HZ)
            iid_solo_t = torch.from_numpy(iid_solo_band.reshape(-1, N_CHUNKS, SYMBOL_N).reshape(-1, SYMBOL_N).astype(np.float32)).to(DEVICE)
            iid_solo_logits = iid_dec(iid_solo_t)
            iid_solo_ber = ((torch.sigmoid(iid_solo_logits)>0.5).float() != bits_iid_b).float().mean().item()

        print(f"  {seed:>2}    {seq_ber*100:>7.2f}%  {iid_ber*100:>7.2f}%      {seq_solo_ber*100:>7.2f}%      {iid_solo_ber*100:>7.2f}%")
        seq_bers.append(seq_ber); iid_bers.append(iid_ber)

        # Save 1 sec of composite audio for the first seed
        if seed == 0:
            sf.write(out_dir / "composite_carrier.wav",
                     np.clip(composite.reshape(-1)[:SR], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "composite_after_opus.wav",
                     np.clip(composite_rt_np.reshape(-1)[:SR], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "speech_band_only.wav",
                     np.clip(a_seq_lp.reshape(-1)[:SR], -1, 1).astype(np.float32), SR)
            sf.write(out_dir / "tone_band_only.wav",
                     np.clip(a_iid_hp.reshape(-1)[:SR], -1, 1).astype(np.float32), SR)

    print(f"\n# AVG seq_BER={np.mean(seq_bers)*100:.2f}%  iid_BER={np.mean(iid_bers)*100:.2f}%")
    raw_seq = n_bits_seq_total / (BLOCK_N/SR)
    raw_iid = n_bits_iid * 1000 / SYMBOL_MS
    print(f"# raw speech-band rate: {raw_seq:.0f} bps  (BER {np.mean(seq_bers)*100:.2f}%)")
    print(f"# raw tone-band   rate: {raw_iid:.0f} bps  (BER {np.mean(iid_bers)*100:.2f}%)")
    print(f"# combined raw       : {raw_seq + raw_iid:.0f} bps")
    print(f"# samples in {out_dir}/")


if __name__ == "__main__":
    main()
