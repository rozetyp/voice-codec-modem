"""
Compare Opus VOIP vs AUDIO application modes for our existing checkpoints.

VOIP mode: SILK-dominant, heavy bit allocation to 200-3500 Hz speech band.
AUDIO mode: CELT-dominant, more uniform allocation across the wideband 0-7 kHz.

Hypothesis: AUDIO mode gives the high-band specialist an actual bit budget,
unlocking the rate-headline beyond what VOIP mode allows.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

import argparse
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch

from neural_codec import Encoder as IIDEncoder, Decoder as IIDDecoder, DEVICE, SR, SYMBOL_MS
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS
from highband_codec import HighBandEncoder, HighBandDecoder, hard_bandpass_torch, HIGH_BAND_LO, HIGH_BAND_HI

SYMBOL_N = SR * SYMBOL_MS // 1000

WORK = Path(tempfile.gettempdir()) / "opus_mode_compare"
WORK.mkdir(exist_ok=True)


def opus_round_trip(audio_np: np.ndarray, application: str = "voip", bitrate_kbps: int = 24) -> np.ndarray:
    """Round-trip a 1D audio array through libopus with the given application mode."""
    tag = str(np.random.randint(1, 1<<30))
    inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
    sf.write(inp, audio_np.astype(np.float32), SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                    "-c:a","libopus","-b:a",f"{bitrate_kbps}k",
                    "-application", application, str(opx)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                    "-ar",str(SR),"-ac","1",str(out)], check=True)
    a, _ = sf.read(out)
    inp.unlink(missing_ok=True); opx.unlink(missing_ok=True); out.unlink(missing_ok=True)
    if len(a) < len(audio_np):
        a = np.pad(a, (0, len(audio_np) - len(a)))
    elif len(a) > len(audio_np):
        a = a[:len(audio_np)]
    return a.astype(np.float32)


def opus_round_trip_batch(audio_t: torch.Tensor, application: str, bitrate_kbps: int) -> torch.Tensor:
    """Pack a batch into one ffmpeg call (much faster than per-row)."""
    audio_np = audio_t.detach().cpu().numpy()
    B, N = audio_np.shape
    flat = audio_np.reshape(-1)
    rt = opus_round_trip(flat, application=application, bitrate_kbps=bitrate_kbps)
    rt = rt[: B*N].reshape(B, N)
    return torch.from_numpy(rt).to(audio_t.device)


def test_iid(ckpt_path: str, n_batches: int = 4, batch_size: int = 64, **opus_kwargs):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = IIDEncoder(n_bits=n_bits).to(DEVICE)
    dec = IIDDecoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            audio = enc(bits)
            rx = opus_round_trip_batch(audio, **opus_kwargs)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers)), n_bits * 1000 / SYMBOL_MS


def test_highband(ckpt_path: str, n_batches: int = 4, batch_size: int = 64, **opus_kwargs):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = HighBandEncoder(n_bits=n_bits).to(DEVICE)
    dec = HighBandDecoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            audio = enc(bits)
            rx = opus_round_trip_batch(audio, **opus_kwargs)
            rx_band = hard_bandpass_torch(rx, HIGH_BAND_LO, HIGH_BAND_HI)
            logits = dec(rx_band)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers)), n_bits * 1000 / SYMBOL_MS


def test_seq(ckpt_path: str, n_batches: int = 4, batch_size: int = 64, **opus_kwargs):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits_per_chunk"] * N_CHUNKS
    enc = SeqEncoder(ckpt["n_bits_per_chunk"]).to(DEVICE)
    dec = SeqDecoder(ckpt["n_bits_per_chunk"]).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            audio = enc(bits)
            rx = opus_round_trip_batch(audio, **opus_kwargs)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers)), n_bits / (BLOCK_N / SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitrate_kbps", type=int, default=24)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    iid_path = str(REPO_ROOT / "core/neural_codec/ckpt_n128_mixed.pt")
    seq_path = str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt")
    hb_path = str(REPO_ROOT / "core/neural_codec/composite_attempt/ckpt_highband.pt")

    print(f"# Comparing VOIP vs AUDIO Opus mode at {args.bitrate_kbps} kbps")
    print(f"# {args.n_batches * args.batch_size} samples per measurement")
    print()
    print(f"{'checkpoint':<30}  {'rate':>10}  {'VOIP_BER':>9}  {'AUDIO_BER':>10}  {'Δ':>8}")

    for label, fn, path in [
        ("IID baseline n=128", test_iid, iid_path),
        ("SEQ adv n=32", test_seq, seq_path),
        ("HighBand specialist (failed)", test_highband, hb_path),
    ]:
        if not Path(path).exists():
            print(f"  {label:<28} (missing {path})"); continue
        ber_voip, raw = fn(path, args.n_batches, args.batch_size,
                           application="voip", bitrate_kbps=args.bitrate_kbps)
        ber_audio, _ = fn(path, args.n_batches, args.batch_size,
                          application="audio", bitrate_kbps=args.bitrate_kbps)
        delta = (ber_voip - ber_audio) / max(ber_voip, 1e-9) * 100
        print(f"  {label:<28}  {raw:>7.0f} bps   {ber_voip*100:>6.2f}%   {ber_audio*100:>7.2f}%   {delta:+6.1f}%")


if __name__ == "__main__":
    main()
