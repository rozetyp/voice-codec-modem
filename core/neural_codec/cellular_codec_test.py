"""
Test our existing checkpoints under cellular-realistic codecs:
  - AMR-NB at 12.2 kbps (2G/3G narrowband, 8 kHz native — harsh)
  - Opus 12 kbps VOIP (proxy for AMR-WB; similar bitrate budget at wideband)
  - Opus 8 kbps VOIP (very tight, similar to lowest AMR-WB modes)

The 24k VOIP we've been targeting is an *app* threat model. Cellular voice
runs much tighter. This is the realistic test for the original
"tunnel through a phone call in a restricted country" use case.
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
from stego_codec import StegEncoder, StegDecoder
from adversarial_realism import RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000
WORK = Path(tempfile.gettempdir()) / "cellular_test"
WORK.mkdir(exist_ok=True)


def codec_round_trip(audio_np: np.ndarray, codec: str) -> np.ndarray:
    """Round-trip 1D audio through the named codec config. audio_np: float32 in [-1,1] @ 16 kHz."""
    tag = str(np.random.randint(1, 1<<30))
    inp = WORK/f"in_{tag}.wav"
    out = WORK/f"out_{tag}.wav"

    if codec == "amrnb":
        # AMR-NB requires 8 kHz mono. ffmpeg handles resample.
        opx = WORK/f"x_{tag}.amr"
        sf.write(inp, audio_np.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopencore_amrnb","-ar","8000","-ac","1",
                        "-b:a","12.2k", str(opx)], check=True)
        # Decode back; ffmpeg up-resamples to 16k for our pipeline
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1",str(out)], check=True)
        Path(opx).unlink(missing_ok=True)

    elif codec == "opus_12k_voip":
        opx = WORK/f"x_{tag}.opus"
        sf.write(inp, audio_np.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a","12k","-application","voip", str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1",str(out)], check=True)
        Path(opx).unlink(missing_ok=True)

    elif codec == "opus_8k_voip":
        opx = WORK/f"x_{tag}.opus"
        sf.write(inp, audio_np.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a","8k","-application","voip", str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1",str(out)], check=True)
        Path(opx).unlink(missing_ok=True)

    elif codec == "opus_24k_voip":  # baseline for comparison
        opx = WORK/f"x_{tag}.opus"
        sf.write(inp, audio_np.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a","24k","-application","voip", str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1",str(out)], check=True)
        Path(opx).unlink(missing_ok=True)
    else:
        raise ValueError(codec)

    a, _ = sf.read(out)
    Path(inp).unlink(missing_ok=True); Path(out).unlink(missing_ok=True)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio_np):
        a = np.pad(a, (0, len(audio_np) - len(a)))
    elif len(a) > len(audio_np):
        a = a[:len(audio_np)]
    return a.astype(np.float32)


def codec_round_trip_batch(audio_t: torch.Tensor, codec: str) -> torch.Tensor:
    audio_np = audio_t.detach().cpu().numpy()
    B, N = audio_np.shape
    flat = audio_np.reshape(-1)
    rt = codec_round_trip(flat, codec)
    rt = rt[: B*N].reshape(B, N)
    return torch.from_numpy(rt).to(audio_t.device)


def test_iid(ckpt_path, codec, n_batches=4, batch_size=64):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = IIDEncoder(n_bits=n_bits).to(DEVICE); dec = IIDDecoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            audio = enc(bits)
            rx = codec_round_trip_batch(audio, codec)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers))


def test_seq(ckpt_path, codec, n_batches=4, batch_size=64):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc = SeqEncoder(ckpt["n_bits_per_chunk"]).to(DEVICE); dec = SeqDecoder(ckpt["n_bits_per_chunk"]).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    n_bits = ckpt["n_bits_per_chunk"] * N_CHUNKS
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            audio = enc(bits)
            rx = codec_round_trip_batch(audio, codec)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers))


def test_stego(ckpt_path, codec, n_batches=4, batch_size=64):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]; pert = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits).to(DEVICE); dec = StegDecoder(n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    sampler = RealSpeechSampler([
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav"))
    ])
    bers = []
    for b in range(n_batches):
        torch.manual_seed(b)
        cover = sampler.sample(batch_size)
        bits = torch.randint(0, 2, (batch_size, n_bits), device=DEVICE).float()
        with torch.no_grad():
            modified, _ = enc(cover, bits, perturbation_scale=pert)
            rx = codec_round_trip_batch(modified, codec)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bers.append((preds != bits).float().mean().item())
    return float(np.mean(bers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    iid_path  = str(REPO_ROOT / "core/neural_codec/ckpt_n128_mixed.pt")
    seq_path  = str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt")
    stego_path = str(REPO_ROOT / "core/neural_codec/stego/ckpt_stego_p3.pt")

    print(f"# {args.n_batches * args.batch_size} samples per measurement")
    print()
    print(f"{'codec':<20}  {'IID n=128 (4267 raw)':>24}  {'SEQ adv (1067 raw)':>22}  {'Stego (1067 raw)':>20}")
    for codec in ["opus_24k_voip", "opus_12k_voip", "opus_8k_voip", "amrnb"]:
        iid_ber = test_iid(iid_path, codec, args.n_batches, args.batch_size) if Path(iid_path).exists() else float("nan")
        seq_ber = test_seq(seq_path, codec, args.n_batches, args.batch_size) if Path(seq_path).exists() else float("nan")
        stego_ber = test_stego(stego_path, codec, args.n_batches, args.batch_size) if Path(stego_path).exists() else float("nan")
        print(f"  {codec:<18}    {iid_ber*100:>20.2f}%  {seq_ber*100:>20.2f}%  {stego_ber*100:>18.2f}%")


if __name__ == "__main__":
    main()
