"""Voice-within-voice demo for the sequence-level codec."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import subprocess
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch

from neural_codec import real_opus_batch, DEVICE, SR
from seq_codec import SeqEncoder, SeqDecoder, BLOCK_N, N_CHUNKS
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes
from voice_within_voice_demo import codec2_encode, codec2_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav"))
    ap.add_argument("--start_s", type=float, default=20.0)
    ap.add_argument("--length_s", type=float, default=10.0)
    ap.add_argument("--codec2_mode", type=int, default=1200)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/sequence/ckpt_seq_adv.pt"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "sequence" / "seq_vwv_out"))
    ap.add_argument("--fec_n_data", type=int, default=191)
    ap.add_argument("--fec_n_total", type=int, default=255)
    ap.add_argument("--fec_depth", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# input={args.input}  clip [{args.start_s}, {args.start_s + args.length_s}] s")
    print(f"# codec2 mode={args.codec2_mode} bps  ckpt={args.ckpt}")

    audio, sr = sf.read(args.input)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000: audio = sps.resample_poly(audio, 16000, sr); sr = 16000
    s = int(args.start_s * sr); e = s + int(args.length_s * sr)
    clip_16k = audio[s:e].astype(np.float32)
    sf.write(out_dir/"01_original_16k.wav", clip_16k, 16000)
    clip_8k = sps.resample_poly(clip_16k, 1, 2)
    clip_8k = (np.clip(clip_8k, -1, 1) * 32767).astype(np.int16)

    payload = codec2_encode(clip_8k, args.codec2_mode)
    print(f"# Codec2 payload: {len(payload)} bytes ({len(payload)*8/args.length_s/1000:.2f} kbps actual)")

    ref_8k = codec2_decode(payload, args.codec2_mode)
    sf.write(out_dir/"04_ref_codec2_only_16k.wav",
             sps.resample_poly(ref_8k.astype(np.float32) / 32768, 2, 1), 16000)

    # Load seq codec
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    n_bits_per_chunk = ckpt["n_bits_per_chunk"]
    n_bits_total = n_bits_per_chunk * N_CHUNKS
    enc = SeqEncoder(n_bits_per_chunk).to(DEVICE)
    dec = SeqDecoder(n_bits_per_chunk).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    fec = InterleavedRS(n_data=args.fec_n_data, n_total=args.fec_n_total, depth=args.fec_depth)

    coded, pad = fec.encode(payload)
    tx_bits = bytes_to_bits(coded)
    pad_to_block = (-len(tx_bits)) % n_bits_total
    tx_bits_full = np.concatenate([tx_bits, np.zeros(pad_to_block, dtype=np.uint8)])
    n_blocks = len(tx_bits_full) // n_bits_total
    blocks = tx_bits_full.reshape(n_blocks, n_bits_total)

    print(f"# transmitting {len(payload)} payload bytes through seq+adv neural codec + libopus 24k VoIP …")
    rx_bits_chunks = []
    audio_chunks = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, n_blocks, batch_size):
            sb = torch.from_numpy(blocks[i:i+batch_size].astype(np.float32)).to(DEVICE)
            au = enc(sb)
            audio_chunks.append(au.cpu().numpy())
            au_codec = real_opus_batch(au)
            logits = dec(au_codec)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
    rx_bits = np.concatenate(rx_bits_chunks)[:len(tx_bits)]
    raw_ber = float(np.mean(rx_bits != tx_bits))
    rx_bytes = bits_to_bytes(rx_bits)
    decoded_payload, n_corr, n_fail = fec.decode(rx_bytes, pad)
    decoded_payload = decoded_payload[:len(payload)]

    transmitted = np.concatenate(audio_chunks, axis=0).reshape(-1)
    sf.write(out_dir/"05_transmitted_carrier_audio_16k.wav",
             np.clip(transmitted, -1, 1).astype(np.float32), 16000)

    print(f"# raw bit BER: {raw_ber*100:.3f}%  (FEC blocks: {n_corr} ok, {n_fail} failed)")
    print(f"# channel time: {len(transmitted)/SR:.2f}s of synthetic audio for {args.length_s}s of speech")
    n_byte_diff = sum(a!=b for a,b in zip(payload, decoded_payload))
    print(f"# bytes mismatch: {n_byte_diff} of {len(payload)}")

    if decoded_payload == payload:
        print("# ✅ payload bytes EXACTLY recovered")
    rec_8k = codec2_decode(decoded_payload, args.codec2_mode)
    sf.write(out_dir/"07_recovered_voice_within_voice_16k.wav",
             sps.resample_poly(rec_8k.astype(np.float32) / 32768, 2, 1), 16000)

    print(f"\n# outputs in {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:<48}  {f.stat().st_size:>10} bytes")


if __name__ == "__main__":
    main()
