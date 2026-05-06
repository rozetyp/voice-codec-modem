"""
Voice-within-voice end-to-end demo.

Pipeline:
   real speech (16 kHz)
   -> resample to 8 kHz int16 PCM
   -> Codec2 @ 2400 bps  -> data bytes
   -> RS(255,191) interleaved FEC encode
   -> bits
   -> neural Encoder (ckpt_n128_mixed)
   -> 30 ms symbols of synthetic audio
   -> ffmpeg -c:a libopus -b:a 24k -application voip   (the channel)
   -> neural Decoder
   -> bits
   -> RS decode
   -> data bytes
   -> Codec2 decode @ 2400 bps
   -> 8 kHz int16 PCM
   -> upsample to 16 kHz
   -> save WAV

If the resulting WAV is intelligible, voice-within-voice works.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))

import numpy as np
import soundfile as sf
import scipy.signal as sps
import torch
import reedsolo

from neural_codec import Encoder, Decoder, real_opus_batch, DEVICE, SR, SYMBOL_MS
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes


def codec2_encode(speech_8k_i16: np.ndarray, mode_bps: int) -> bytes:
    """Encode 8kHz int16 PCM via codec2 -> headerless raw bit file."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f_in:
        speech_8k_i16.astype("<i2").tofile(f_in)
        in_path = f_in.name
    out_path = in_path + ".bin"
    subprocess.run(["c2enc", str(mode_bps), in_path, out_path], check=True)
    data = Path(out_path).read_bytes()
    Path(in_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)
    return data


def codec2_decode(coded: bytes, mode_bps: int) -> np.ndarray:
    """Decode codec2 bits back to 8kHz int16 PCM."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_in:
        f_in.write(coded)
        in_path = f_in.name
    out_path = in_path + ".raw"
    subprocess.run(["c2dec", str(mode_bps), in_path, out_path], check=True)
    audio = np.fromfile(out_path, dtype="<i2")
    Path(in_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)
    return audio


def neural_channel_send(data_bytes: bytes, ckpt_path: str,
                         fec_n_data: int = 191, fec_n_total: int = 255, fec_depth: int = 8,
                         batch_size: int = 256):
    """Send data_bytes through neural-codec + RS-FEC + real libopus, return recovered bytes."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits_per_sym = ckpt["n_bits"]
    enc = Encoder(n_bits=n_bits_per_sym).to(DEVICE)
    dec = Decoder(n_bits=n_bits_per_sym).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()

    fec = InterleavedRS(n_data=fec_n_data, n_total=fec_n_total, depth=fec_depth)
    coded, pad = fec.encode(data_bytes)
    tx_bits = bytes_to_bits(coded)
    pad_to_sym = (-len(tx_bits)) % n_bits_per_sym
    tx_bits_full = np.concatenate([tx_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
    n_sym = len(tx_bits_full) // n_bits_per_sym
    symbols = tx_bits_full.reshape(n_sym, n_bits_per_sym)

    # Process in batches to keep memory sane and to be representative of a stream
    rx_bits_chunks = []
    raw_errs = 0; raw_total = 0
    audio_chunks = []
    with torch.no_grad():
        for i in range(0, n_sym, batch_size):
            sb = torch.from_numpy(symbols[i:i+batch_size].astype(np.float32)).to(DEVICE)
            audio = enc(sb)
            audio_chunks.append(audio.cpu().numpy())
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
            rx_bits_chunks.append(preds.reshape(-1))
            raw_errs += int(np.sum(preds != symbols[i:i+batch_size]))
            raw_total += preds.size
    rx_bits = np.concatenate(rx_bits_chunks)[:len(tx_bits)]
    raw_ber = raw_errs / raw_total
    rx_bytes = bits_to_bytes(rx_bits)
    decoded, n_corr, n_fail = fec.decode(rx_bytes, pad)
    decoded = decoded[:len(data_bytes)]
    transmitted_audio = np.concatenate(audio_chunks, axis=0).reshape(-1)
    return decoded, dict(raw_ber=raw_ber, n_fec_blocks_corrected=n_corr,
                          n_fec_blocks_failed=n_fail, transmitted_audio=transmitted_audio,
                          n_bits_per_sym=n_bits_per_sym, n_sym=n_sym,
                          channel_seconds=n_sym * SYMBOL_MS / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav"))
    ap.add_argument("--start_s", type=float, default=20.0)
    ap.add_argument("--length_s", type=float, default=10.0)
    ap.add_argument("--codec2_mode", type=int, default=2400, choices=[1200, 1300, 1400, 1600, 2400, 3200])
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "core/neural_codec/ckpt_n128_mixed.pt"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "core" / "neural_codec" / "demo_outputs"))
    ap.add_argument("--fec_n_data", type=int, default=191, help="RS data bytes (default 191 -> rate 191/255)")
    ap.add_argument("--fec_n_total", type=int, default=255)
    ap.add_argument("--fec_depth", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# input={args.input}")
    print(f"# clip [{args.start_s}, {args.start_s + args.length_s}] s")
    print(f"# codec2 mode={args.codec2_mode} bps")
    print(f"# neural codec ckpt={args.ckpt}")

    # Load and prepare clip
    audio, sr = sf.read(args.input)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        audio = sps.resample_poly(audio, 16000, sr); sr = 16000
    s = int(args.start_s * sr); e = s + int(args.length_s * sr)
    clip_16k = audio[s:e].astype(np.float32)
    sf.write(out_dir/"01_original_16k.wav", clip_16k, 16000)

    # 16k -> 8k int16 (Codec2 native)
    clip_8k = sps.resample_poly(clip_16k, 1, 2)
    clip_8k = (np.clip(clip_8k, -1, 1) * 32767).astype(np.int16)
    sf.write(out_dir/"02_clip_8k.wav", clip_8k, 8000)

    # Codec2 encode -> bytes
    payload = codec2_encode(clip_8k, args.codec2_mode)
    payload_kbps = len(payload) * 8 / args.length_s / 1000
    print(f"# Codec2 payload: {len(payload)} bytes ({payload_kbps:.2f} kbps actual)")

    # Reference: round-trip Codec2 with no channel
    ref_8k = codec2_decode(payload, args.codec2_mode)
    sf.write(out_dir/"03_ref_codec2_only_8k.wav", ref_8k, 8000)
    sf.write(out_dir/"04_ref_codec2_only_16k.wav",
             sps.resample_poly(ref_8k.astype(np.float32) / 32768, 2, 1), 16000)

    # Through the neural channel
    print(f"# transmitting through neural codec + libopus 24k VoIP, "
          f"FEC RS({args.fec_n_total},{args.fec_n_data})x{args.fec_depth} …")
    recovered_payload, m = neural_channel_send(
        payload, args.ckpt,
        fec_n_data=args.fec_n_data, fec_n_total=args.fec_n_total, fec_depth=args.fec_depth,
    )
    print(f"# raw bit BER: {m['raw_ber']*100:.3f}%  "
          f"(FEC blocks: {m['n_fec_blocks_corrected']} ok, {m['n_fec_blocks_failed']} failed)")
    print(f"# channel time: {m['channel_seconds']:.2f}s of synthetic audio for {args.length_s:.1f}s of speech")
    print(f"# bytes mismatch: {sum(a!=b for a,b in zip(payload, recovered_payload))} of {len(payload)}")

    # Save the synthetic transmitted audio (what an eavesdropper would hear on the line)
    transmitted = m["transmitted_audio"]
    sf.write(out_dir/"05_transmitted_carrier_audio_16k.wav",
             np.clip(transmitted, -1, 1).astype(np.float32), 16000)

    # Codec2 decode the recovered bytes -> 8k -> 16k
    recovered_8k = codec2_decode(recovered_payload, args.codec2_mode)
    sf.write(out_dir/"06_recovered_voice_within_voice_8k.wav", recovered_8k, 8000)
    sf.write(out_dir/"07_recovered_voice_within_voice_16k.wav",
             sps.resample_poly(recovered_8k.astype(np.float32) / 32768, 2, 1), 16000)

    # Quality metric: byte-perfect or close?
    if recovered_payload == payload:
        print("# ✅ payload bytes EXACTLY recovered — decoded audio bit-identical to Codec2 reference")
    else:
        n_diff = sum(a!=b for a,b in zip(payload, recovered_payload))
        print(f"# {n_diff}/{len(payload)} bytes differ; comparing audio sample-wise")
        L = min(len(ref_8k), len(recovered_8k))
        diff = (ref_8k[:L].astype(np.float32) - recovered_8k[:L].astype(np.float32))
        ref_pwr = (ref_8k[:L].astype(np.float32) ** 2).mean() + 1e-9
        snr_db = 10 * np.log10(ref_pwr / (diff ** 2).mean() + 1e-9)
        print(f"# audio SNR vs. Codec2-only reference: {snr_db:.1f} dB")

    print(f"\n# outputs in {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        sz = f.stat().st_size
        print(f"  {f.name:<50}  {sz:>10} bytes")


if __name__ == "__main__":
    main()
