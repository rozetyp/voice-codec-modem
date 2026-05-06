"""
End-to-end test of stacked prosody + stego_opus channels through Opus 24k VOIP.

Pipeline:
  cover.wav  -- prosody.encode(bits1) -->  audio_with_pitch_mod
                                             |
                   StegoOpusPipeline.encode(text2)  (uses the pitch-mod audio
                                             |        as its cover)
                                             v
                                       audio_stego
                                             |
                                       Opus 24k VOIP round-trip
                                             v
                                       audio_received
                          /                    \\
        StegoOpusPipeline.decode    prosody.decode(cover_f0)
              |                                 |
            text2_recovered                  bits1_recovered

Channels are spectrally orthogonal: prosody acts on f0/harmonics, stego adds
broadband perturbation. We expect both decoders to operate independently.
"""
from __future__ import annotations
import argparse, io, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.neural_codec.prosody import prosody_modem as pm  # type: ignore
from app.pipelines import StegoOpusPipeline, SR  # type: ignore


def opus_rt(audio_np: np.ndarray, sr: int, kbps: int = 24, app: str = "voip") -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.opus"; out = Path(tmp)/"out.wav"
        sf.write(inp, audio_np.astype(np.float32), sr)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a",f"{kbps}k","-application",app,
                        str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(sr),"-ac","1",str(out)], check=True)
        a, _ = sf.read(out)
    if a.ndim > 1: a = a.mean(axis=1)
    if len(a) < len(audio_np): a = np.pad(a, (0, len(audio_np)-len(a)))
    elif len(a) > len(audio_np): a = a[:len(audio_np)]
    return a.astype(np.float32)


def numpy_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello! This is a stacked-channel test transmission.")
    ap.add_argument("--n-pitch-bits", type=int, default=128)
    ap.add_argument("--win-ms", type=float, default=80.0)
    ap.add_argument("--delta-cents", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opus-kbps", type=int, default=24)
    ap.add_argument("--opus-app", default="voip")
    args = ap.parse_args()

    print(f"=== Stacked channel test ===")
    print(f"  text:      {args.text!r}")
    print(f"  prosody:   {args.n_pitch_bits} bits, win={args.win_ms}ms, Δ={args.delta_cents}c")
    print(f"  Opus:      {args.opus_kbps}k {args.opus_app}\n")

    # ----- Stage 0: load cover -----
    cover_path = ROOT / "app" / "static" / "cover.wav"
    cover, sr_in = sf.read(cover_path)
    if cover.ndim > 1: cover = cover.mean(axis=1)
    if sr_in != SR:
        from scipy.signal import resample_poly
        cover = resample_poly(cover, SR, sr_in)
    cover = cover.astype(np.float32)
    print(f"cover: {len(cover)/SR:.1f}s @ {SR} Hz")

    # ----- Stage 1: prosody encode -----
    rng = np.random.RandomState(args.seed)
    pitch_bits = rng.randint(0, 2, size=args.n_pitch_bits, dtype=np.int32)
    audio_pitch, used_pitch, cover_f0 = pm.encode(
        cover, SR, pitch_bits, win_ms=args.win_ms, delta_cents=args.delta_cents)
    n_sent = len(used_pitch)
    print(f"\n[1] prosody: encoded {n_sent} bits ({n_sent/(len(audio_pitch)/SR):.2f} bps raw)")

    # ----- Stage 2: stego on the pitch-modulated audio -----
    # Stego encoder uses its bundled cover internally; we monkey-patch
    # the loaded cover so it uses our pitch-modulated audio as the cover.
    stego = StegoOpusPipeline()
    stego._cover = audio_pitch.copy()
    stego_result = stego.encode_text(args.text)
    audio_stego = sf.read(io.BytesIO(stego_result.wav_bytes))[0].astype(np.float32)
    print(f"[2] stego:   embedded {stego_result.n_data_bytes} bytes ({stego_result.eff_bps:.0f} eff_bps), audio {len(audio_stego)/SR:.1f}s")

    # ----- Stage 3: Opus round-trip -----
    audio_rx = opus_rt(audio_stego, SR, args.opus_kbps, args.opus_app)
    print(f"[3] Opus:    {args.opus_kbps}k {args.opus_app} round-trip done")

    # ----- Stage 4: decode both channels -----
    # Stego decoder
    rx_wav = numpy_to_wav_bytes(audio_rx, SR)
    stego_dec = stego.decode_audio(rx_wav, expected_text_bytes=stego_result.n_data_bytes)
    stego_text_ok = (stego_dec.text == args.text)
    print(f"\n[4a] stego decode: text={'✓' if stego_text_ok else '✗'}  "
          f"recovered={stego_dec.text!r}  fec_blocks corrected={stego_dec.n_fec_blocks_corrected}, failed={stego_dec.n_fec_blocks_failed}")

    # Prosody decoder
    pitch_out, used_dec = pm.decode(audio_rx, SR, cover_f0,
                                    n_bits=n_sent, win_ms=args.win_ms)
    n = min(n_sent, len(pitch_out))
    pitch_errs = int((pitch_bits[:n] != pitch_out[:n]).sum())
    print(f"[4b] pitch decode: n={n}  errors={pitch_errs}  BER={pitch_errs/n*100:.2f}%")

    # ----- Summary -----
    secs = len(audio_stego) / SR
    raw_pitch_bps = n_sent / secs
    raw_stego_bps = stego_result.raw_bps
    eff_stego_bps = stego_result.eff_bps if stego_text_ok else 0.0  # only counts if delivered
    print(f"\n=== summary over {secs:.1f}s of audio ===")
    print(f"  pitch channel: {raw_pitch_bps:.2f} bps raw, {pitch_errs/n*100:.2f}% BER")
    print(f"  stego channel: {raw_stego_bps:.0f} bps raw, {eff_stego_bps:.0f} bps reliable (text {'delivered' if stego_text_ok else 'FAILED'})")
    if pitch_errs == 0 and stego_text_ok:
        print(f"  total reliable: {eff_stego_bps + raw_pitch_bps:.0f} bps  (stego + pitch)")


if __name__ == "__main__":
    main()
