"""Standalone smoke-test for the prosody modem: clean encode/decode +
through-Opus end-to-end at 24k/12k/8k VOIP and 48k AUDIO."""
from __future__ import annotations
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.neural_codec.prosody import prosody_modem as pm  # type: ignore


def opus_rt(audio_np: np.ndarray, sr: int, kbps: int, app: str) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", default="app/static/cover.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win-ms", type=float, default=80.0)
    ap.add_argument("--delta-cents", type=float, default=100.0)
    ap.add_argument("--n-bits", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cover, sr_in = sf.read(args.cover)
    if cover.ndim > 1: cover = cover.mean(axis=1)
    if sr_in != args.sr:
        from scipy.signal import resample_poly
        cover = resample_poly(cover, args.sr, sr_in)
    cover = cover.astype(np.float32)
    print(f"cover: {len(cover)/args.sr:.1f}s @ {args.sr} Hz")

    rng = np.random.RandomState(args.seed)
    bits = rng.randint(0, 2, size=args.n_bits, dtype=np.int32)

    print(f"\n=== prosody modem: win={args.win_ms}ms  Δ={args.delta_cents}c "
          f"intent_n_bits={args.n_bits} ===\n")

    # Encode (returns the cover_f0 baseline; both ends know cover so this is shareable)
    audio_mod, used, cover_f0 = pm.encode(
        cover, args.sr, bits, win_ms=args.win_ms, delta_cents=args.delta_cents)
    secs = len(audio_mod) / args.sr
    print(f"encoded {len(used)} bits in {secs:.1f}s ({len(used)/secs:.2f} bps raw)")

    # Self-test (no codec)
    bits_clean, _ = pm.decode(audio_mod, args.sr, cover_f0,
                              n_bits=len(used), win_ms=args.win_ms)
    n = min(len(used), len(bits_clean))
    err = int((bits[:n] != bits_clean[:n]).sum())
    print(f"\nCLEAN     n={n}  errors={err}  BER={err/n*100:.2f}%")

    # Through Opus modes
    for kbps, app in [(24, "voip"), (12, "voip"), (8, "voip"), (48, "audio")]:
        y = opus_rt(audio_mod, args.sr, kbps, app)
        bits_out, _ = pm.decode(y, args.sr, cover_f0,
                                n_bits=len(used), win_ms=args.win_ms)
        n = min(len(used), len(bits_out))
        err = int((bits[:n] != bits_out[:n]).sum())
        print(f"OPUS{kbps:>2}k {app:<5}  n={n}  errors={err}  BER={err/n*100:.2f}%")

    # Save outputs for ear-test
    out = Path("/tmp/prosody_modem"); out.mkdir(exist_ok=True)
    sf.write(out/"cover.wav", cover, args.sr)
    sf.write(out/"modulated.wav", audio_mod, args.sr)
    print(f"\nsaved {out}/cover.wav and modulated.wav")


if __name__ == "__main__":
    main()
