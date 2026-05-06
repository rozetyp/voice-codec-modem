"""
Probe γ: prosody-only modem. Encode bits as small pitch deviations on a real-
speech carrier; decode by tracking pitch on the receiver. Pitch is preserved by
every speech codec by definition (it IS one of the codec's parameters), so this
should survive Opus, AMR-NB, *and* the speaker→mic loop that's been killing
synthesized carriers.

Pipeline:
  real_speech.wav  --pyworld extract f0,sp,ap-->  modify f0 (bit i ⇒ ±Δ cents)
                                              -->  pyworld synth
                                              -->  Opus 24k VOIP round-trip
                                              -->  pyworld extract f0
                                              -->  threshold per window  -->  bits

Rate target: 100–200 bps reliable. Floor.
"""
from __future__ import annotations
import argparse, subprocess, tempfile, sys
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import pyworld as pw
except ImportError:
    print("pip install pyworld", file=sys.stderr); sys.exit(1)


def opus_rt(audio_np: np.ndarray, sr: int, kbps: int = 24, app: str = "voip") -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.wav"
        opx = Path(tmp) / "x.opus"
        out = Path(tmp) / "out.wav"
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


def encode(audio: np.ndarray, sr: int, bits: np.ndarray,
           win_ms: float, delta_cents: float):
    """Modulate f0: bit=1 → +delta_cents, bit=0 → -delta_cents, in win_ms windows."""
    audio = audio.astype(np.float64)
    f0, t = pw.dio(audio, sr, frame_period=5.0)  # 5 ms frame
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)

    n_frames = len(f0)
    frames_per_win = int(win_ms / 5.0)
    n_wins = n_frames // frames_per_win
    bits_used = min(len(bits), n_wins)

    f0_mod = f0.copy()
    cent = 2 ** (delta_cents / 1200.0)
    for i in range(bits_used):
        a, b = i*frames_per_win, (i+1)*frames_per_win
        seg = f0_mod[a:b]
        voiced = seg > 0
        factor = cent if bits[i] == 1 else 1.0/cent
        seg[voiced] = seg[voiced] * factor
        f0_mod[a:b] = seg
    y = pw.synthesize(f0_mod, sp, ap, sr, 5.0)
    return y.astype(np.float32), bits_used, f0  # also return clean f0 baseline


def decode(audio: np.ndarray, sr: int, baseline_f0: np.ndarray,
           n_bits: int, win_ms: float, min_voiced_frames: int = 6):
    """Re-extract f0 from received audio, compare against baseline f0 in each
    window: average ratio > 1 ⇒ bit 1, < 1 ⇒ bit 0.
    Returns bits, confidences, voiced_mask (True = window had enough voiced frames)."""
    audio = audio.astype(np.float64)
    f0, t = pw.dio(audio, sr, frame_period=5.0)
    f0 = pw.stonemask(audio, f0, t, sr)
    n = min(len(f0), len(baseline_f0))
    f0 = f0[:n]; base = baseline_f0[:n]
    frames_per_win = int(win_ms / 5.0)

    bits_out = np.zeros(n_bits, dtype=np.int32)
    confidences = np.zeros(n_bits)
    voiced_mask = np.zeros(n_bits, dtype=bool)
    for i in range(n_bits):
        a, b = i*frames_per_win, (i+1)*frames_per_win
        if b > n: break
        seg_rx = f0[a:b]; seg_bs = base[a:b]
        voiced = (seg_rx > 0) & (seg_bs > 0)
        if voiced.sum() < min_voiced_frames:
            continue
        voiced_mask[i] = True
        cents = 1200.0 * np.log2(seg_rx[voiced] / seg_bs[voiced])
        m = float(np.mean(cents))
        bits_out[i] = 1 if m > 0 else 0
        confidences[i] = abs(m)
    return bits_out, confidences, voiced_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None,
                    help="WAV file (any sr); if absent, synthesize from a TTS-ish drone")
    ap.add_argument("--win-ms", type=float, default=80.0,
                    help="window per bit (ms). 80 ms ⇒ 12.5 bps")
    ap.add_argument("--delta-cents", type=float, default=80.0,
                    help="pitch shift magnitude per bit (cents)")
    ap.add_argument("--n-bits", type=int, default=64)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    bits = np.random.randint(0, 2, size=args.n_bits, dtype=np.int32)

    # Get carrier
    if args.audio and Path(args.audio).exists():
        audio, sr_in = sf.read(args.audio)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr_in != args.sr:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, args.sr, sr_in)
        sr = args.sr
        # need ≥ n_bits * win_ms of audio
        need = int(args.n_bits * args.win_ms / 1000 * sr) + sr  # +1 sec headroom
        if len(audio) < need:
            reps = need // len(audio) + 1
            audio = np.tile(audio, reps)
        audio = audio[:need].astype(np.float32)
    else:
        # No real-speech available; fall back to a synthetic vowel-like drone:
        # carrier = harmonic stack at f0=140Hz with formants, modulated by vowel envelope.
        sr = args.sr
        dur = args.n_bits * args.win_ms / 1000 + 1.0
        n = int(sr * dur)
        t = np.arange(n) / sr
        f0 = 140.0
        # vowel /a/ formants ≈ 700, 1220, 2600
        sig = np.zeros(n, dtype=np.float32)
        for h, gain in [(1, 1.0), (2, 0.5), (3, 0.4), (4, 0.3),
                        (5, 0.25), (6, 0.2), (7, 0.15), (8, 0.1)]:
            sig += gain * np.sin(2*np.pi*h*f0*t).astype(np.float32)
        # AM modulation to mimic syllabic envelope
        env = 0.6 + 0.4*np.sin(2*np.pi*4*t)
        audio = (sig * env).astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.7

    print(f"[γ] sr={sr}  audio_dur={len(audio)/sr:.1f}s  n_bits={args.n_bits}  "
          f"win_ms={args.win_ms}  Δ={args.delta_cents}c  rate={1000/args.win_ms:.1f} bps")

    # Encode
    y_mod, bits_used, baseline_f0 = encode(audio, sr, bits, args.win_ms, args.delta_cents)
    print(f"[γ] encoded {bits_used} bits into {len(y_mod)/sr:.1f}s of audio")

    # Save for inspection
    out_dir = Path("/tmp/probe_gamma")
    out_dir.mkdir(exist_ok=True)
    sf.write(out_dir/"clean.wav", audio, sr)
    sf.write(out_dir/"mod.wav", y_mod, sr)

    # Test 1: clean (no codec) — sanity check
    bits_clean, _, vmask_clean = decode(y_mod, sr, baseline_f0, bits_used, args.win_ms)
    err_clean = int(np.sum(bits_clean[:bits_used] != bits[:bits_used]))
    print(f"[γ] CLEAN  BER: {err_clean}/{bits_used} = {err_clean/bits_used*100:.2f}%")

    # Test 2: through Opus 24k VOIP (Zoom-class)
    y_opus = opus_rt(y_mod, sr, kbps=24, app="voip")
    sf.write(out_dir/"opus_voip.wav", y_opus, sr)
    bits_opus, _ = decode(y_opus, sr, baseline_f0, bits_used, args.win_ms)
    err_opus = int(np.sum(bits_opus[:bits_used] != bits[:bits_used]))
    print(f"[γ] OPUS24k BER: {err_opus}/{bits_used} = {err_opus/bits_used*100:.2f}%")

    # Test 3: through Opus 12k VOIP (WhatsApp-class)
    y_opus12 = opus_rt(y_mod, sr, kbps=12, app="voip")
    sf.write(out_dir/"opus_12k.wav", y_opus12, sr)
    bits_opus12, _ = decode(y_opus12, sr, baseline_f0, bits_used, args.win_ms)
    err_opus12 = int(np.sum(bits_opus12[:bits_used] != bits[:bits_used]))
    print(f"[γ] OPUS12k BER: {err_opus12}/{bits_used} = {err_opus12/bits_used*100:.2f}%")

    # Test 4: through Opus 8k VOIP (cellular VoLTE-class)
    y_opus8 = opus_rt(y_mod, sr, kbps=8, app="voip")
    sf.write(out_dir/"opus_8k.wav", y_opus8, sr)
    bits_opus8, _ = decode(y_opus8, sr, baseline_f0, bits_used, args.win_ms)
    err_opus8 = int(np.sum(bits_opus8[:bits_used] != bits[:bits_used]))
    print(f"[γ] OPUS8k  BER: {err_opus8}/{bits_used} = {err_opus8/bits_used*100:.2f}%")

    print(f"\n[γ] outputs saved to {out_dir}")
    print(f"[γ] rate={1000/args.win_ms:.1f} bps raw, before any FEC")


if __name__ == "__main__":
    main()
