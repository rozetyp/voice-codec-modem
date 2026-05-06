"""
TRIZ inversion experiment: encode bits as pitch values (Opus SILK preserves these natively),
round-trip through real libopus 24k VoIP, recover via autocorrelation, measure surviving bits/sym.
"""
import os
import numpy as np, soundfile as sf, subprocess, scipy.signal as sp
from pathlib import Path

SR = 16000
SYMBOL_MS = 40
N_SAMP = SR * SYMBOL_MS // 1000
WORK = Path(os.environ.get("MODEM_WORK_DIR", "/tmp/modem_test"))
WORK.mkdir(parents=True, exist_ok=True)

def voiced_tone(f0, n):
    """Glottal-pulse-like voiced signal at pitch f0. Keep fundamental loud
    (Opus SILK tracks pitch but discards weak f0). Mild formant shaping only."""
    t = np.arange(n) / SR
    # Harmonic stack with shallow rolloff so f0 + a few harmonics dominate
    sig = np.zeros(n)
    for k in range(1, 20):
        if k * f0 > SR/2 - 500: break
        # 1/sqrt(k) keeps f0 loud relative to high harmonics
        sig += np.sin(2*np.pi*k*f0*t) / np.sqrt(k)
    sig /= np.max(np.abs(sig)) + 1e-9
    # Gentle high-frequency rolloff (single-pole lowpass at 3 kHz)
    b, a = sp.butter(2, 3000/(SR/2), btype="low")
    out = sp.filtfilt(b, a, sig)
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.6
    # Soft attack/release to suppress symbol-edge clicks
    ramp = int(0.005 * SR)
    win = np.ones(n)
    win[:ramp] = np.linspace(0, 1, ramp)
    win[-ramp:] = np.linspace(1, 0, ramp)
    return (out * win).astype(np.float32)

def encode(bits, pitch_levels):
    bps = int(np.log2(len(pitch_levels)))
    audio = []
    syms = []
    for i in range(0, len(bits) - bps + 1, bps):
        chunk = bits[i:i+bps]
        idx = int("".join(map(str, chunk)), 2)
        syms.append(idx)
        audio.append(voiced_tone(pitch_levels[idx], N_SAMP))
    return np.concatenate(audio), np.array(syms)

def detect_pitch(seg):
    """Bandpass to voiced range, then autocorrelation peak. Reject frame edges."""
    s = int(len(seg) * 0.15)
    e = int(len(seg) * 0.85)
    seg = seg[s:e]
    if np.std(seg) < 1e-4:
        return 0.0
    # Bandpass 70-400 Hz to isolate fundamental
    b, a = sp.butter(4, [70/(SR/2), 400/(SR/2)], btype="band")
    seg = sp.filtfilt(b, a, seg)
    seg = seg - seg.mean()
    if np.std(seg) < 1e-5:
        return 0.0
    corr = np.correlate(seg, seg, mode="full")
    corr = corr[len(corr)//2:]
    min_lag = SR // 400  # 400 Hz max
    max_lag = SR // 70   # 70 Hz min
    if max_lag >= len(corr):
        return 0.0
    peak = np.argmax(corr[min_lag:max_lag]) + min_lag
    return SR / peak

def opus_rt(audio, tag):
    inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
    sf.write(inp, audio, SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                    "-c:a","libopus","-b:a","24k","-application","voip",str(opx)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                    "-ar",str(SR),"-ac","1",str(out)], check=True)
    a, _ = sf.read(out)
    return a.astype(np.float32)

def run(n_levels, n_symbols=400, seed=0):
    rng = np.random.default_rng(seed)
    bps = int(np.log2(n_levels))
    bits = rng.integers(0, 2, n_symbols * bps).tolist()
    # Log-spaced pitch levels in voiced range (perceptually + Opus-friendly)
    pitch_levels = np.exp(np.linspace(np.log(110), np.log(260), n_levels))
    audio, syms_tx = encode(bits, pitch_levels)
    audio_rx = opus_rt(audio, f"L{n_levels}")
    n_dec = min(len(syms_tx), len(audio_rx) // N_SAMP)
    syms_rx = []
    for i in range(n_dec):
        seg = audio_rx[i*N_SAMP:(i+1)*N_SAMP]
        f0 = detect_pitch(seg)
        idx = int(np.argmin(np.abs(pitch_levels - f0))) if f0 > 0 else 0
        syms_rx.append(idx)
    syms_rx = np.array(syms_rx)
    sym_err = np.mean(syms_rx != syms_tx[:n_dec])
    # Bit-level: gray-code mapping would be better; using natural binary here
    def to_bits(v, b): return [(v>>k)&1 for k in range(b-1,-1,-1)]
    bits_tx = np.array([b for s in syms_tx[:n_dec] for b in to_bits(int(s), bps)])
    bits_rx = np.array([b for s in syms_rx for b in to_bits(int(s), bps)])
    ber = np.mean(bits_tx != bits_rx)
    bitrate = bps * (1000 / SYMBOL_MS)
    return dict(levels=n_levels, bps=bps, bitrate=bitrate, sym_err=sym_err, ber=ber, n=n_dec)

if __name__ == "__main__":
    print(f"{'levels':>6} {'bits/sym':>9} {'bitrate':>9} {'sym_err':>8} {'BER':>7}  symbols")
    for N in [2, 4, 8, 16, 32, 64]:
        r = run(N)
        print(f"{r['levels']:>6} {r['bps']:>9} {r['bitrate']:>8.0f}bps {r['sym_err']*100:>7.1f}% {r['ber']*100:>6.2f}%  {r['n']}")
