"""
Three orthogonal Opus-friendly channels:
  1. Pitch (SILK preserves)
  2. Amplitude (gain encoding preserves)
  3. Brightness (low-vs-high spectral tilt -> LPC envelope; should survive)
Test 30ms and 40ms with each combination.
"""
import numpy as np, soundfile as sf, subprocess, scipy.signal as sp
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pitch_channel_probe import detect_pitch, opus_rt, SR

def voiced_with_tilt(f0, n, tilt):
    """tilt in [0,1]: 0 = darker (more low harmonics), 1 = brighter."""
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k in range(1, 25):
        if k * f0 > SR/2 - 500: break
        # Tilt-dependent rolloff: tilt=0 -> 1/k, tilt=1 -> 1/sqrt(k)
        rolloff = 1.0 - tilt * 0.5
        sig += np.sin(2*np.pi*k*f0*t) / (k ** rolloff)
    sig /= np.max(np.abs(sig)) + 1e-9
    b, a = sp.butter(2, 3500/(SR/2), btype="low")
    out = sp.filtfilt(b, a, sig)
    # Normalize to unit RMS so tilt is independent of amplitude
    out = out / (np.sqrt(np.mean(out**2)) + 1e-9) * 0.25
    ramp = int(0.005 * SR)
    win = np.ones(n); win[:ramp] = np.linspace(0,1,ramp); win[-ramp:] = np.linspace(1,0,ramp)
    return (out * win).astype(np.float32)

def make_symbol(f0, amp, tilt, n):
    return voiced_with_tilt(f0, n, tilt) * amp

def detect_amp(seg):
    s = int(len(seg) * 0.15); e = int(len(seg) * 0.85)
    return float(np.sqrt(np.mean(seg[s:e]**2)))

def detect_brightness(seg):
    """Ratio of energy above 1 kHz vs below 1 kHz (after removing very low band)."""
    s = int(len(seg) * 0.15); e = int(len(seg) * 0.85)
    seg = seg[s:e]
    if len(seg) < 64: return 0.0
    # Welch PSD
    f, P = sp.welch(seg, fs=SR, nperseg=min(256, len(seg)))
    low = np.sum(P[(f >= 200) & (f < 1000)])
    high = np.sum(P[(f >= 1000) & (f < 3000)])
    return float(high / (low + high + 1e-12))

def run(symbol_ms, n_pitch, n_amp, n_tilt, n_symbols=400, seed=0):
    n_samp = SR * symbol_ms // 1000
    rng = np.random.default_rng(seed)
    bps_p = int(np.log2(n_pitch)) if n_pitch > 1 else 0
    bps_a = int(np.log2(n_amp)) if n_amp > 1 else 0
    bps_t = int(np.log2(n_tilt)) if n_tilt > 1 else 0
    bps = bps_p + bps_a + bps_t
    if bps == 0: return None
    pitch_levels = np.exp(np.linspace(np.log(120), np.log(260), max(n_pitch, 1)))
    amp_levels = np.linspace(0.30, 0.95, max(n_amp, 1)) if n_amp > 1 else np.array([0.7])
    tilt_levels = np.linspace(0.0, 1.0, max(n_tilt, 1)) if n_tilt > 1 else np.array([0.5])

    bits = rng.integers(0, 2, n_symbols * bps).tolist()
    sp_tx, sa_tx, st_tx, audio = [], [], [], []
    for i in range(0, len(bits) - bps + 1, bps):
        chunk = bits[i:i+bps]; off = 0
        ip = int("".join(map(str, chunk[off:off+bps_p])), 2) if bps_p else 0; off += bps_p
        ia = int("".join(map(str, chunk[off:off+bps_a])), 2) if bps_a else 0; off += bps_a
        it = int("".join(map(str, chunk[off:off+bps_t])), 2) if bps_t else 0
        sp_tx.append(ip); sa_tx.append(ia); st_tx.append(it)
        audio.append(make_symbol(pitch_levels[ip], amp_levels[ia], tilt_levels[it], n_samp))
    sp_tx, sa_tx, st_tx = map(np.array, (sp_tx, sa_tx, st_tx))
    audio = np.concatenate(audio).astype(np.float32)
    rx = opus_rt(audio, f"s{symbol_ms}_{n_pitch}_{n_amp}_{n_tilt}")

    n_dec = min(len(sp_tx), len(rx) // n_samp)
    rx_p_raw, rx_a_raw, rx_t_raw = [], [], []
    for i in range(n_dec):
        seg = rx[i*n_samp:(i+1)*n_samp]
        rx_p_raw.append(detect_pitch(seg))
        rx_a_raw.append(detect_amp(seg))
        rx_t_raw.append(detect_brightness(seg))
    rx_p_raw = np.array(rx_p_raw); rx_a_raw = np.array(rx_a_raw); rx_t_raw = np.array(rx_t_raw)

    rx_p = np.array([int(np.argmin(np.abs(pitch_levels - f))) if f>0 else 0 for f in rx_p_raw])

    def quantile_decode(raw, n):
        if n <= 1: return np.zeros_like(raw, dtype=int)
        sorted_raw = np.sort(raw)
        thresholds = [sorted_raw[int(len(sorted_raw) * (k+1)/n)] for k in range(n - 1)]
        return np.array([int(np.sum([v > t for t in thresholds])) for v in raw])

    rx_a = quantile_decode(rx_a_raw, n_amp)
    rx_t = quantile_decode(rx_t_raw, n_tilt)

    ep = np.mean(rx_p != sp_tx[:n_dec]) if bps_p else 0.0
    ea = np.mean(rx_a != sa_tx[:n_dec]) if bps_a else 0.0
    et = np.mean(rx_t != st_tx[:n_dec]) if bps_t else 0.0

    def to_bits(v, b): return [(v>>k)&1 for k in range(b-1,-1,-1)]
    bx, by = [], []
    for i in range(n_dec):
        if bps_p: bx += to_bits(int(sp_tx[i]), bps_p); by += to_bits(int(rx_p[i]), bps_p)
        if bps_a: bx += to_bits(int(sa_tx[i]), bps_a); by += to_bits(int(rx_a[i]), bps_a)
        if bps_t: bx += to_bits(int(st_tx[i]), bps_t); by += to_bits(int(rx_t[i]), bps_t)
    ber = float(np.mean(np.array(bx) != np.array(by))) if bx else 1.0
    bitrate = bps * (1000 / symbol_ms)
    return dict(ms=symbol_ms, P=n_pitch, A=n_amp, T=n_tilt, bps=bps, br=bitrate,
                ep=ep, ea=ea, et=et, ber=ber, n=n_dec)

def fmt(r):
    return (f"  sym={r['ms']:>3}ms  P={r['P']:<2} A={r['A']:<2} T={r['T']:<2}  "
            f"{r['bps']}b -> {r['br']:>5.0f}bps   "
            f"e_p={r['ep']*100:>5.1f}% e_a={r['ea']*100:>5.1f}% e_t={r['et']*100:>5.1f}%  "
            f"BER={r['ber']*100:>5.2f}%")

print("=== Sanity: brightness-only @ 40ms ===")
for T in [2, 4]:
    print(fmt(run(40, 1, 1, T)))

print("\n=== 30ms symbols, multi-channel ===")
for cfg in [(8,1,1),(8,2,1),(8,2,2),(8,4,2),(8,4,4),(16,2,2),(16,4,2)]:
    P,A,T = cfg
    print(fmt(run(30, P, A, T)))

print("\n=== 40ms symbols, multi-channel (target: max bps with BER<1%) ===")
for cfg in [(8,2,2),(8,4,2),(16,2,2),(16,4,2),(16,4,4),(8,4,4),(8,8,2),(16,8,2)]:
    P,A,T = cfg
    print(fmt(run(40, P, A, T)))
