"""
Validate the best config across multiple seeds and add Hamming(7,4) FEC.
Best: 30ms symbols, P=16 A=4 T=2 -> 7 bits/sym -> 233 bps raw.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triple_channel import run

# Multi-seed validation of two top configs
print("=== Multi-seed validation ===")
configs = [
    (30, 16, 4, 2, "best raw"),
    (40,  8, 2, 2, "conservative"),
    (30,  8, 2, 1, "pitch+amp only"),
    (30, 16, 2, 2, "wide pitch"),
]
for ms, P, A, T, label in configs:
    bers = []
    eps = []; eas = []; ets = []
    for seed in range(5):
        r = run(ms, P, A, T, n_symbols=300, seed=seed)
        bers.append(r["ber"]); eps.append(r["ep"]); eas.append(r["ea"]); ets.append(r["et"])
    bers = np.array(bers);
    bps = int(np.log2(P)) + (int(np.log2(A)) if A>1 else 0) + (int(np.log2(T)) if T>1 else 0)
    br = bps * 1000 / ms
    print(f"  {label:20s} sym={ms}ms P={P} A={A} T={T}: {br:.0f} bps  "
          f"BER mean={bers.mean()*100:.2f}% std={bers.std()*100:.2f}% "
          f"(e_p={np.mean(eps)*100:.1f}% e_a={np.mean(eas)*100:.1f}% e_t={np.mean(ets)*100:.1f}%)")

# Hamming(7,4): 4 data bits -> 7 code bits, corrects 1 bit error per block
def hamming_encode_74(bits):
    bits = np.array(bits, dtype=int)
    # pad to multiple of 4
    pad = (-len(bits)) % 4
    bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    H_G = np.array([
        [1,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,1],
        [0,1,0,0],[0,0,1,0],[0,0,0,1]
    ])  # 7x4 generator
    out = []
    for i in range(0, len(bits), 4):
        d = bits[i:i+4]
        c = (H_G @ d) % 2
        out.extend(c.tolist())
    return out

def hamming_decode_74(bits):
    bits = np.array(bits, dtype=int)
    H_check = np.array([
        [1,0,1,0,1,0,1],
        [0,1,1,0,0,1,1],
        [0,0,0,1,1,1,1]
    ])
    out = []
    n_corr = 0
    for i in range(0, len(bits)-6, 7):
        c = bits[i:i+7].copy()
        s = (H_check @ c) % 2
        idx = int(s[0] + 2*s[1] + 4*s[2])  # syndrome -> 1-based bit position
        if idx > 0:
            c[idx-1] ^= 1
            n_corr += 1
        # Data bits in positions 3,5,6,7 (0-indexed: 2,4,5,6)
        out.extend([int(c[2]), int(c[4]), int(c[5]), int(c[6])])
    return out, n_corr

# Apply FEC to the best raw config and remeasure end-to-end
print("\n=== With Hamming(7,4) FEC on the best raw channel (30ms P=16 A=4 T=2) ===")
print("Code rate 4/7; effective bitrate = 233 * 4/7 ≈ 133 bps")
# We'll simulate FEC by decoding raw bits, applying Hamming, comparing to original data
# Since we can't easily get the raw bit sequence out of run(), do a custom roundtrip here:
import scipy.signal as sp, soundfile as sf, subprocess
from triple_channel import make_symbol, detect_pitch, detect_amp, detect_brightness, opus_rt, SR
from pathlib import Path

def custom_run_with_fec(ms, P, A, T, n_data_bits=600, seed=0):
    n_samp = SR * ms // 1000
    rng = np.random.default_rng(seed)
    bps_p, bps_a, bps_t = int(np.log2(P)), int(np.log2(A)) if A>1 else 0, int(np.log2(T)) if T>1 else 0
    bps = bps_p + bps_a + bps_t
    pitch_levels = np.exp(np.linspace(np.log(120), np.log(260), P))
    amp_levels = np.linspace(0.30, 0.95, A) if A>1 else np.array([0.7])
    tilt_levels = np.linspace(0.0, 1.0, T) if T>1 else np.array([0.5])

    data_bits = rng.integers(0, 2, n_data_bits).tolist()
    coded_bits = hamming_encode_74(data_bits)

    audio = []
    for i in range(0, len(coded_bits) - bps + 1, bps):
        chunk = coded_bits[i:i+bps]; off = 0
        ip = int("".join(map(str, chunk[off:off+bps_p])), 2); off += bps_p
        ia = int("".join(map(str, chunk[off:off+bps_a])), 2) if bps_a else 0; off += bps_a
        it = int("".join(map(str, chunk[off:off+bps_t])), 2) if bps_t else 0
        audio.append(make_symbol(pitch_levels[ip], amp_levels[ia], tilt_levels[it], n_samp))
    audio = np.concatenate(audio).astype(np.float32)
    rx = opus_rt(audio, f"fec_{ms}_{P}_{A}_{T}_s{seed}")

    n_dec = len(coded_bits) // bps
    n_dec = min(n_dec, len(rx) // n_samp)
    rx_p_raw, rx_a_raw, rx_t_raw = [], [], []
    for i in range(n_dec):
        seg = rx[i*n_samp:(i+1)*n_samp]
        rx_p_raw.append(detect_pitch(seg))
        rx_a_raw.append(detect_amp(seg))
        rx_t_raw.append(detect_brightness(seg))
    rx_p_raw = np.array(rx_p_raw); rx_a_raw = np.array(rx_a_raw); rx_t_raw = np.array(rx_t_raw)
    rx_p = np.array([int(np.argmin(np.abs(pitch_levels - f))) if f>0 else 0 for f in rx_p_raw])

    def qd(raw, n):
        if n <= 1: return np.zeros_like(raw, dtype=int)
        s = np.sort(raw); th = [s[int(len(s)*(k+1)/n)] for k in range(n-1)]
        return np.array([int(np.sum([v > t for t in th])) for v in raw])
    rx_a = qd(rx_a_raw, A); rx_t = qd(rx_t_raw, T)

    def to_bits(v, b): return [(v>>k)&1 for k in range(b-1,-1,-1)]
    rx_coded = []
    for i in range(n_dec):
        if bps_p: rx_coded += to_bits(int(rx_p[i]), bps_p)
        if bps_a: rx_coded += to_bits(int(rx_a[i]), bps_a)
        if bps_t: rx_coded += to_bits(int(rx_t[i]), bps_t)
    rx_coded = rx_coded[:len(coded_bits)]

    raw_ber = float(np.mean(np.array(rx_coded) != np.array(coded_bits[:len(rx_coded)])))

    decoded, n_corr = hamming_decode_74(rx_coded)
    decoded = decoded[:len(data_bits)]
    post_ber = float(np.mean(np.array(decoded) != np.array(data_bits[:len(decoded)]))) if decoded else 1.0
    return raw_ber, post_ber, n_corr

raws, posts = [], []
for seed in range(5):
    rb, pb, nc = custom_run_with_fec(30, 16, 4, 2, seed=seed)
    raws.append(rb); posts.append(pb)
    print(f"  seed {seed}: raw BER {rb*100:.2f}%  post-FEC BER {pb*100:.3f}%  ({nc} blocks corrected)")
print(f"  -> raw BER avg {np.mean(raws)*100:.2f}%   post-FEC BER avg {np.mean(posts)*100:.3f}%")
print(f"  -> effective bitrate: {233 * 4/7:.0f} bps  (Hamming 4/7 rate)")
