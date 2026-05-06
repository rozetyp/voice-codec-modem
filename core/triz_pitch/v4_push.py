"""
v4: push the ceiling.
- P=32 pitch (with refined detector)
- Wider RS sweep including big block codes
- 25ms symbols
- Goal: max net bps with post-FEC BER < 0.1%
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v3_rs_fec import Modem, InterleavedRS, measure, opus_codec_rt, voiced

# Override Modem.__init__ would be cleaner; just test directly
print("### A) Pitch resolution sweep (4-channel) @ 30ms ###")
print(f"{'P':>3} {'A':>2} {'T':>2}  {'bps':>5}  {'raw_BER':>9}")
for P in [8, 16, 24, 32]:
    for A, T in [(2,2), (4,2), (2,4)]:
        try:
            m = Modem(symbol_ms=30, n_pitch=P, n_amp=A, n_tilt=T)
        except Exception:
            # P not power of 2 — skip
            continue
        if int(np.log2(P)) != np.log2(P): continue
        raws = []
        for seed in range(3):
            r = measure(m, None, n_data_bytes=400, seed=seed)
            raws.append(r["raw_ber"])
        print(f"  {P:>2}  {A:>1}  {T:>1}  {m.bitrate():>4.0f}bps  {np.mean(raws)*100:>6.2f}%")

print("\n### B) Symbol-duration sweep @ best stack ###")
print(f"{'ms':>3} {'P':>3} {'A':>2} {'T':>2}  {'bps':>5}  {'raw_BER':>9}")
for ms in [40, 35, 30, 25]:
    for P, A, T in [(16,2,2), (16,4,2), (8,2,2)]:
        m = Modem(symbol_ms=ms, n_pitch=P, n_amp=A, n_tilt=T)
        raws = []
        for seed in range(3):
            r = measure(m, None, n_data_bytes=400, seed=seed)
            raws.append(r["raw_ber"])
        print(f"  {ms:>2}  {P:>2}  {A:>1}  {T:>1}  {m.bitrate():>4.0f}bps  {np.mean(raws)*100:>6.2f}%")

print("\n### C) Final sweep: target net bps with FEC, multi-seed ###")
configs = [
    # (cfg, fec_params, label)
    (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (255,191,8), "P16A2T2 + RS(255,191)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (255,159,8), "P16A2T2 + RS(255,159)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (15,9,8),    "P16A2T2 + RS(15,9)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (255,191,8), "P16A4T2 + RS(255,191)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (15,9,8),    "P16A4T2 + RS(15,9)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (15,7,8),    "P16A4T2 + RS(15,7)x8"),
    (dict(symbol_ms=25, n_pitch=16, n_amp=2, n_tilt=2), (255,191,8), "25ms P16A2T2 + RS(255,191)x8"),
    (dict(symbol_ms=25, n_pitch=8,  n_amp=2, n_tilt=2), (255,191,8), "25ms P8A2T2 + RS(255,191)x8"),
    (dict(symbol_ms=40, n_pitch=16, n_amp=2, n_tilt=2), (255,223,8), "40ms P16A2T2 + RS(255,223)x8"),
    (dict(symbol_ms=30, n_pitch=32, n_amp=2, n_tilt=2), (15,7,8),    "P32A2T2 + RS(15,7)x8"),
    (dict(symbol_ms=30, n_pitch=32, n_amp=2, n_tilt=2), (255,159,8), "P32A2T2 + RS(255,159)x8"),
]
print(f"{'label':<40}  {'raw':>7}  {'post':>9}  {'net_bps':>7}")
results = []
for cfg, rs, label in configs:
    raws=[]; posts=[]; effs=[]
    for seed in range(5):
        m = Modem(**cfg)
        n_total, n_data, depth = rs
        f = InterleavedRS(n_data=n_data, n_total=n_total, depth=depth)
        # Use enough data bytes to fill several FEC blocks
        n_bytes = max(n_data * depth * 3, 500)
        r = measure(m, f, n_data_bytes=n_bytes, seed=seed)
        raws.append(r["raw_ber"]); posts.append(r["post_ber"]); effs.append(r["eff_bitrate"])
    line = f"  {label:<40}  {np.mean(raws)*100:>5.2f}%  {np.mean(posts)*100:>7.4f}%  {np.mean(effs):>5.0f}"
    print(line)
    results.append(dict(label=label, raw=np.mean(raws), post=np.mean(posts), net=np.mean(effs)))

# Filter to the winners: post-FEC BER < 0.1%
print("\n### Reliable (post-FEC BER < 0.1%) configs sorted by net bps ###")
reliable = [r for r in results if r["post"] < 0.001]
reliable.sort(key=lambda x: -x["net"])
for r in reliable:
    print(f"  {r['label']:<40}  {r['post']*100:>7.4f}%  {r['net']:>5.0f} bps")
