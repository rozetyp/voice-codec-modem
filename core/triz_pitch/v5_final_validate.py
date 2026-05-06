"""
Final extended-seed validation of the winning configuration.
Long messages (10x more than v4) and 10 seeds to stress-test the 0% claim.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v3_rs_fec import Modem, InterleavedRS, measure

# Top candidates
configs = [
    (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (255,191,8), "WINNER 30ms P16A4T2 + RS(255,191)x8"),
    (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (255,191,8), "30ms P16A2T2 + RS(255,191)x8"),
    (dict(symbol_ms=25, n_pitch=8,  n_amp=2, n_tilt=2), (255,191,8), "25ms P8A2T2 + RS(255,191)x8"),
]

n_seeds = 10
n_data_bytes = 1900  # ~5 full RS(255,191) blocks of 191 bytes/codeword * 8 depth = 1528... use 1900 for ~7 blocks

print(f"### Extended validation: {n_seeds} seeds, {n_data_bytes} data bytes per run ###")
print(f"{'config':<40}  {'raw_BER':>8}  {'post_BER':>9}  {'fail_blks':>9}  {'net_bps':>7}")
for cfg, rs, label in configs:
    raws = []; posts = []; fails = []; effs = []
    total_bytes_tested = 0
    total_bit_errors = 0
    for seed in range(n_seeds):
        m = Modem(**cfg)
        n_total, n_data, depth = rs
        f = InterleavedRS(n_data=n_data, n_total=n_total, depth=depth)
        r = measure(m, f, n_data_bytes=n_data_bytes, seed=seed)
        raws.append(r["raw_ber"]); posts.append(r["post_ber"])
        fails.append(r["nfail"]); effs.append(r["eff_bitrate"])
        total_bytes_tested += n_data_bytes
        total_bit_errors += int(round(r["post_ber"] * n_data_bytes * 8))
    print(f"  {label:<40}  {np.mean(raws)*100:>5.2f}%  {np.mean(posts)*100:>7.4f}%  "
          f"{int(np.sum(fails)):>7}    {np.mean(effs):>5.0f}")
    print(f"      total: {total_bytes_tested*8} bits transmitted, {total_bit_errors} bit errors observed")
