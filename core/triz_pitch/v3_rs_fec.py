"""
v3: v1 signal model (pitch + amp + tilt) + in-process libopus + bit-interleaved RS FEC.
Goal: sub-0.1% post-FEC BER at >=100 bps net through real Opus 24k VoIP, multi-seed.
"""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import numpy as np, scipy.signal as sp
import opuslib
import reedsolo

SR = 16000

# ---------- Opus round-trip via ffmpeg (matches v1 ground-truth numbers) ----------
import subprocess, soundfile as sf, tempfile
from pathlib import Path
WORK = Path(tempfile.gettempdir()) / "modem_v3"
WORK.mkdir(exist_ok=True)
def opus_codec_rt(audio, bitrate=24000):
    """ffmpeg libopus 24k VoIP round-trip. Slower than opuslib but matches v1 numbers."""
    tag = str(np.random.randint(1, 1<<30))
    inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
    sf.write(inp, audio.astype(np.float32), SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                    "-c:a","libopus","-b:a",f"{bitrate//1000}k",
                    "-application","voip",str(opx)], check=True)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                    "-ar",str(SR),"-ac","1",str(out)], check=True)
    a, _ = sf.read(out)
    inp.unlink(missing_ok=True); opx.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return a.astype(np.float32)

# ---------- v1-style signal: low-passed harmonic stack with optional spectral tilt ----------
def voiced(f0, n, amp=0.6, tilt=0.5):
    """tilt in [0,1]: 0 = darker (faster harmonic rolloff), 1 = brighter (flatter)."""
    t = np.arange(n) / SR
    sig = np.zeros(n)
    rolloff = 1.0 - tilt * 0.5  # 1.0 (1/k) at tilt=0, 0.5 (1/sqrt(k)) at tilt=1
    for k in range(1, 25):
        if k * f0 > SR/2 - 500: break
        sig += np.sin(2*np.pi*k*f0*t) / (k ** rolloff)
    sig /= np.max(np.abs(sig)) + 1e-9
    b, a = sp.butter(2, 3500/(SR/2), btype="low")
    out = sp.filtfilt(b, a, sig)
    # Match v1 scaling: unit RMS * 0.25 base * amp scalar -> peaks stay in safe Opus range
    out = out / (np.sqrt(np.mean(out**2)) + 1e-9) * 0.25 * amp
    ramp = max(8, int(0.005 * SR))
    win = np.ones(n); win[:ramp] = np.linspace(0,1,ramp); win[-ramp:] = np.linspace(1,0,ramp)
    return (out * win).astype(np.float32)

# ---------- detectors (v1-proven) ----------
def detect_pitch(seg):
    s = int(len(seg)*0.15); e = int(len(seg)*0.85)
    seg = seg[s:e]
    if np.std(seg) < 1e-4: return 0.0
    b, a = sp.butter(4, [70/(SR/2), 400/(SR/2)], btype="band")
    seg = sp.filtfilt(b, a, seg)
    seg = seg - seg.mean()
    if np.std(seg) < 1e-5: return 0.0
    corr = np.correlate(seg, seg, mode="full")
    corr = corr[len(corr)//2:]
    min_lag = SR // 400
    max_lag = SR // 70
    if max_lag >= len(corr): return 0.0
    peak = np.argmax(corr[min_lag:max_lag]) + min_lag
    return SR / peak

def detect_amp(seg):
    s = int(len(seg)*0.15); e = int(len(seg)*0.85)
    return float(np.sqrt(np.mean(seg[s:e]**2)))

def detect_brightness(seg):
    s = int(len(seg)*0.15); e = int(len(seg)*0.85)
    seg = seg[s:e]
    if len(seg) < 64: return 0.0
    f, P = sp.welch(seg, fs=SR, nperseg=min(256, len(seg)))
    low = np.sum(P[(f >= 200) & (f < 1000)])
    high = np.sum(P[(f >= 1000) & (f < 3000)])
    return float(high / (low + high + 1e-12))

# ---------- modem ----------
class Modem:
    def __init__(self, symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2):
        self.symbol_ms = symbol_ms
        self.n_samp = SR * symbol_ms // 1000
        self.n_pitch = n_pitch; self.n_amp = n_amp; self.n_tilt = n_tilt
        self.bps_p = int(np.log2(n_pitch)) if n_pitch > 1 else 0
        self.bps_a = int(np.log2(n_amp))   if n_amp   > 1 else 0
        self.bps_t = int(np.log2(n_tilt))  if n_tilt  > 1 else 0
        self.bps = self.bps_p + self.bps_a + self.bps_t
        self.pitch_levels = np.exp(np.linspace(np.log(120), np.log(260), n_pitch)) if n_pitch>1 else np.array([180.0])
        self.amp_levels   = np.linspace(0.30, 0.95, n_amp) if n_amp>1 else np.array([0.7])
        self.tilt_levels  = np.linspace(0.0, 1.0, n_tilt)  if n_tilt>1 else np.array([0.5])

    def encode(self, bits):
        bits = list(bits)
        n_sym = len(bits) // self.bps
        audio = []
        for s in range(n_sym):
            chunk = bits[s*self.bps:(s+1)*self.bps]; off = 0
            ip = int("".join(map(str, chunk[off:off+self.bps_p])), 2) if self.bps_p else 0; off += self.bps_p
            ia = int("".join(map(str, chunk[off:off+self.bps_a])), 2) if self.bps_a else 0; off += self.bps_a
            it = int("".join(map(str, chunk[off:off+self.bps_t])), 2) if self.bps_t else 0
            audio.append(voiced(self.pitch_levels[ip], self.n_samp,
                                amp=self.amp_levels[ia], tilt=self.tilt_levels[it]))
        return np.concatenate(audio).astype(np.float32) if audio else np.zeros(0, dtype=np.float32)

    def decode(self, audio):
        n_sym = len(audio) // self.n_samp
        rx_p_raw, rx_a_raw, rx_t_raw = [], [], []
        for i in range(n_sym):
            seg = audio[i*self.n_samp:(i+1)*self.n_samp]
            rx_p_raw.append(detect_pitch(seg))
            rx_a_raw.append(detect_amp(seg))
            rx_t_raw.append(detect_brightness(seg))
        rx_p_raw = np.array(rx_p_raw); rx_a_raw = np.array(rx_a_raw); rx_t_raw = np.array(rx_t_raw)
        def qd(raw, n):
            if n <= 1: return np.zeros_like(raw, dtype=int)
            srt = np.sort(raw); th = [srt[int(len(srt)*(k+1)/n)] for k in range(n-1)]
            return np.array([int(np.sum([v > t for t in th])) for v in raw])
        rx_p = np.array([int(np.argmin(np.abs(self.pitch_levels - f))) if f>0 else 0 for f in rx_p_raw])
        rx_a = qd(rx_a_raw, self.n_amp)
        rx_t = qd(rx_t_raw, self.n_tilt)
        def to_bits(v, b): return [(v>>k)&1 for k in range(b-1,-1,-1)]
        bits = []
        for i in range(n_sym):
            if self.bps_p: bits += to_bits(int(rx_p[i]), self.bps_p)
            if self.bps_a: bits += to_bits(int(rx_a[i]), self.bps_a)
            if self.bps_t: bits += to_bits(int(rx_t[i]), self.bps_t)
        return bits

    def bitrate(self): return self.bps * 1000 / self.symbol_ms

# ---------- bit-interleaved RS over GF(256) ----------
class InterleavedRS:
    """RS(n_total, n_data) over depth interleaved codewords.
    A burst error of <= depth bytes spreads at most 1 byte error into each codeword,
    each of which can correct (n_total - n_data) // 2 byte errors."""
    def __init__(self, n_data=11, n_total=15, depth=8):
        assert (n_total - n_data) % 2 == 0
        self.n_data = n_data; self.n_total = n_total; self.depth = depth
        self.rs = reedsolo.RSCodec(n_total - n_data, nsize=n_total)
        self.t = (n_total - n_data) // 2  # correctable byte errors per codeword

    def encode(self, data: bytes) -> tuple[bytes, int]:
        block = self.n_data * self.depth
        pad = (-len(data)) % block
        # Random pad, not zeros — quantile-based amplitude/tilt decoder needs balanced
        # symbol distribution in the received signal. Long zero runs bias thresholds.
        if pad:
            data = data + np.random.bytes(pad)
        out = bytearray()
        for i in range(0, len(data), block):
            rows = [self.rs.encode(data[i + d*self.n_data : i + (d+1)*self.n_data])
                    for d in range(self.depth)]
            # Column-major interleave: byte j of each row (j=0..n_total-1)
            for col in range(self.n_total):
                for d in range(self.depth):
                    out.append(rows[d][col])
        return bytes(out), pad

    def decode(self, coded: bytes, pad: int) -> tuple[bytes, int, int]:
        block = self.n_total * self.depth
        out = bytearray()
        n_corr = 0; n_fail = 0
        for i in range(0, len(coded) - block + 1, block):
            seg = coded[i:i+block]
            rows = [bytearray() for _ in range(self.depth)]
            idx = 0
            for col in range(self.n_total):
                for d in range(self.depth):
                    rows[d].append(seg[idx]); idx += 1
            for r in rows:
                try:
                    dec = self.rs.decode(bytes(r))[0]
                    out += dec; n_corr += 1
                except reedsolo.ReedSolomonError:
                    out += bytes(r[:self.n_data]); n_fail += 1
        if pad and len(out) >= pad: out = out[:-pad]
        return bytes(out), n_corr, n_fail

# ---------- helpers ----------
def bytes_to_bits(b): return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).tolist()
def bits_to_bytes(bits):
    bits = list(bits); pad = (-len(bits)) % 8
    bits = bits + [0]*pad
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()

# ---------- experiment runner ----------
def measure(modem: Modem, fec: InterleavedRS | None, n_data_bytes=200, seed=0):
    rng = np.random.default_rng(seed)
    data = bytes(rng.integers(0, 256, n_data_bytes).tolist())
    if fec is not None:
        coded, pad = fec.encode(data)
        tx_bits = bytes_to_bits(coded)
    else:
        tx_bits = bytes_to_bits(data); pad = 0
    pad_bits = (-len(tx_bits)) % modem.bps
    tx_bits_pad = tx_bits + [0]*pad_bits
    audio = modem.encode(tx_bits_pad)
    rx_audio = opus_codec_rt(audio)
    rx_bits = modem.decode(rx_audio)
    L = min(len(tx_bits), len(rx_bits))
    raw_ber = float(np.mean(np.array(tx_bits[:L]) != np.array(rx_bits[:L]))) if L else 1.0
    if fec is None:
        return dict(raw_ber=raw_ber, post_ber=raw_ber,
                    bitrate=modem.bitrate(), eff_bitrate=modem.bitrate(),
                    ncorr=0, nfail=0)
    # Pad rx_bits to expected length (so RS de-interleaver gets full blocks)
    if len(rx_bits) < len(tx_bits):
        rx_bits = rx_bits + [0] * (len(tx_bits) - len(rx_bits))
    rx_bits = rx_bits[:len(tx_bits)]
    rx_bytes = bits_to_bytes(rx_bits)
    decoded, ncorr, nfail = fec.decode(rx_bytes, pad)
    decoded = decoded[:n_data_bytes]
    bx = bytes_to_bits(data); by = bytes_to_bits(decoded)[:len(bx)]
    post_ber = float(np.mean(np.array(bx) != np.array(by))) if by else 1.0
    eff_bitrate = modem.bitrate() * fec.n_data / fec.n_total
    return dict(raw_ber=raw_ber, post_ber=post_ber,
                bitrate=modem.bitrate(), eff_bitrate=eff_bitrate,
                ncorr=ncorr, nfail=nfail)

if __name__ == "__main__":
    # Sanity: in-process Opus + v1 signal + autocorr should match ffmpeg numbers
    print("### Sanity: raw BER through in-process Opus, no FEC ###")
    print(f"{'config':<40}  {'raw_BER':>8}  {'bitrate':>8}")
    for ms, P, A, T in [(30, 16, 2, 2), (30, 16, 4, 2), (40, 8, 2, 2), (30, 8, 2, 1)]:
        m = Modem(symbol_ms=ms, n_pitch=P, n_amp=A, n_tilt=T)
        r = measure(m, None, n_data_bytes=300, seed=0)
        print(f"  ms={ms} P={P} A={A} T={T:<26}  {r['raw_ber']*100:>6.2f}%   {r['bitrate']:>5.0f}bps")

    print("\n### Phase A: RS code-rate sweep on best raw config (30ms P=16 A=2 T=2) ###")
    print(f"{'RS(n,k)':<14} {'depth':<6} {'rate':<6} {'raw_BER':>8} {'post_BER':>9} {'eff_bps':>8}")
    base = dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2)
    rs_specs = [
        (15, 11, 8),  # rate 11/15 ≈ 0.73
        (15,  9, 8),  # rate 9/15 ≈ 0.60
        (15,  7, 8),  # rate 7/15 ≈ 0.47
        (255, 223, 8), # 0.875, big block
        (255, 191, 8), # 0.749
        (255, 159, 8), # 0.624
    ]
    for n_total, n_data, depth in rs_specs:
        raws=[]; posts=[]; effs=[]; nfs=[]
        for seed in range(5):
            m = Modem(**base)
            f = InterleavedRS(n_data=n_data, n_total=n_total, depth=depth)
            r = measure(m, f, n_data_bytes=n_data*depth*4, seed=seed)
            raws.append(r["raw_ber"]); posts.append(r["post_ber"]); effs.append(r["eff_bitrate"]); nfs.append(r["nfail"])
        rate = n_data/n_total
        print(f"  RS({n_total},{n_data})    {depth:<6} {rate:.3f}  {np.mean(raws)*100:>6.2f}%  {np.mean(posts)*100:>7.3f}%  {np.mean(effs):>5.0f} (fail blks={int(np.mean(nfs))})")

    print("\n### Phase B: best (raw_BER,FEC) combinations, multi-seed ###")
    combos = [
        # (modem cfg, RS cfg, label)
        (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (15,11,8), "stable raw + light FEC"),
        (dict(symbol_ms=30, n_pitch=16, n_amp=2, n_tilt=2), (15, 9,8), "stable raw + medium FEC"),
        (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (15,11,8), "high raw + light FEC"),
        (dict(symbol_ms=30, n_pitch=16, n_amp=4, n_tilt=2), (15, 9,8), "high raw + medium FEC"),
        (dict(symbol_ms=40, n_pitch=8,  n_amp=2, n_tilt=2), (15,13,8), "conservative + minimal FEC"),
    ]
    print(f"{'label':<30}  {'raw':>7}  {'post':>8}  {'net_bps':>8}")
    for cfg, rs, label in combos:
        raws=[]; posts=[]; effs=[]
        for seed in range(5):
            m = Modem(**cfg)
            n_total, n_data, depth = rs
            f = InterleavedRS(n_data=n_data, n_total=n_total, depth=depth)
            r = measure(m, f, n_data_bytes=n_data*depth*4, seed=seed)
            raws.append(r["raw_ber"]); posts.append(r["post_ber"]); effs.append(r["eff_bitrate"])
        print(f"  {label:<30}  {np.mean(raws)*100:>5.2f}%  {np.mean(posts)*100:>6.3f}%  {np.mean(effs):>5.0f}")
