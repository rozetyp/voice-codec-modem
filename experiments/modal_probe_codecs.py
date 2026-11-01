"""
Multi-codec probe: which neural audio codec's tokens survive Opus best?

EnCodec hit a 76% q0 ceiling that wouldn't move under encoder fine-tuning. The
hypothesis: it's not that we trained wrong — it's that we picked the wrong base
codec. EnCodec was trained for general-purpose audio. Mimi (Kyutai, Sept 2024)
was explicitly designed for streaming speech + telephony robustness. DAC
(Descript) is the strongest general-purpose neural codec to date.

This probe runs the same protocol as probe β across three codecs:
    real_speech → encode → tokens_A → decode → audio → Opus 48k AUDIO
                                                    → encode → tokens_B
    measure: per-codebook (tokens_A == tokens_B) match rate

The codec whose first codebook(s) survives Opus best is what we build the demo on.
"""
from __future__ import annotations
import modal

APP_NAME = "codec-shootout"
GPU = "A10G"
TIMEOUT = 30 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        "transformers==4.46.3",       # for Mimi (added late 2024)
        "huggingface_hub==0.26.2",
        "encodec==0.1.1",             # control / baseline
        "descript-audio-codec==1.0.0",# DAC
        "numpy==2.1.3", "scipy==1.14.1", "soundfile==0.12.1",
        "tqdm==4.67.1", "requests==2.32.3",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT, cpu=4.0, memory=24*1024)
def probe(n_clips: int = 5, opus_kbps: int = 48, opus_app: str = "audio"):
    import subprocess, tempfile, urllib.request
    from pathlib import Path
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shootout] device={DEVICE}  opus={opus_kbps}k {opus_app}\n")

    # ---------- LibriSpeech ----------
    cache = Path("/tmp/libri")
    if not cache.exists():
        print("[shootout] downloading LibriSpeech dev-clean…")
        cache.mkdir()
        tar = cache/"x.tar.gz"
        urllib.request.urlretrieve(
            "https://www.openslr.org/resources/12/dev-clean.tar.gz", tar)
        subprocess.run(["tar","xzf",str(tar),"-C",str(cache)], check=True)
        tar.unlink()
    flacs = sorted((cache/"LibriSpeech"/"dev-clean").rglob("*.flac"))[:n_clips]
    print(f"[shootout] {len(flacs)} clips")

    def opus_rt(audio_np: np.ndarray, sr: int) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.opus"; out = Path(tmp)/"out.wav"
            sf.write(inp, audio_np.astype(np.float32), sr)
            subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                            "-c:a","libopus","-b:a",f"{opus_kbps}k","-application",opus_app,
                            str(opx)], check=True)
            subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                            "-ar",str(sr),"-ac","1",str(out)], check=True)
            a, _ = sf.read(out)
        if a.ndim > 1: a = a.mean(axis=1)
        if len(a) < len(audio_np): a = np.pad(a, (0, len(audio_np)-len(a)))
        elif len(a) > len(audio_np): a = a[:len(audio_np)]
        return a.astype(np.float32)

    def load_clip(path: Path, target_sr: int, n_seconds: float = 4.0) -> np.ndarray:
        a, sr = sf.read(path)
        if a.ndim > 1: a = a.mean(axis=1)
        if sr != target_sr: a = resample_poly(a, target_sr, sr)
        a = a.astype(np.float32)
        n_target = int(target_sr * n_seconds)
        if len(a) > n_target: a = a[:n_target]
        peak = float(np.max(np.abs(a))) + 1e-9
        return (a / peak * 0.9).astype(np.float32)

    # =========================================================================
    # CODEC 1: EnCodec_24khz @ 6kbps  (control / known baseline ~75% q0)
    # =========================================================================
    def run_encodec():
        from encodec import EncodecModel
        SR = 24000
        m = EncodecModel.encodec_model_24khz()
        m.set_target_bandwidth(6.0)
        m = m.to(DEVICE).eval()
        n_q = m.quantizer.get_num_quantizers_for_bandwidth(m.frame_rate, 6.0)
        rates = np.zeros(n_q)
        for p in flacs:
            audio = load_clip(p, SR)
            x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                tokens_A = m.encode(x)[0][0]                # (1, n_q, T)
                audio_recon = m.decode([(tokens_A, None)])  # (1, 1, N)
            audio_recon_np = audio_recon.squeeze().cpu().numpy().astype(np.float32)
            audio_after = opus_rt(audio_recon_np, SR)
            with torch.no_grad():
                xb = torch.from_numpy(audio_after).unsqueeze(0).unsqueeze(0).to(DEVICE)
                tokens_B = m.encode(xb)[0][0]
            T = min(tokens_A.size(-1), tokens_B.size(-1))
            for q in range(n_q):
                rates[q] += (tokens_A[0, q, :T] == tokens_B[0, q, :T]).float().mean().item()
        rates /= len(flacs)
        return {"name": "EnCodec_24k @ 6kbps", "n_q": int(n_q), "fps": 75,
                "bits_per_q": 10, "rates": rates.tolist()}

    # =========================================================================
    # CODEC 2: Mimi (Kyutai, telephony-robust by design)
    # =========================================================================
    def run_mimi():
        from transformers import MimiModel, AutoFeatureExtractor
        SR = 24000  # Mimi uses 24kHz
        print("[shootout] loading kyutai/mimi…")
        m = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
        fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        # Mimi has 32 codebooks max but typical use is 8 ("semantic" first + 7 acoustic)
        n_q_use = 8
        rates = np.zeros(n_q_use)
        for p in flacs:
            audio = load_clip(p, SR)
            inp = fe(raw_audio=audio, sampling_rate=SR, return_tensors="pt")
            x = inp.input_values.to(DEVICE)
            with torch.no_grad():
                enc_out = m.encode(x, num_quantizers=n_q_use)
                tokens_A = enc_out.audio_codes  # (B=1, n_q, T_frames)
                dec_out = m.decode(tokens_A)
                audio_recon = dec_out.audio_values  # (1, 1, N)
            audio_recon_np = audio_recon.squeeze().cpu().numpy().astype(np.float32)
            audio_after = opus_rt(audio_recon_np, SR)
            inp2 = fe(raw_audio=audio_after, sampling_rate=SR, return_tensors="pt")
            xb = inp2.input_values.to(DEVICE)
            with torch.no_grad():
                tokens_B = m.encode(xb, num_quantizers=n_q_use).audio_codes
            T = min(tokens_A.size(-1), tokens_B.size(-1))
            for q in range(n_q_use):
                rates[q] += (tokens_A[0, q, :T] == tokens_B[0, q, :T]).float().mean().item()
        rates /= len(flacs)
        # Mimi is 12.5 fps × 11 bits each (codebook size 2048)
        return {"name": "Mimi (kyutai/mimi)", "n_q": n_q_use, "fps": 12,
                "bits_per_q": 11, "rates": rates.tolist()}

    # =========================================================================
    # CODEC 3: DAC (Descript) 24kHz
    # =========================================================================
    def run_dac():
        import dac
        SR = 24000
        print("[shootout] loading DAC 24kHz…")
        model_path = dac.utils.download(model_type="24khz")
        m = dac.DAC.load(model_path).to(DEVICE).eval()
        # DAC 24kHz default: 32 codebooks @ 75 fps × 10 bits = 24 kbps. Most "useful"
        # depth depends on bandwidth; check first 8 for fairness with EnCodec.
        n_q_use = 8
        rates = np.zeros(n_q_use)
        for p in flacs:
            audio = load_clip(p, SR)
            x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # DAC's API: encode returns (z, codes, latents, _, _)
                z, codes_A, latents, _, _ = m.encode(x, n_quantizers=n_q_use)
                # codes_A shape: (B, n_q, T)
                audio_recon = m.decode(z)
            audio_recon_np = audio_recon.squeeze().cpu().numpy().astype(np.float32)
            audio_after = opus_rt(audio_recon_np, SR)
            with torch.no_grad():
                xb = torch.from_numpy(audio_after).unsqueeze(0).unsqueeze(0).to(DEVICE)
                _, codes_B, _, _, _ = m.encode(xb, n_quantizers=n_q_use)
            T = min(codes_A.size(-1), codes_B.size(-1))
            for q in range(n_q_use):
                rates[q] += (codes_A[0, q, :T] == codes_B[0, q, :T]).float().mean().item()
        rates /= len(flacs)
        return {"name": "DAC_24k", "n_q": n_q_use, "fps": 75,
                "bits_per_q": 10, "rates": rates.tolist()}

    # ---------- Run all, report ----------
    results = []
    for runner, label in [(run_encodec, "EnCodec"),
                          (run_mimi, "Mimi"),
                          (run_dac, "DAC")]:
        try:
            print(f"\n[shootout] === {label} ===")
            r = runner()
            results.append(r)
            print(f"[shootout] {r['name']}: q0={r['rates'][0]*100:.2f}%  "
                  f"q1={r['rates'][1]*100:.2f}%  q2={r['rates'][2]*100:.2f}%")
        except Exception as e:
            import traceback
            print(f"[shootout] {label} FAILED: {e}")
            traceback.print_exc()
            results.append({"name": label, "error": str(e)})

    # ---------- Summary table ----------
    print("\n\n=== SUMMARY: per-codebook match rate (real-speech tokens through Opus) ===")
    print(f"{'codec':>30}  {'fps':>4} {'bits':>4}  q0     q1     q2     q3     q4     reliable_bps")
    print("-" * 110)
    for r in results:
        if "error" in r:
            print(f"{r['name']:>30}  ERROR: {r['error']}")
            continue
        rs = r["rates"]
        # Reliable bps: assume tokens with >50% match contribute 2*(match-0.5)*bits
        # (channel-capacity heuristic for 1024-ary symbol with substitution noise)
        bits = r["bits_per_q"] * r["fps"]
        reliable = 0.0
        for m in rs:
            if m > 0.5:
                # Simple: usable_fraction = 2*(m - 0.5), only if m > 0.5
                reliable += bits * 2 * (m - 0.5)
        print(f"{r['name']:>30}  {r['fps']:>4} {r['bits_per_q']:>4}  "
              + "  ".join([f"{m*100:5.1f}%" for m in rs[:5]])
              + f"  {reliable:>5.0f}")

    return results


@app.local_entrypoint()
def main(opus_kbps: int = 48, opus_app: str = "audio"):
    print(f"\n=== Multi-codec shootout: Opus {opus_kbps}k {opus_app} ===")
    probe.remote(n_clips=5, opus_kbps=opus_kbps, opus_app=opus_app)
