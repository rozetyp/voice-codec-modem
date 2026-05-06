"""
Probe α: voice-within-voice via two-talker mixing + neural source separation.

Hypothesis: the "voice within voice" dream isn't watermark-class steganography —
it's two speakers mixed together, sent through Opus, then split apart on the
receiver with a modern source separator. The cover party hears the dominant
talker (cocktail-party effect ignores the quieter one); the decoder party runs
Sepformer and recovers the second voice as text.

Pipeline:
  cover_speech (0 dB) + data_speech * mix_ratio  ->  Opus 24k VOIP
                                                  ->  Sepformer
                                                  ->  Whisper.transcribe
                                                  ->  WER vs ground-truth

If WER on the data speaker stays < 30% at mix_ratio = -10 dB, the dream is real
and the rate is "speech rate" (~150 wpm = effectively unbounded). If WER blows
up by -10 dB, two-talker is too fragile and we fall back.
"""
from __future__ import annotations
import modal

APP_NAME = "two-talker-probe"
GPU = "A10G"  # Sepformer + Whisper want a real GPU
TIMEOUT = 30 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        "speechbrain==1.0.2",
        "faster-whisper==1.0.3",
        "soundfile==0.12.1", "scipy==1.14.1", "numpy==2.1.3",
        "jiwer==3.0.5",
        "requests==2.32.3", "huggingface_hub==0.26.2",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT, cpu=4.0, memory=16 * 1024)
def probe(n_pairs: int = 4, opus_kbps: int = 24, opus_app: str = "voip"):
    import subprocess, tempfile, urllib.request, itertools
    from pathlib import Path
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from jiwer import wer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SR = 16000  # Sepformer wsj02mix is 8 kHz, but we'll feed 16k and resample
    SEP_SR = 8000
    print(f"[α] device={DEVICE}  opus={opus_kbps}k {opus_app}")

    # 1. Fetch LibriSpeech dev-clean (small subset)
    cache = Path("/tmp/libri")
    if not cache.exists():
        print("[α] downloading LibriSpeech dev-clean…")
        cache.mkdir()
        tar = cache/"x.tar.gz"
        urllib.request.urlretrieve(
            "https://www.openslr.org/resources/12/dev-clean.tar.gz", tar)
        subprocess.run(["tar","xzf",str(tar),"-C",str(cache)], check=True)
        tar.unlink()

    # Pick utterances from DIFFERENT speakers
    spk_dirs = sorted([p for p in (cache/"LibriSpeech"/"dev-clean").iterdir() if p.is_dir()])
    print(f"[α] {len(spk_dirs)} speakers available")

    def load_clip(spk_dir, target_sec=6.0):
        flacs = sorted(spk_dir.rglob("*.flac"))
        for f in flacs:
            a, sr = sf.read(f)
            if a.ndim > 1: a = a.mean(axis=1)
            if sr != SR: a = resample_poly(a, SR, sr)
            if len(a) >= int(SR*target_sec):
                a = a[:int(SR*target_sec)].astype(np.float32)
                # transcript is in <spk>-<chap>.trans.txt under the chapter dir
                trans_file = next(f.parent.glob("*.trans.txt"))
                key = f.stem
                txt = ""
                for line in open(trans_file):
                    if line.startswith(key+" "):
                        txt = line[len(key)+1:].strip()
                        break
                return a, txt
        return None, None

    pairs = []
    used = set()
    for s in spk_dirs:
        if len(pairs) >= n_pairs * 2: break
        a, t = load_clip(s)
        if a is None: continue
        used.add(s.name)
        pairs.append((a, t, s.name))
    print(f"[α] loaded {len(pairs)} clips from {len(used)} speakers")

    # 2. Load Sepformer + Whisper
    print("[α] loading Sepformer (sepformer-wsj02mix)…")
    from speechbrain.inference.separation import SepformerSeparation
    sep = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="/tmp/sep",
        run_opts={"device": DEVICE},
    )
    print("[α] loading Whisper (small.en) via faster-whisper…")
    from faster_whisper import WhisperModel
    asr = WhisperModel("small.en", device=DEVICE,
                       compute_type="float16" if DEVICE=="cuda" else "int8")

    def opus_rt(audio_np, sr):
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

    def transcribe(a16k):
        # faster-whisper expects float32 16kHz mono numpy array
        segments, _ = asr.transcribe(a16k.astype(np.float32), beam_size=1, language="en")
        return " ".join(s.text.strip() for s in segments).strip()

    def to_wer(ref, hyp):
        if not ref.strip() or not hyp.strip(): return 1.0
        try: return float(wer(ref.lower(), hyp.lower()))
        except Exception: return 1.0

    # 3. Run grid
    results = []
    ratios_db = [-3, -6, -10, -15]
    print(f"\n{'pair':>4}  {'ratio_dB':>8}  {'WER_cover':>9}  {'WER_data':>9}  ground_truth_data")
    print("-" * 100)

    pair_idx = 0
    for i in range(0, len(pairs)-1, 2):
        cover_a, cover_t, cover_id = pairs[i]
        data_a, data_t, data_id = pairs[i+1]
        if cover_id == data_id: continue
        if pair_idx >= n_pairs: break
        pair_idx += 1

        # Normalize each to peak 0.7
        cover_a = cover_a / (np.max(np.abs(cover_a))+1e-9) * 0.7
        data_a  = data_a  / (np.max(np.abs(data_a))+1e-9) * 0.7

        for r_db in ratios_db:
            scale = 10 ** (r_db / 20.0)
            mix = cover_a + scale * data_a
            # prevent clipping
            peak = np.max(np.abs(mix))
            if peak > 0.99: mix = mix / peak * 0.95

            # Opus round-trip at 16 kHz (Opus internally resamples to 24k VOIP)
            mix_after = opus_rt(mix, SR)

            # Sepformer wants 8 kHz, so resample to 8k for separation
            mix_8k = resample_poly(mix_after, SEP_SR, SR).astype(np.float32)
            with torch.no_grad():
                est = sep.separate_batch(torch.from_numpy(mix_8k).unsqueeze(0).to(DEVICE))
            # est: (1, T, 2)
            est = est[0].cpu().numpy()  # (T, 2)
            s1 = resample_poly(est[:, 0], SR, SEP_SR).astype(np.float32)
            s2 = resample_poly(est[:, 1], SR, SEP_SR).astype(np.float32)
            # pad/trim to original length
            for s in (s1, s2):
                if len(s) < len(cover_a):
                    pass  # don't mutate
            def fit(s):
                if len(s) < len(cover_a):
                    return np.pad(s, (0, len(cover_a)-len(s)))
                return s[:len(cover_a)]
            s1, s2 = fit(s1), fit(s2)

            # Decide which separated stream is cover vs data via correlation
            # to the (post-Opus) cover and data references
            cover_ref = opus_rt(cover_a, SR)  # what cover sounds like after Opus alone
            def corr(x, y):
                x = x - x.mean(); y = y - y.mean()
                d = np.linalg.norm(x)*np.linalg.norm(y)+1e-9
                return float(np.dot(x, y) / d)
            # If s1 correlates with cover_ref more than s2, then s1=cover, s2=data
            if corr(s1, cover_ref) > corr(s2, cover_ref):
                rec_cover, rec_data = s1, s2
            else:
                rec_cover, rec_data = s2, s1

            # Transcribe
            hyp_cover = transcribe(rec_cover)
            hyp_data  = transcribe(rec_data)
            w_cover = to_wer(cover_t, hyp_cover)
            w_data  = to_wer(data_t,  hyp_data)

            print(f"  {pair_idx:>2}  {r_db:>7}dB  {w_cover*100:>7.1f}%  {w_data*100:>7.1f}%  "
                  f"DATA: {data_t[:60]}", flush=True)
            print(f"      cover hyp: {hyp_cover[:80]}")
            print(f"      data  hyp: {hyp_data[:80]}", flush=True)
            results.append({
                "pair": pair_idx, "ratio_db": r_db,
                "wer_cover": w_cover, "wer_data": w_data,
                "data_truth": data_t, "data_hyp": hyp_data,
            })

    # 4. Aggregate
    print("\n[α] === averages by ratio ===")
    print(f"{'ratio_dB':>8}  {'WER_cover':>10}  {'WER_data':>10}  n")
    by_r = {}
    for r in results:
        by_r.setdefault(r["ratio_db"], []).append(r)
    for r_db in sorted(by_r.keys(), reverse=True):
        rows = by_r[r_db]
        wc = np.mean([x["wer_cover"] for x in rows])
        wd = np.mean([x["wer_data"]  for x in rows])
        print(f"  {r_db:>5}dB  {wc*100:>8.1f}%  {wd*100:>8.1f}%  {len(rows)}")

    return results


@app.local_entrypoint()
def main():
    print("\n=== two-talker through Opus 24k VOIP + Sepformer + Whisper ===")
    probe.remote(n_pairs=4, opus_kbps=24, opus_app="voip")
