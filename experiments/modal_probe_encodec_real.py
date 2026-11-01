"""
Second probe: how stable is the EnCodec encode->Opus->encode round-trip on
*real* speech (not random tokens)? This is the ceiling. If real speech can't
survive token-stable through Opus, we know EnCodec is fundamentally too fragile
and need to fine-tune. If it does survive, the random-token failure is just
"unnatural input" and we have a path forward.

Pipeline:
  real_speech.wav -> EnCodec.encode -> tokens_A
  tokens_A -> EnCodec.decode -> audio_decoded
  audio_decoded -> Opus 24k VOIP -> audio_after_opus
  audio_after_opus -> EnCodec.encode -> tokens_B
  match(tokens_A, tokens_B) per codebook

Also tests the cleaner pipeline (skip the decode stage):
  audio_after_opus = Opus(audio_decoded)
  but we can compare tokens_B to tokens_A directly.
"""
from __future__ import annotations
import modal

APP_NAME = "encodec-real-probe"
GPU = "T4"
TIMEOUT = 30 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "encodec==0.1.1",
        "numpy==2.1.3", "soundfile==0.12.1", "scipy==1.14.1",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT, cpu=4.0, memory=8 * 1024)
def probe(bandwidth_kbps: float = 6.0, n_clips: int = 5,
          opus_kbps: int = 24, opus_app: str = "voip"):
    import subprocess, tempfile, urllib.request
    from pathlib import Path
    import numpy as np
    import soundfile as sf
    import torch
    from encodec import EncodecModel

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SR = 24000
    print(f"[probe] device={DEVICE}  bw={bandwidth_kbps}kbps  opus={opus_kbps}k {opus_app}")

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth_kbps)
    model = model.to(DEVICE).eval()
    n_q = model.quantizer.get_num_quantizers_for_bandwidth(model.frame_rate, bandwidth_kbps)
    print(f"[probe] codebooks={n_q}\n")

    # Use small known speech samples that ship with torchaudio test suite, or
    # download something quick. Use LibriSpeech samples from openslr.
    LIBRI_TARBALL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    cache = Path("/tmp/libri")
    if not cache.exists():
        print("[probe] downloading LibriSpeech dev-clean…")
        cache.mkdir()
        tar = cache/"x.tar.gz"
        urllib.request.urlretrieve(LIBRI_TARBALL, tar)
        subprocess.run(["tar","xzf",str(tar),"-C",str(cache)], check=True)
        tar.unlink()
    flacs = sorted((cache/"LibriSpeech"/"dev-clean").rglob("*.flac"))[:n_clips]
    print(f"[probe] testing {len(flacs)} LibriSpeech clips\n")

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

    print(f"{'clip':>4}  {'duration':>8}  " + "  ".join([f"q{i}" for i in range(n_q)]))
    sums = np.zeros(n_q); count = 0
    for i, p in enumerate(flacs):
        audio, sr = sf.read(p)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, SR, sr)
        audio = audio.astype(np.float32)
        # Take 4 seconds, normalize peak
        audio = audio[:SR*4]
        peak = np.max(np.abs(audio)) + 1e-9
        audio = audio / peak * 0.9

        with torch.no_grad():
            x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)
            tokens_A = model.encode(x)[0][0]  # (1, n_q, n_frames)

        # Path 1: skip decode-then-encode (just check encoder stability through Opus)
        audio_after = opus_rt(audio, SR)

        with torch.no_grad():
            xb = torch.from_numpy(audio_after).unsqueeze(0).unsqueeze(0).to(DEVICE)
            tokens_B = model.encode(xb)[0][0]

        n_frames = min(tokens_A.size(-1), tokens_B.size(-1))
        line_parts = [f"  {i:>2}  {len(audio)/SR:>6.1f}s  "]
        for q in range(n_q):
            m = (tokens_A[0, q, :n_frames] == tokens_B[0, q, :n_frames]).float().mean().item()
            sums[q] += m
            line_parts.append(f"{m*100:5.1f}%")
        print("  ".join(line_parts), flush=True)
        count += 1

    avg = sums / count
    FPS = 75; codebook_bits = 10
    print("\n[probe] === averages ===")
    cum_correct = 0.0
    for q, m in enumerate(avg):
        per_q_bps = FPS * codebook_bits
        correct_bps = per_q_bps * m
        cum_correct += correct_bps
        print(f"  q{q}: {m*100:5.2f}%  ({correct_bps:.0f} bps correct of {per_q_bps:.0f} bps raw)")
    print(f"\n  total correct rate: {cum_correct:.0f} bps")
    print(f"  q0 alone:           {avg[0]*100:.2f}%  ({FPS*codebook_bits*avg[0]:.0f} bps correct)")
    return {"per_codebook_match": avg.tolist(), "n_q": int(n_q),
            "correct_total_bps": float(cum_correct)}


@app.local_entrypoint()
def main():
    print("\n=== bandwidth=6 kbps, Opus 24k VOIP ===")
    probe.remote(bandwidth_kbps=6.0, n_clips=5, opus_kbps=24, opus_app="voip")
    print("\n=== bandwidth=1.5 kbps, Opus 24k VOIP ===")
    probe.remote(bandwidth_kbps=1.5, n_clips=5, opus_kbps=24, opus_app="voip")
    print("\n=== bandwidth=6 kbps, Opus 48k AUDIO ===")
    probe.remote(bandwidth_kbps=6.0, n_clips=5, opus_kbps=48, opus_app="audio")
