"""
Probe: how well do EnCodec tokens survive a round-trip through Opus 24k VOIP?

  random_tokens  ->  EnCodec.decode  ->  speech-distribution audio
                                              |
                                          libopus 24k VOIP
                                              |
                                        recovered audio
                                              |
                                       EnCodec.encode  ->  recovered_tokens
                                              |
              compare to original: % indices that match per codebook

If the bottom-most codebook (the one that carries the broadest spectral structure)
recovers >90% accurately, that's our channel. ~75 frames/sec * 10 bits/frame =
750 bps native, before any FEC. Higher codebooks are more fragile but stack the
rate up to 6 kbps if any of them survive.

This is a single round-trip test on Modal — no training. Tells us whether the
EnCodec-as-channel approach is viable at all before we invest in fine-tuning.
"""
from __future__ import annotations
import modal

APP_NAME = "encodec-probe"
GPU = "T4"  # cheaper, plenty for inference-only round-trip
TIMEOUT = 30 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "encodec==0.1.1",
        "numpy==2.1.3",
        "soundfile==0.12.1",
        "scipy==1.14.1",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT, cpu=4.0, memory=8 * 1024)
def probe(
    n_seconds: float = 4.0,
    bandwidth_kbps: float = 6.0,   # 1.5 / 3.0 / 6.0 / 12.0 / 24.0 are valid
    n_seeds: int = 4,
    opus_kbps: int = 24,
    opus_app: str = "voip",
):
    import subprocess, tempfile
    from pathlib import Path
    import numpy as np
    import soundfile as sf
    import torch
    from encodec import EncodecModel

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={DEVICE}  bandwidth={bandwidth_kbps} kbps  opus={opus_kbps}k {opus_app}")

    # 24 kHz model: codebooks at the chosen bandwidth.
    # bandwidth_kbps -> n_q codebooks: 1.5->2, 3->4, 6->8, 12->16, 24->32 (each codebook adds 0.75 kbps)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth_kbps)
    model = model.to(DEVICE).eval()

    # Frame rate is fixed at 75 Hz for the 24 kHz EnCodec model.
    SR = 24000
    FPS = 75
    n_q = model.quantizer.get_num_quantizers_for_bandwidth(model.frame_rate, bandwidth_kbps)
    n_frames = int(n_seconds * FPS)
    codebook_size = model.quantizer.bins  # 1024
    print(f"[probe] codebooks={n_q}  frames={n_frames} ({n_frames/FPS:.1f}s)  "
          f"codebook_size={codebook_size} ({np.log2(codebook_size):.0f} bits each)")
    print(f"[probe] raw token-channel rate = {n_q * FPS * np.log2(codebook_size):.0f} bps")
    print()

    def opus_round_trip(audio_np: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp)/"in.wav"; opx = Path(tmp)/"x.opus"; out = Path(tmp)/"out.wav"
            sf.write(inp, audio_np.astype(np.float32), SR)
            subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                            "-c:a","libopus","-b:a",f"{opus_kbps}k","-application",opus_app,
                            str(opx)], check=True)
            subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                            "-ar",str(SR),"-ac","1",str(out)], check=True)
            a, _ = sf.read(out)
        if a.ndim > 1: a = a.mean(axis=1)
        if len(a) < len(audio_np): a = np.pad(a, (0, len(audio_np) - len(a)))
        elif len(a) > len(audio_np): a = a[:len(audio_np)]
        return a.astype(np.float32)

    print(f"{'seed':>4}  " + "  ".join([f"q{i}_match" for i in range(n_q)]))

    per_codebook_match = np.zeros(n_q)
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)

        # Random tokens
        tokens_tx = torch.randint(0, codebook_size, (1, n_q, n_frames), device=DEVICE)
        # Decode to audio. EnCodec expects (frames=[(codes, scale)]) format.
        with torch.no_grad():
            audio_tx = model.decoder(model.quantizer.decode(tokens_tx.transpose(0, 1)))  # (1, 1, samples)
        audio_tx_np = audio_tx.squeeze().cpu().numpy().astype(np.float32)
        # Normalize to safe peak before Opus
        peak = np.max(np.abs(audio_tx_np)) + 1e-9
        audio_tx_np = audio_tx_np / peak * 0.9

        # Round-trip through real Opus
        audio_rx_np = opus_round_trip(audio_tx_np)

        # Re-encode through EnCodec
        with torch.no_grad():
            audio_rx_t = torch.from_numpy(audio_rx_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            encoded = model.encode(audio_rx_t)
            tokens_rx = encoded[0][0]  # (1, n_q, n_frames)

        # Compare per-codebook match rate
        match_per_q = []
        for q in range(n_q):
            m = (tokens_tx[0, q] == tokens_rx[0, q]).float().mean().item()
            match_per_q.append(m)
            per_codebook_match[q] += m

        line = f"  {seed:>2}    " + "  ".join([f"{m*100:6.2f}%" for m in match_per_q])
        print(line, flush=True)

    print("\n[probe] === averages ===")
    avg = per_codebook_match / n_seeds
    print(f"  per-codebook match rate (avg across {n_seeds} seeds):")
    cumulative_bps = 0.0
    cumulative_correct_bps = 0.0
    for q, m in enumerate(avg):
        per_q_bps = FPS * np.log2(codebook_size)
        correct_bps = per_q_bps * m
        cumulative_bps += per_q_bps
        cumulative_correct_bps += correct_bps
        print(f"    q{q}: {m*100:5.2f}%  ({per_q_bps:.0f} bps raw, {correct_bps:.0f} bps correct)")
    print(f"\n  total raw rate:     {cumulative_bps:.0f} bps")
    print(f"  total correct rate: {cumulative_correct_bps:.0f} bps")
    print(f"  combined index match: {avg.mean()*100:.2f}%")

    return {
        "per_codebook_match": avg.tolist(),
        "n_q": int(n_q),
        "frame_rate": int(FPS),
        "codebook_size": int(codebook_size),
        "bandwidth_kbps": float(bandwidth_kbps),
        "opus_kbps": int(opus_kbps),
        "raw_total_bps": float(cumulative_bps),
        "correct_total_bps": float(cumulative_correct_bps),
    }


@app.local_entrypoint()
def main(bandwidth: float = 6.0, n_seeds: int = 4, opus_kbps: int = 24):
    result = probe.remote(bandwidth_kbps=bandwidth, n_seeds=n_seeds, opus_kbps=opus_kbps)
    print(f"\n[local] result: {result}")
