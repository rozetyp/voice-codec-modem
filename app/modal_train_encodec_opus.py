"""
Modal training: fine-tune EnCodec encoder for Opus survival.

Probe β showed pretrained EnCodec_24khz preserves q0 = 75% of real-speech tokens
through Opus 48k AUDIO and 50% through Opus 24k VOIP. The model wasn't trained
with Opus in the loop — that's the gap. Goal here: push q0 → 95% by training
ONLY the encoder, with the decoder + RVQ codebook frozen.

Why this works (from probe β, the pattern):
  Opus is a speech codec. Real-speech-distributed audio survives 4–5x better
  than random tokens through it. The vanilla EnCodec encoder maps audio → 128-D
  embeddings; those embeddings drift through Opus, sometimes into a different
  Voronoi cell of q0's codebook (=> token mismatch). If we re-train the encoder
  to produce embeddings that DON'T drift through Opus, q0 round-trips reliably.

Pipeline per training step:
    audio  → encoder (trainable)         → z_clean
    z_clean → quantizer (frozen)          → tokens t (n_q, B, T_frames)
    t      → decoder (frozen)             → audio_recon
    audio_recon → Opus 48k AUDIO          → audio_after  (straight-through identity in backward)
    audio_after → encoder (trainable)     → z_noisy

  L_consistency = MSE(z_noisy, z_clean.detach())
                  pulls z_noisy toward z_clean → same Voronoi cell → same token
  L_recon       = STFT_L1(audio_recon, audio_clean)
                  prevents the encoder collapsing to "easy-to-round-trip but
                  meaningless" embeddings — they still must reconstruct.

The encoder is trainable; the RVQ codebook and decoder are frozen so the token
DICTIONARY is preserved. This means a non-fine-tuned EnCodec_24khz on the
receiver still decodes our audio normally — the channel is interoperable.

Run:
  modal run app/modal_train_encodec_opus.py
  modal volume get stego-codec-vol /checkpoints/encodec_opus_best.pt ./
"""
from __future__ import annotations
import modal

APP_NAME = "encodec-opus-ft"
VOLUME_NAME = "stego-codec-vol"
GPU = "A10G"
TIMEOUT_SECONDS = 4 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "encodec==0.1.1",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "soundfile==0.12.1",
        "tqdm==4.67.1",
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU, volumes={"/vol": vol},
    timeout=TIMEOUT_SECONDS, cpu=4.0, memory=32 * 1024,
)
def train(
    n_steps: int = 3000,
    batch_size: int = 8,
    clip_seconds: float = 2.0,
    bandwidth_kbps: float = 6.0,        # 8 codebooks
    opus_kbps: int = 48,
    opus_app: str = "audio",            # 48k AUDIO mode (Discord/music) — best-case
    lr: float = 3e-5,
    q0_ce_weight: float = 1.0,          # direct Voronoi-cell CE on q0 (the stuck codebook)
    consistency_weight: float = 0.5,    # MSE on embeddings (helps q1+ implicitly)
    recon_weight: float = 1.0,          # keep encoder anchored to "produces meaningful tokens"
    softmax_temperature: float = 1.0,
    eval_every: int = 200,
    eval_clips: int = 8,
    seed: int = 0,
):
    import math, os, random, subprocess, tempfile, time
    from pathlib import Path
    import numpy as np
    import scipy.signal as sps
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import tqdm
    from encodec import EncodecModel

    SR = 24000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ft] device={DEVICE}  torch={torch.__version__}")
    if torch.cuda.is_available():
        print(f"[ft] gpu={torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # ---------- LibriSpeech (cached on volume) ----------
    LIBRI_ROOT = Path("/vol/librispeech")
    LIBRI_SUBSET = LIBRI_ROOT / "LibriSpeech" / "dev-clean"
    if not LIBRI_SUBSET.exists():
        LIBRI_ROOT.mkdir(parents=True, exist_ok=True)
        tarball = LIBRI_ROOT / "dev-clean.tar.gz"
        if not tarball.exists():
            print("[ft] downloading LibriSpeech dev-clean…")
            subprocess.run(["wget", "-q", "-O", str(tarball),
                            "https://www.openslr.org/resources/12/dev-clean.tar.gz"],
                           check=True)
        print("[ft] extracting…")
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(LIBRI_ROOT)], check=True)
        tarball.unlink(missing_ok=True)
        vol.commit()

    # Pre-cache filenames; load on the fly
    flacs = sorted(LIBRI_SUBSET.rglob("*.flac"))
    print(f"[ft] {len(flacs)} flacs available")
    CLIP_N = int(SR * clip_seconds)

    def load_clip(path: Path) -> np.ndarray | None:
        a, sr = sf.read(path)
        if a.ndim > 1: a = a.mean(axis=1)
        if sr != SR:
            a = sps.resample_poly(a, SR, sr)
        a = a.astype(np.float32)
        if len(a) < CLIP_N: return None
        # Random offset
        off = np.random.randint(0, len(a) - CLIP_N + 1)
        clip = a[off:off+CLIP_N]
        # Normalize peak
        p = float(np.max(np.abs(clip)))
        if p < 1e-3: return None
        return (clip / p * 0.9).astype(np.float32)

    def sample_batch(b: int) -> torch.Tensor:
        out = []
        while len(out) < b:
            p = flacs[np.random.randint(0, len(flacs))]
            c = load_clip(p)
            if c is not None: out.append(c)
        x = np.stack(out, axis=0)
        return torch.from_numpy(x).unsqueeze(1).to(DEVICE)  # (B, 1, N)

    # Held-out eval set: fixed clips for stable comparison across steps
    rng_eval = np.random.RandomState(12345)
    eval_paths = [flacs[i] for i in rng_eval.choice(len(flacs), size=eval_clips, replace=False)]
    eval_clips_arr = []
    for p in eval_paths:
        c = load_clip(p)
        if c is not None: eval_clips_arr.append(c)
    eval_x = torch.from_numpy(np.stack(eval_clips_arr, axis=0)).unsqueeze(1).to(DEVICE)
    print(f"[ft] eval set: {eval_x.size(0)} clips at {clip_seconds}s")

    # ---------- EnCodec ----------
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth_kbps)
    model = model.to(DEVICE)
    n_q = model.quantizer.get_num_quantizers_for_bandwidth(model.frame_rate, bandwidth_kbps)
    print(f"[ft] EnCodec 24khz @ {bandwidth_kbps} kbps  → n_q={n_q}  fps={model.frame_rate}")

    # Freeze decoder + quantizer; only encoder trainable
    for p in model.decoder.parameters(): p.requires_grad = False
    for p in model.quantizer.parameters(): p.requires_grad = False
    for p in model.encoder.parameters(): p.requires_grad = True

    n_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"[ft] trainable encoder params: {n_train/1e6:.1f}M")

    # Frozen sub-modules in eval mode (no BN/dropout drift); encoder in train mode
    model.decoder.eval()
    model.quantizer.eval()
    model.encoder.train()

    # Cache q0 codebook for direct CE loss. Walk the encodec layout to find it
    # robustly across encodec versions: ResidualVectorQuantizer.vq.layers[0]._codebook.embed
    rvq = model.quantizer.vq if hasattr(model.quantizer, "vq") else model.quantizer
    layer0 = rvq.layers[0]
    cb0 = layer0._codebook if hasattr(layer0, "_codebook") else layer0.codebook
    codebook_q0 = cb0.embed  # (codebook_size=1024, D=128). Frozen by quantizer.eval().
    print(f"[ft] q0 codebook: {tuple(codebook_q0.shape)}")

    opt = torch.optim.AdamW(model.encoder.parameters(), lr=lr, betas=(0.9, 0.999))

    # ---------- Opus straight-through ----------
    WORK = Path(tempfile.gettempdir()) / "opus_rt"; WORK.mkdir(exist_ok=True)
    def opus_batch(audio_t: torch.Tensor) -> torch.Tensor:
        """audio_t: (B, 1, N). Returns (B, 1, N) after Opus round-trip."""
        a_np = audio_t.detach().cpu().numpy().squeeze(1)  # (B, N)
        B, N = a_np.shape
        flat = a_np.reshape(-1)
        tag = str(np.random.randint(1, 1<<30))
        inp = WORK/f"in_{tag}.wav"; opx = WORK/f"x_{tag}.opus"; out = WORK/f"out_{tag}.wav"
        sf.write(inp, flat.astype(np.float32), SR)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(inp),
                        "-c:a","libopus","-b:a",f"{opus_kbps}k","-application",opus_app,
                        str(opx)], check=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(opx),
                        "-ar",str(SR),"-ac","1", str(out)], check=True)
        a, _ = sf.read(out)
        for f in (inp, opx, out): f.unlink(missing_ok=True)
        if a.ndim > 1: a = a.mean(axis=1)
        if len(a) < B*N: a = np.pad(a, (0, B*N - len(a)))
        elif len(a) > B*N: a = a[:B*N]
        return torch.from_numpy(a.reshape(B, 1, N).astype(np.float32)).to(audio_t.device)

    def opus_st(audio_t: torch.Tensor) -> torch.Tensor:
        """Straight-through: forward = real Opus, backward = identity."""
        with torch.no_grad():
            y = opus_batch(audio_t)
        return audio_t + (y - audio_t).detach()

    # ---------- Multi-scale STFT for recon loss ----------
    def stft_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, 1, N)
        x_ = x.squeeze(1); y_ = y.squeeze(1)
        loss = 0.0
        for n_fft in (256, 512, 1024, 2048):
            window = torch.hann_window(n_fft, device=x.device)
            S_x = torch.stft(x_, n_fft=n_fft, hop_length=n_fft//4,
                             window=window, return_complex=True).abs()
            S_y = torch.stft(y_, n_fft=n_fft, hop_length=n_fft//4,
                             window=window, return_complex=True).abs()
            loss = loss + F.l1_loss(S_x, S_y)
        return loss

    # ---------- Eval (real Opus, real tokens) ----------
    @torch.no_grad()
    def eval_q0_match() -> tuple[float, list[float]]:
        """Returns (avg q0 match rate, per-codebook match rates) on eval set."""
        model.eval()
        per_q = np.zeros(n_q)
        for i in range(eval_x.size(0)):
            x = eval_x[i:i+1]
            tokens_A = model.encode(x)[0][0]  # (1, n_q, T_frames)
            audio_recon_A = model.decode([(tokens_A, None)])  # (1, 1, N)
            audio_after = opus_batch(audio_recon_A)
            # crop to encoder-friendly length
            min_n = min(audio_after.size(-1), x.size(-1))
            tokens_B = model.encode(audio_after[..., :min_n])[0][0]
            T = min(tokens_A.size(-1), tokens_B.size(-1))
            for q in range(n_q):
                m = (tokens_A[0, q, :T] == tokens_B[0, q, :T]).float().mean().item()
                per_q[q] += m
        per_q /= eval_x.size(0)
        model.train()
        return float(per_q[0]), per_q.tolist()

    # ---------- Train loop ----------
    print(f"\n[ft] training {n_steps} steps  bs={batch_size}  Opus={opus_kbps}k {opus_app}")
    print(f"     q0_ce_w={q0_ce_weight}  cons_w={consistency_weight}  recon_w={recon_weight}  lr={lr}  τ={softmax_temperature}\n")
    print(f"{'step':>5} {'L_tot':>7} {'L_q0ce':>7} {'L_cons':>7} {'L_recon':>7} "
          f"{'q0_eval':>8} {'q1_eval':>8} {'q2_eval':>8} {'note'}")

    t0 = time.time()
    best_q0 = 0.0
    for step in range(1, n_steps + 1):
        x = sample_batch(batch_size)  # (B, 1, N)

        # Forward A: clean encode → tokens → decoded audio
        z_clean = model.encoder(x)  # (B, 128, T_frames)
        # Quantize using frozen codebook (use .encode/.decode directly to avoid
        # any train-mode randomness in the quantizer's forward path):
        with torch.no_grad():
            tokens = model.quantizer.encode(z_clean, model.frame_rate, bandwidth_kbps)  # (n_q, B, T)
            z_q = model.quantizer.decode(tokens)  # (B, 128, T) continuous quantized
        # Decode (frozen): tokens → audio. Detach z_q so no gradient flows
        # back into encoder via the *clean* path; we only want the consistency
        # loss path to drive encoder updates.
        audio_recon = model.decoder(z_q.detach())  # (B, 1, N)

        # Forward B: through Opus, re-encode
        audio_after = opus_st(audio_recon)
        z_noisy = model.encoder(audio_after)

        # Align frame dims
        T = min(z_noisy.size(-1), z_clean.size(-1))
        z_noisy_T = z_noisy[..., :T]                # (B, D, T)
        z_clean_T = z_clean[..., :T].detach()       # (B, D, T)
        tokens_T = tokens[:, :, :T]                  # (n_q, B, T) — already detached (no_grad)

        # ---- L_q0_ce: direct Voronoi-cell CE on q0 ----
        # For every frame in z_noisy, build a logit vector over q0's 1024 codewords
        # using negative squared distance, then CE against the q0 token from clean
        # encode. This drives the encoder to put z_noisy in the SAME Voronoi cell
        # as z_clean — what actually decides token survival.
        B, D, _ = z_noisy_T.shape
        z_n_flat = z_noisy_T.permute(0, 2, 1).reshape(B*T, D)        # (B*T, D)
        # ||z||² + ||c||² − 2 z·c   (memory-friendly cdist)
        zsq = z_n_flat.pow(2).sum(-1, keepdim=True)                  # (B*T, 1)
        csq = codebook_q0.pow(2).sum(-1).unsqueeze(0)                # (1, 1024)
        dot = z_n_flat @ codebook_q0.T                               # (B*T, 1024)
        dist_q0 = (zsq + csq - 2 * dot).clamp(min=0.0)
        logits_q0 = -dist_q0 / softmax_temperature
        target_q0 = tokens_T[0].reshape(-1).long()                   # (B*T,)
        L_q0_ce = F.cross_entropy(logits_q0, target_q0)

        # ---- L_consistency: MSE pulls higher-q residuals toward stable cells too ----
        L_consistency = F.mse_loss(z_noisy_T, z_clean_T)

        # ---- L_recon: anchor encoder to "still produces meaningful tokens" ----
        L_recon = stft_l1(audio_recon, x)

        loss = (q0_ce_weight * L_q0_ce
                + consistency_weight * L_consistency
                + recon_weight * L_recon)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        opt.step()

        # ---- Eval ----
        if step == 1 or step % eval_every == 0 or step == n_steps:
            q0, per_q = eval_q0_match()
            note = ""
            if q0 > best_q0:
                best_q0 = q0
                ckpt_path = Path("/vol/checkpoints/encodec_opus_best.pt")
                ckpt_path.parent.mkdir(exist_ok=True)
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "step": step, "q0": q0, "per_q": per_q,
                    "bandwidth_kbps": bandwidth_kbps, "opus_kbps": opus_kbps,
                    "opus_app": opus_app, "n_q": n_q,
                }, ckpt_path)
                vol.commit()
                note = f"★ best (saved q0={q0*100:.1f}%)"
            print(f"{step:>5} {loss.item():>7.3f} {L_q0_ce.item():>7.3f} "
                  f"{L_consistency.item():>7.3f} {L_recon.item():>7.3f} "
                  f"{per_q[0]*100:>7.2f}% "
                  f"{per_q[1]*100 if n_q>1 else 0:>7.2f}% "
                  f"{per_q[2]*100 if n_q>2 else 0:>7.2f}%  {note}", flush=True)
        elif step % 25 == 0:
            print(f"{step:>5} {loss.item():>7.3f} {L_q0_ce.item():>7.3f} "
                  f"{L_consistency.item():>7.3f} {L_recon.item():>7.3f}", flush=True)

    # Final
    final_path = Path("/vol/checkpoints/encodec_opus_final.pt")
    torch.save({
        "encoder": model.encoder.state_dict(),
        "step": n_steps,
        "bandwidth_kbps": bandwidth_kbps, "opus_kbps": opus_kbps, "opus_app": opus_app,
        "n_q": n_q,
    }, final_path)
    vol.commit()

    elapsed = time.time() - t0
    print(f"\n[ft] DONE in {elapsed/60:.1f} min  best q0 = {best_q0*100:.2f}%")
    print(f"[ft]   final: {final_path}")
    print(f"[ft]   best:  /vol/checkpoints/encodec_opus_best.pt")
    return {"best_q0": best_q0, "elapsed_minutes": elapsed/60, "n_steps": n_steps}


@app.local_entrypoint()
def main(n_steps: int = 3000, batch_size: int = 8,
         opus_kbps: int = 48, opus_app: str = "audio"):
    result = train.remote(n_steps=n_steps, batch_size=batch_size,
                          opus_kbps=opus_kbps, opus_app=opus_app)
    print(f"\n[local] training finished: {result}")
    print(f"[local] pull: modal volume get {VOLUME_NAME} /checkpoints/encodec_opus_best.pt ./")
