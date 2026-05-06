"""Eval the AMR-NB stego ckpt with repetition coding to find actual reliable rate."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import numpy as np
import soundfile as sf
import torch

from neural_codec import DEVICE, SR, SYMBOL_MS
from stego_codec import StegEncoder, StegDecoder
from stego_amrnb import amrnb_round_trip_batch
from adversarial_realism import RealSpeechSampler

SYMBOL_N = SR * SYMBOL_MS // 1000


def encode_with_repetition(data_bits: np.ndarray, n_rep: int) -> np.ndarray:
    """Repeat each data bit n_rep times in sequence."""
    return np.repeat(data_bits, n_rep)


def decode_with_repetition(noisy_bits: np.ndarray, n_data: int, n_rep: int) -> np.ndarray:
    """Majority vote over groups of n_rep bits."""
    grouped = noisy_bits[:n_data * n_rep].reshape(n_data, n_rep)
    return (grouped.sum(axis=1) >= (n_rep + 1) // 2 + 1 - 1).astype(int) if False else (grouped.mean(axis=1) >= 0.5).astype(int)


def measure(ckpt_path: str, n_data_bits_list: list[int], n_rep_list: list[int],
            n_seeds: int = 3, samples_per_seed_blocks: int = 100):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits_per_sym = ckpt["n_bits"]
    pert = ckpt.get("perturbation_scale", 0.5)
    enc = StegEncoder(n_bits_per_sym).to(DEVICE)
    dec = StegDecoder(n_bits_per_sym).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    sampler = RealSpeechSampler([
        Path(str(REPO_ROOT / "tests/data/raw/synth_readaloud_01.wav")),
        Path(str(REPO_ROOT / "tests/data/raw/lex_jensen.wav")),
    ])

    raw_bps = n_bits_per_sym * 1000 / SYMBOL_MS
    print(f"# ckpt: {ckpt_path}")
    print(f"# n_bits/sym={n_bits_per_sym}  raw_bps={raw_bps:.0f}  pert_scale={pert}")
    print()

    # First: baseline raw BER with no FEC
    raw_bers = []
    snrs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        cover = sampler.sample(samples_per_seed_blocks)
        bits = torch.randint(0, 2, (samples_per_seed_blocks, n_bits_per_sym), device=DEVICE).float()
        with torch.no_grad():
            modified, _ = enc(cover, bits, perturbation_scale=pert)
            rx = amrnb_round_trip_batch(modified)
            logits = dec(rx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            raw_bers.append((preds != bits).float().mean().item())
            cover_pwr = cover.pow(2).mean(dim=-1) + 1e-9
            pert_pwr = (modified - cover).pow(2).mean(dim=-1) + 1e-12
            snrs.append((10 * torch.log10(cover_pwr / pert_pwr)).mean().item())
    raw_ber = float(np.mean(raw_bers))
    snr_db = float(np.mean(snrs))
    print(f"# baseline raw: BER={raw_ber*100:.2f}%  SNR={snr_db:.1f} dB  ({n_seeds*samples_per_seed_blocks*n_bits_per_sym} bits transmitted)")
    print()

    # Now: try repetition coding at different rates
    print(f"{'n_rep':>6}  {'eff_rate':>10}  {'post_BER':>10}  {'errors / bits':>15}")
    for n_rep in n_rep_list:
        all_data_bits = []
        all_decoded = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + n_rep)
            # Generate data bits, expand with repetition, embed, recover, decode
            n_total_bits_per_block = n_bits_per_sym
            n_data_bits_per_block = n_total_bits_per_block // n_rep
            if n_data_bits_per_block < 1: continue
            data_bits = np.random.randint(0, 2, (samples_per_seed_blocks, n_data_bits_per_block))
            tx_bits = np.zeros((samples_per_seed_blocks, n_total_bits_per_block), dtype=int)
            for i, d in enumerate(data_bits):
                rep = encode_with_repetition(d, n_rep)
                tx_bits[i, :len(rep)] = rep
                # Remaining slots get random fill (don't carry payload, decoder ignores)
                if len(rep) < n_total_bits_per_block:
                    tx_bits[i, len(rep):] = np.random.randint(0, 2, n_total_bits_per_block - len(rep))
            tx = torch.from_numpy(tx_bits.astype(np.float32)).to(DEVICE)
            cover = sampler.sample(samples_per_seed_blocks)
            with torch.no_grad():
                modified, _ = enc(cover, tx, perturbation_scale=pert)
                rx = amrnb_round_trip_batch(modified)
                logits = dec(rx)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(int)
            decoded = np.array([decode_with_repetition(p, n_data_bits_per_block, n_rep)
                                for p in preds])
            all_data_bits.append(data_bits.flatten())
            all_decoded.append(decoded.flatten())
        data_concat = np.concatenate(all_data_bits)
        decoded_concat = np.concatenate(all_decoded)
        post_ber = float(np.mean(data_concat != decoded_concat))
        n_errs = int(np.sum(data_concat != decoded_concat))
        n_total = len(data_concat)
        eff_rate = raw_bps / n_rep
        print(f"  {n_rep:>4}    {eff_rate:>6.0f} bps   {post_ber*100:>7.4f}%   {n_errs} / {n_total}")


if __name__ == "__main__":
    measure(str(REPO_ROOT / "core/neural_codec/cellular/ckpt_amrnb_real.pt"),
            n_data_bits_list=[8, 4, 2, 1],
            n_rep_list=[1, 2, 3, 5, 7, 11, 16],
            n_seeds=3,
            samples_per_seed_blocks=200)
