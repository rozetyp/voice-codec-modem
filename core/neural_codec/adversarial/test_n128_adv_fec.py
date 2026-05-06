"""Test adversarial n=128 with multiple RS codes to pick the working one."""
import os
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import sys
sys.path.insert(0, str(REPO_ROOT / "core" / "neural_codec"))
import warnings
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
import numpy as np
import torch

from neural_codec import Encoder, Decoder, real_opus_batch, DEVICE, SR, SYMBOL_MS
from neural_with_fec import InterleavedRS, bytes_to_bits, bits_to_bytes
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test(ckpt_path, n_data, n_total, n_data_bytes=1900, n_seeds=5):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    n_bits = ckpt["n_bits"]
    enc = Encoder(n_bits=n_bits).to(DEVICE)
    dec = Decoder(n_bits=n_bits).to(DEVICE)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    fec = InterleavedRS(n_data=n_data, n_total=n_total, depth=8)
    raw_bps = n_bits * 1000 / SYMBOL_MS
    eff_bps = raw_bps * n_data / n_total

    print(f"# RS({n_total},{n_data}) -> {eff_bps:.0f} bps reliable")
    raws = []; posts = []; fails = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        data = bytes(rng.integers(0, 256, n_data_bytes).tolist())
        coded, pad = fec.encode(data)
        tx_bits = bytes_to_bits(coded)
        pad_to_sym = (-len(tx_bits)) % n_bits
        tx_bits_full = np.concatenate([tx_bits, np.zeros(pad_to_sym, dtype=np.uint8)])
        symbols = tx_bits_full.reshape(-1, n_bits)
        with torch.no_grad():
            sb = torch.from_numpy(symbols.astype(np.float32)).to(DEVICE)
            audio = enc(sb)
            audio_codec = real_opus_batch(audio)
            logits = dec(audio_codec)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
        rx_bits = preds.reshape(-1)[:len(tx_bits)]
        raw_ber = float(np.mean(rx_bits != tx_bits))
        rx_bytes = bits_to_bytes(rx_bits)
        decoded, n_corr, n_fail = fec.decode(rx_bytes, pad)
        decoded = decoded[:n_data_bytes]
        bx = bytes_to_bits(data); by = bytes_to_bits(decoded)[:len(bx)]
        post_ber = float(np.mean(np.array(bx) != np.array(by)))
        raws.append(raw_ber); posts.append(post_ber); fails.append(n_fail)
        print(f"   seed {seed}: raw {raw_ber*100:.2f}%  post {post_ber*100:.4f}%  fail_blks {n_fail}")
    print(f"   AVG: raw {np.mean(raws)*100:.2f}%  post {np.mean(posts)*100:.4f}%  fail {np.mean(fails):.1f}")
    return np.mean(raws), np.mean(posts), np.mean(fails)


print(f"# adversarial n=128 checkpoint")
for n_data, n_total in [(191, 255), (159, 255), (127, 255)]:
    test(str(REPO_ROOT / "core/neural_codec/adversarial/ckpt_n128_adv.pt"), n_data, n_total)
    print()
