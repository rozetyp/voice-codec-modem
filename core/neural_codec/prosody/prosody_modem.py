"""
Prosody modem: encode bits in per-window pitch deviations of a real-speech
carrier. Decoder compares received f0 against a shared cover-audio baseline
(both ends have the cover — same arrangement as stego_opus).

Why uniform-per-window beats intra-window tilt:
  - pyworld's pitch tracker smooths within-window slopes, so 80 ms tilts
    don't survive the synthesize→re-analyze pipeline (we hit 45-60% BER).
  - A uniform shift across a whole window survives synthesize→re-analyze with
    >98% sign-correctness (median detected shift within 1 cent of target).
  - And it survives Opus 8/12/24 kbps with no extra loss (probe γ result).

Encode: for each voiced window, multiply f0 by cent (if bit=1) or divide by cent
        (if bit=0).
Decode: extract received f0; compare mean log-pitch in window to cover baseline.

Voicing mask is computed from the COVER audio (deterministic, both ends agree).

Rate: 80 ms windows × 1 bit/window × ~50% voiced fraction (real speech) → 6 bps.
      40 ms windows → 12 bps. We default to 80 ms for safety; 40 ms also works.
"""
from __future__ import annotations
import numpy as np
import pyworld as pw

DEFAULT_WIN_MS = 80.0
DEFAULT_DELTA_CENTS = 100.0
DEFAULT_FRAME_PERIOD_MS = 5.0
DEFAULT_MIN_VOICED_FRACTION = 0.7
DEFAULT_F0_FLOOR = 60.0
DEFAULT_F0_CEIL = 500.0


def _f0_full(audio: np.ndarray, sr: int):
    """f0 + sp + ap (sp/ap only needed by encoder for synthesis)."""
    a = audio.astype(np.float64)
    f0, t = pw.dio(a, sr, frame_period=DEFAULT_FRAME_PERIOD_MS,
                   f0_floor=DEFAULT_F0_FLOOR, f0_ceil=DEFAULT_F0_CEIL)
    f0 = pw.stonemask(a, f0, t, sr)
    sp = pw.cheaptrick(a, f0, t, sr)
    ap = pw.d4c(a, f0, t, sr)
    return f0, t, sp, ap


def _f0_only(audio: np.ndarray, sr: int) -> np.ndarray:
    """Just f0 (the receiver doesn't need sp/ap)."""
    a = audio.astype(np.float64)
    f0, t = pw.dio(a, sr, frame_period=DEFAULT_FRAME_PERIOD_MS,
                   f0_floor=DEFAULT_F0_FLOOR, f0_ceil=DEFAULT_F0_CEIL)
    return pw.stonemask(a, f0, t, sr)


def _voicing_mask(f0: np.ndarray, frames_per_win: int,
                  min_voiced_fraction: float = DEFAULT_MIN_VOICED_FRACTION) -> np.ndarray:
    n_wins = len(f0) // frames_per_win
    mask = np.zeros(n_wins, dtype=bool)
    for i in range(n_wins):
        seg = f0[i*frames_per_win:(i+1)*frames_per_win]
        mask[i] = (seg > 0).mean() >= min_voiced_fraction
    return mask


def encode(cover: np.ndarray, sr: int, bits: np.ndarray,
           win_ms: float = DEFAULT_WIN_MS,
           delta_cents: float = DEFAULT_DELTA_CENTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed bits in cover by uniform per-window pitch shift.

    Returns (audio_modified, used_window_indices, cover_f0).
      cover_f0 is the receiver's reference baseline (1D array at 5ms cadence).
    """
    cover = cover.astype(np.float64)
    f0, t, sp, ap = _f0_full(cover, sr)
    frames_per_win = int(round(win_ms / DEFAULT_FRAME_PERIOD_MS))
    mask = _voicing_mask(f0, frames_per_win)
    usable = np.where(mask)[0]
    n_to_send = min(len(bits), len(usable))

    f0_mod = f0.copy()
    cent = 2 ** (delta_cents / 1200.0)
    for i in range(n_to_send):
        win_idx = int(usable[i])
        a = win_idx * frames_per_win
        b = a + frames_per_win
        seg = f0_mod[a:b]
        v = seg > 0
        if int(bits[i]) == 1:
            seg[v] = seg[v] * cent
        else:
            seg[v] = seg[v] / cent
        f0_mod[a:b] = seg

    y = pw.synthesize(f0_mod, sp, ap, sr, DEFAULT_FRAME_PERIOD_MS)
    return y.astype(np.float32), usable[:n_to_send], f0


def decode(received: np.ndarray, sr: int, cover_f0: np.ndarray,
           n_bits: int = -1,
           win_ms: float = DEFAULT_WIN_MS) -> tuple[np.ndarray, np.ndarray]:
    """Decode bits from received audio against the shared cover-baseline f0.

    Returns (bits, used_window_indices).
      The voicing mask comes from cover_f0 (both ends agree).
    """
    f0_rx = _f0_only(received, sr)
    frames_per_win = int(round(win_ms / DEFAULT_FRAME_PERIOD_MS))
    mask = _voicing_mask(cover_f0, frames_per_win)
    usable = np.where(mask)[0]
    if n_bits > 0 and len(usable) > n_bits:
        usable = usable[:n_bits]

    bits = np.zeros(len(usable), dtype=np.int32)
    n = min(len(f0_rx), len(cover_f0))
    for i, win_idx in enumerate(usable):
        a = int(win_idx) * frames_per_win
        b = a + frames_per_win
        if b > n:
            break
        seg_rx = f0_rx[a:b]
        seg_bs = cover_f0[a:b]
        m = (seg_rx > 0) & (seg_bs > 0)
        if m.sum() < 4:
            bits[i] = 0
            continue
        # Average cents shift across voiced frames in window
        cents = 1200.0 * (np.log2(seg_rx[m]) - np.log2(seg_bs[m]))
        bits[i] = 1 if cents.mean() > 0 else 0
    return bits, usable


def encode_decode_test(cover: np.ndarray, sr: int, n_bits: int = 64,
                       win_ms: float = DEFAULT_WIN_MS,
                       delta_cents: float = DEFAULT_DELTA_CENTS,
                       seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    bits_in = rng.randint(0, 2, size=n_bits, dtype=np.int32)
    audio, used_enc, cover_f0 = encode(cover, sr, bits_in, win_ms, delta_cents)
    bits_out, used_dec = decode(audio, sr, cover_f0, n_bits=len(used_enc), win_ms=win_ms)
    n = min(len(used_enc), len(used_dec))
    if n == 0:
        return {"sent": 0, "decoded": 0, "errors": 0, "ber": 1.0}
    errs = int((bits_in[:n] != bits_out[:n]).sum())
    return {"sent": int(n), "decoded": int(n), "errors": errs, "ber": errs / n if n else 1.0}
