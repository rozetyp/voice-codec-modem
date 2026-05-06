from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import parselmouth
import polars as pl
from parselmouth.praat import call

from .transcribe import Segment


# Simple filler and hesitation lexicons — tuned to what whisper actually emits
FILLER_RE = re.compile(r"\b(uh+|um+|hmm+|er+|mm+|mmm+|erm+)\b", re.I)
HESITATION_PHRASES = [
    r"\bi mean\b",
    r"\byou know\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\blike,\b",
    r"\bright\?",
]
HESITATION_RE = re.compile("|".join(HESITATION_PHRASES), re.I)


def segments_to_turns(segments: list[Segment], min_words: int = 3, merge_gap_s: float = 0.0) -> list[Segment]:
    """For single-speaker audio (smoke test or per-channel), optionally collapse
    whisper segments into larger 'turn-like' chunks by merging adjacent
    segments separated by < merge_gap_s of silence.

    Default merge_gap_s=0.0 keeps whisper's native segmentation (roughly
    phrase- or sentence-level, ~5-8s chunks) as the per-turn granularity.

    For real two-speaker audio use per-channel processing: each channel is
    already one speaker's turns, and you'd pass merge_gap_s ~ 0.4-0.8 to
    merge within-turn breaths.
    """
    if not segments:
        return []
    merged: list[Segment] = []
    cur = segments[0]
    for nxt in segments[1:]:
        if nxt.start - cur.end < merge_gap_s:
            cur = Segment(
                start=cur.start,
                end=nxt.end,
                text=(cur.text + " " + nxt.text).strip(),
                words=cur.words + nxt.words,
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return [s for s in merged if len(s.words) >= min_words]


def _voice_stats(snd: parselmouth.Sound, t0: float, t1: float) -> dict:
    """Per-turn voice-quality features: F0 std, jitter, shimmer, HNR.

    Jitter/shimmer are cycle-to-cycle perturbations (vocal fold dynamics).
    HNR is harmonics-to-noise ratio (phonation quality).
    All come from parselmouth/Praat. TTS cannot produce realistic drift in
    these features from text alone — they reflect vocal fold physiology.
    """
    out = {"f0_std_hz": float("nan"), "jitter_pct": float("nan"),
           "shimmer_pct": float("nan"), "hnr_db": float("nan")}
    if t1 - t0 < 0.4:
        return out
    try:
        part = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
        pitch = part.to_pitch_ac(pitch_floor=75.0, pitch_ceiling=500.0)
        vals = pitch.selected_array["frequency"]
        voiced = vals[vals > 0]
        if len(voiced) >= 5:
            out["f0_std_hz"] = float(np.std(voiced))
        # Need a PointProcess for jitter/shimmer
        try:
            pp = call([part, pitch], "To PointProcess (cc)")
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([part, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            if jitter is not None and jitter == jitter:  # not NaN
                out["jitter_pct"] = float(jitter) * 100
            if shimmer is not None and shimmer == shimmer:
                out["shimmer_pct"] = float(shimmer) * 100
        except Exception:
            pass
        try:
            harmonicity = part.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            if hnr is not None and hnr == hnr and hnr != -200:  # -200 = Praat's undefined
                out["hnr_db"] = float(hnr)
        except Exception:
            pass
    except Exception:
        pass
    return out


def _text_stats(turn_text: str, words: list, duration_s: float) -> dict:
    """Per-turn text features: filler rate, hesitation count, word repetition.

    Fillers ("uh", "um") and hesitation phrases ("I mean", "you know") rise
    under cognitive load. Immediate word repetitions ("the-the", "I-I") are
    a working-memory failure signature.
    """
    out = {"filler_rate_per_min": float("nan"),
           "hesitation_rate_per_min": float("nan"),
           "word_repetition_rate": float("nan")}
    if duration_s <= 0.1:
        return out
    text = turn_text.lower()
    n_fillers = len(FILLER_RE.findall(text))
    n_hesit = len(HESITATION_RE.findall(text))
    out["filler_rate_per_min"] = 60.0 * n_fillers / duration_s
    out["hesitation_rate_per_min"] = 60.0 * n_hesit / duration_s

    # Immediate word repetitions — normalize by total word count
    if len(words) >= 4:
        reps = 0
        for i in range(1, len(words)):
            w0 = re.sub(r"[^\w]", "", words[i - 1].text).lower()
            w1 = re.sub(r"[^\w]", "", words[i].text).lower()
            if w0 and w0 == w1 and len(w0) > 1:
                reps += 1
        out["word_repetition_rate"] = reps / len(words)
    return out


def _pause_features(words, t0: float, t1: float) -> tuple[float, float]:
    """Within-turn pause rate (#/min) and pause-duration std dev (s).

    A 'pause' = inter-word gap >= 250ms inside the turn.
    """
    if len(words) < 2:
        return float("nan"), float("nan")
    gaps = [max(0.0, words[i].start - words[i - 1].end) for i in range(1, len(words))]
    pauses = [g for g in gaps if g >= 0.25]
    duration_s = max(t1 - t0, 1e-6)
    rate_per_min = 60.0 * len(pauses) / duration_s
    std = float(np.std(pauses)) if len(pauses) >= 2 else float("nan")
    return rate_per_min, std


def extract_turn_features(
    audio_path: Path | str,
    turns: list[Segment],
    prev_turn_end_global: float | None = None,
) -> pl.DataFrame:
    """Compute per-turn features.

    Columns: turn_idx, start, end, duration_s, elapsed_min, n_words,
             latency_s, tempo_wps, f0_std_hz, pause_rate_per_min, pause_std_s

    `latency_s` for a single-speaker recording is the gap to the previous turn
    of the *same* speaker — useful as an internal-cadence proxy but not the
    true conversational response latency. For real conversation analysis pass
    stereo + match across channels (see analysis script).
    """
    snd = parselmouth.Sound(str(audio_path))
    rows = []
    prev_end = prev_turn_end_global
    t_ref = turns[0].start if turns else 0.0
    for i, t in enumerate(turns):
        duration = max(t.end - t.start, 1e-6)
        n_words = len(t.words)
        tempo = n_words / duration
        latency = float("nan") if prev_end is None else max(0.0, t.start - prev_end)
        voice = _voice_stats(snd, t.start, t.end)
        text_feat = _text_stats(t.text, t.words, duration)
        pause_rate, pause_std = _pause_features(t.words, t.start, t.end)
        rows.append(
            {
                "turn_idx": i,
                "start": t.start,
                "end": t.end,
                "duration_s": duration,
                "elapsed_min": (t.start - t_ref) / 60.0,
                "text": t.text,
                "n_words": n_words,
                "latency_s": latency,
                "tempo_wps": tempo,
                "pause_rate_per_min": pause_rate,
                "pause_std_s": pause_std,
                **voice,
                **text_feat,
            }
        )
        prev_end = t.end
    return pl.DataFrame(rows)


# Stage 3 = decision/motor, Stage 4 = vocabulary, Stage 5 = working memory,
# Stage 7 = affect/voice quality — per the 1989 Soviet paper cascade
FEATURE_SPECS: list[tuple[str, str, int]] = [
    ("latency_s",            "up",   3),
    ("tempo_wps",            "down", 3),
    ("pause_rate_per_min",   "up",   3),
    ("filler_rate_per_min",  "up",   3),
    ("n_words",              "down", 4),
    ("hesitation_rate_per_min", "up", 5),
    ("word_repetition_rate", "up",   5),
    ("jitter_pct",           "up",   7),
    ("shimmer_pct",          "up",   7),
    ("f0_std_hz",            "down", 7),
    ("hnr_db",               "down", 7),
]

FEATURES = [f[0] for f in FEATURE_SPECS]
