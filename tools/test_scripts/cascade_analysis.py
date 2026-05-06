"""Paper-aligned cascade analysis with Mann-Kendall trend + coupled drift.

Per the 1989 Soviet paper's cascade model (see blueprint):

    Stage 3 (decision/motor):   latency ↑, tempo ↓, pause rate ↑, filler ↑
    Stage 4 (vocabulary):       n_words/turn ↓
    Stage 5 (working memory):   hesitation ↑, word repetition ↑
    Stage 7 (affect/voice):     jitter ↑, shimmer ↑, F0 std ↓, HNR ↓

A real tired human shows:
  - Monotonic drift in the PREDICTED direction for 3+ features (Mann-Kendall)
  - High coupled-drift score (features move together — shared physiology)
  - Forward cascade timing: Stage 3 drift appears before Stage 7 drift

Synthetic voice shows one of:
  - No monotonic trend on any feature (flat)
  - Anti-fatigue drift (tempo speeding up etc.) — narrative-arc TTS
  - Backward cascade (Stage 7 drift without Stage 3) — physiologically impossible

A well-rested human shows:
  - Little trend in either direction (LOW-SIGNAL, not synthetic)

Usage:
    uv run python scripts/cascade_analysis.py data/features/*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from voice_detector.features import FEATURE_SPECS


def mann_kendall(x: np.ndarray) -> tuple[float | None, str | None]:
    """Return (p_value, direction) for a monotonic trend test.

    Direction is 'up', 'down', or None if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return None, None
    try:
        import pymannkendall as mk

        r = mk.original_test(x)
        return float(r.p), r.trend  # 'increasing' / 'decreasing' / 'no trend'
    except Exception:
        return None, None


def analyze(path: Path, label: str | None = None, trend_alpha: float = 0.05) -> dict:
    df = pl.read_parquet(path)
    n = len(df)
    dur = df["elapsed_min"].max()
    label = label or path.stem

    print(f"\n{'=' * 88}")
    print(f"  {label}  —  {n} turns, {dur:.1f} min")
    print(f"{'=' * 88}")
    print(f"{'feature':24s} {'stage':>5s} {'n':>5s} {'mk_p':>8s} {'trend':>12s} {'pred':>6s} {'verdict':>14s}")
    print("-" * 88)

    per_feature = []
    for feat, predicted, stage in FEATURE_SPECS:
        if feat not in df.columns:
            continue
        series = df[feat].drop_nulls().to_numpy()
        p, trend = mann_kendall(series)
        if p is None:
            per_feature.append(dict(feat=feat, stage=stage, n=len(series), p=None, trend=None, match=False))
            print(f"{feat:24s} {stage:>5d} {len(series):>5d} {'n/a':>8s} {'no data':>12s}")
            continue
        pred_dir = "increasing" if predicted == "up" else "decreasing"
        significant = p < trend_alpha
        match = significant and trend == pred_dir
        anti = significant and trend != pred_dir and trend != "no trend"
        if match:
            verdict = "fatigue ✓"
        elif anti:
            verdict = "ANTI-fatigue"
        else:
            verdict = "-"
        per_feature.append(dict(feat=feat, stage=stage, n=len(series), p=p, trend=trend,
                                 match=match, anti=anti, predicted=predicted))
        print(f"{feat:24s} {stage:>5d} {len(series):>5d} {p:>8.4f} {trend:>12s} {predicted:>6s} {verdict:>14s}")

    # Aggregate
    n_match = sum(1 for r in per_feature if r["match"])
    n_anti = sum(1 for r in per_feature if r.get("anti"))
    n_tested = sum(1 for r in per_feature if r["p"] is not None)

    # Per-stage fraction
    stage_totals: dict[int, list[bool]] = {}
    for r in per_feature:
        if r["p"] is None:
            continue
        stage_totals.setdefault(r["stage"], []).append(r["match"])

    # Coupled drift score — Spearman correlation of fatigue-aligned rank series
    # For each eligible feature, sign-flip if prediction is "down" so all
    # fatigue-aligned series point the same direction, then compute mean
    # pairwise Spearman rho.
    fatigue_series = []
    for feat, predicted, _ in FEATURE_SPECS:
        if feat not in df.columns:
            continue
        s = df[feat].to_numpy()
        mask = ~np.isnan(s)
        if mask.sum() < 20:
            continue
        signed = s * (+1 if predicted == "up" else -1)
        fatigue_series.append((feat, signed, mask))

    couplings = []
    if len(fatigue_series) >= 2:
        for i in range(len(fatigue_series)):
            for j in range(i + 1, len(fatigue_series)):
                _, a, mask_a = fatigue_series[i]
                _, b, mask_b = fatigue_series[j]
                m = mask_a & mask_b
                if m.sum() < 20:
                    continue
                rho, _ = spearmanr(a[m], b[m])
                if not np.isnan(rho):
                    couplings.append(rho)

    coupled_drift = float(np.mean(couplings)) if couplings else float("nan")

    # Cascade: do early-stage (3/4/5) features fire more than late-stage (7)?
    early_frac = late_frac = None
    if stage_totals:
        early = [x for s in (3, 4, 5) for x in stage_totals.get(s, [])]
        late = stage_totals.get(7, [])
        if early and late:
            early_frac = sum(early) / len(early)
            late_frac = sum(late) / len(late)

    print()
    print(f"  Fatigue-aligned trends:   {n_match}/{n_tested}")
    print(f"  Anti-fatigue trends:      {n_anti}/{n_tested}")
    for stage in sorted(stage_totals):
        hits = sum(stage_totals[stage])
        tot = len(stage_totals[stage])
        print(f"  Stage {stage}: {hits}/{tot}")
    print(f"  Coupled drift (Spearman ρ̄):  {coupled_drift:.3f}"
          if coupled_drift == coupled_drift else "  Coupled drift: n/a")

    cascade_msg = "n/a"
    if early_frac is not None and late_frac is not None:
        if early_frac > late_frac + 0.1:
            cascade_msg = f"forward — early {early_frac:.0%} > late {late_frac:.0%}"
        elif late_frac > early_frac + 0.1:
            cascade_msg = f"BACKWARDS — late {late_frac:.0%} > early {early_frac:.0%}"
        else:
            cascade_msg = f"balanced — early {early_frac:.0%} ≈ late {late_frac:.0%}"
    print(f"  Cascade order:             {cascade_msg}")

    # ─── Verdict logic ─────────────────────────────────────────────────────
    # Rules (tentative, pre-registered — will tune only on unseen data):
    #   HUMAN  :  ≥3/tested fatigue-aligned  AND  coupled_drift ≥ 0.15
    #                                         AND  not backward cascade
    #   SYNTHETIC: anti-fatigue drift dominates (≥30% of tested)
    #                OR backward cascade (late > early by ≥20pp)
    #                OR coupled drift ≤ 0.00 and ≥2 anti trends
    #   LOW-SIGNAL: ≤1 trend total, coupled drift near 0 (could be well-rested human)
    #   UNCERTAIN: everything else

    match_frac = n_match / max(n_tested, 1)
    anti_frac = n_anti / max(n_tested, 1)
    backward = (late_frac is not None and early_frac is not None
                and late_frac > early_frac + 0.2)

    if backward:
        verdict = "SYNTHETIC (backward cascade)"
    elif anti_frac >= 0.30:
        verdict = "SYNTHETIC (anti-fatigue drift)"
    elif n_match + n_anti <= 1 and abs(coupled_drift if coupled_drift == coupled_drift else 0) < 0.15:
        verdict = "LOW-SIGNAL (insufficient drift — could be well-rested human)"
    elif n_match >= 3 and (coupled_drift == coupled_drift and coupled_drift >= 0.15):
        verdict = "HUMAN (forward cascade + coupled fatigue drift)"
    elif n_match >= 3:
        verdict = "HUMAN-LIKE (fatigue drift, weak coupling)"
    else:
        verdict = "UNCERTAIN"

    print(f"  Overall:                   {verdict}")

    return dict(
        label=label, match=n_match, anti=n_anti, tested=n_tested,
        coupled_drift=coupled_drift, verdict=verdict,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()

    results = []
    for p in args.paths:
        if p.is_dir():
            for f in sorted(p.glob("*.parquet")):
                results.append(analyze(f))
        else:
            results.append(analyze(p))

    if len(results) > 1:
        print(f"\n{'=' * 88}")
        print("Summary")
        print(f"{'=' * 88}")
        print(f"{'label':40s} {'match':>6s} {'anti':>6s} {'ρ̄':>7s}  verdict")
        for r in results:
            rho = f"{r['coupled_drift']:+.2f}" if r['coupled_drift'] == r['coupled_drift'] else "  n/a"
            print(f"  {r['label']:38s} {r['match']:>6d} {r['anti']:>6d} {rho:>7s}  {r['verdict']}")


if __name__ == "__main__":
    main()
