"""Run stationarity tests and produce plots comparing human vs synthetic recordings.

Usage:
    python scripts/analyze.py

Reads every .parquet in data/features/. Files prefixed 'human_' or 'call_*_ch1'
with a human label are treated as human; 'synth_' or 'call_*' with a synthetic
label are treated as synthetic.

For first pass we infer label from filename prefix:
  human_*   -> human
  synth_*   -> synthetic
  readalouod_human_* / readaloud_synth_*  -> smoke test

Emits:
  data/plots/rolling_variance.png
  data/plots/feature_trajectories.png
  data/plots/stationarity_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from voice_detector.features import FEATURES
from voice_detector.plots import feature_trajectory_plot, rolling_variance_plot
from voice_detector.stationarity import score_recording


def label_from_name(name: str) -> str:
    if name.startswith("human") or "human" in name:
        return "human"
    if name.startswith("synth") or "synth" in name:
        return "synthetic"
    # fallback — ch1 of a call of unknown label
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--plots-dir", type=Path, default=Path("data/plots"))
    args = parser.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.features_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no feature parquets in {args.features_dir}")

    recordings: dict[str, pl.DataFrame] = {}
    summary_rows = []

    for f in files:
        df = pl.read_parquet(f)
        label = label_from_name(f.stem)
        recordings[f.stem] = df

        score, results = score_recording(df, FEATURES)
        for r in results:
            summary_rows.append({"recording": f.stem, "label": label, **r.as_dict()})
        print(f"{f.stem:40s}  [{label:9s}]  non-stationary features: {score}/{len(FEATURES)}")

    summary = pl.DataFrame(summary_rows)
    summary.write_csv(args.plots_dir / "stationarity_summary.csv")
    print(f"\nwrote {args.plots_dir / 'stationarity_summary.csv'}")

    # Aggregate: mean non-stationary count per label
    print("\n── Mean stationarity verdict by label ─────────────────")
    agg = (
        summary.group_by(["label", "feature"])
        .agg(pl.col("is_non_stationary").mean().alias("frac_non_stationary"))
        .sort(["label", "feature"])
    )
    print(agg)

    rolling_variance_plot(recordings, args.plots_dir / "rolling_variance.png")
    feature_trajectory_plot(recordings, args.plots_dir / "feature_trajectories.png")
    print(f"\nwrote plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
