from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from .features import FEATURES


def rolling_variance_plot(
    recordings: dict[str, pl.DataFrame],
    out_path: Path | str,
    window_turns: int = 8,
) -> None:
    """One subplot per feature, one line per recording.

    recordings: {label: dataframe with turn features, time in elapsed_min}.
    Colors: 'human_*' -> blue shades, 'synth_*' -> red shades.
    """
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(10, 3 * len(FEATURES)), sharex=True)
    if len(FEATURES) == 1:
        axes = [axes]

    for ax, feat in zip(axes, FEATURES):
        for label, df in recordings.items():
            if feat not in df.columns:
                continue
            s = df[feat].cast(pl.Float64)
            if s.drop_nulls().len() < window_turns:
                continue
            rolling = s.rolling_std(window_size=window_turns, min_samples=max(3, window_turns // 2))
            color = "C0" if label.startswith("human") else "C3"
            alpha = 0.8
            ax.plot(df["elapsed_min"], rolling, label=label, color=color, alpha=alpha)
        ax.set_ylabel(f"{feat}\nrolling std")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("elapsed minutes")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def feature_trajectory_plot(
    recordings: dict[str, pl.DataFrame],
    out_path: Path | str,
) -> None:
    """Raw feature values over time, one subplot per feature."""
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(10, 3 * len(FEATURES)), sharex=True)
    if len(FEATURES) == 1:
        axes = [axes]

    for ax, feat in zip(axes, FEATURES):
        for label, df in recordings.items():
            if feat not in df.columns:
                continue
            color = "C0" if label.startswith("human") else "C3"
            ax.plot(df["elapsed_min"], df[feat], marker=".", linestyle="-", label=label, color=color, alpha=0.6)
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("elapsed minutes")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
