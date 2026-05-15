#!/usr/bin/env python3
"""Create a publication-quality runtime comparison figure.

The script reads ``runtime_vs_features_results.csv`` from this directory and
writes ``runtime_comparison`` as PNG, PDF, and SVG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "MS_linear": "MS-linear",
    "mRMR_MS_linear": "mRMR-MS",
    "mRMR": "mRMR",
    "CONMI_FS": "CONMI-FS",
    "greedy": "Greedy",
}

METHOD_ORDER = [
    "MS_linear",
    "CONMI_FS",
    "mRMR",
    "mRMR_MS_linear",
    "greedy",
]

COLORS = {
    "MS_linear": "#0072B2",
    "CONMI_FS": "#009E73",
    "mRMR": "#D55E00",
    "mRMR_MS_linear": "#CC79A7",
    "greedy": "#4D4D4D",
}

MARKERS = {
    "MS_linear": "o",
    "CONMI_FS": "s",
    "mRMR": "^",
    "mRMR_MS_linear": "D",
    "greedy": "v",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot runtime against total number of features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "runtime_vs_features_results.csv",
        help="CSV produced by the runtime benchmark.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=script_dir / "runtime_comparison",
        help="Output path without extension.",
    )
    return parser.parse_args()


def ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"])
    methods = [method for method in METHOD_ORDER if method in present]
    methods.extend(sorted(present.difference(methods)))
    return methods


def clean_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "n_classes",
        "n_samples",
        "n_features",
        "method",
        "mean_s",
        "std_s",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    for column in ["n_classes", "n_samples", "n_features"]:
        df[column] = df[column].astype(int)
    df["mean_s"] = df["mean_s"].astype(float)
    df["std_s"] = df["std_s"].fillna(0).astype(float)
    return df.sort_values(["n_classes", "n_samples", "n_features", "method"])


def apply_axis_style(ax: plt.Axes, feature_values: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(feature_values)
    ax.set_xticklabels([f"{value:,}" for value in feature_values])
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.35)
    ax.grid(axis="x", which="major", color="#ECECEC", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", which="major", length=3.2, pad=2)
    ax.tick_params(axis="both", which="minor", length=1.8)


def add_method_series(ax: plt.Axes, sub: pd.DataFrame, methods: list[str]) -> None:
    min_positive = max(float(sub["mean_s"].min()) / 20.0, 1e-4)
    for method in methods:
        series = sub[sub["method"] == method].sort_values("n_features")
        if series.empty:
            continue

        x = series["n_features"].to_numpy(dtype=float)
        y = series["mean_s"].to_numpy(dtype=float)
        yerr = series["std_s"].to_numpy(dtype=float)
        lower = np.maximum(y - yerr, min_positive)
        upper = y + yerr

        ax.fill_between(
            x,
            lower,
            upper,
            color=COLORS.get(method, "#777777"),
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=COLORS.get(method, "#777777"),
            marker=MARKERS.get(method, "o"),
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=METHOD_LABELS.get(method, method),
            zorder=3,
        )


def make_figure(df: pd.DataFrame) -> plt.Figure:
    configure_style()

    classes = sorted(df["n_classes"].unique())
    samples = sorted(df["n_samples"].unique())
    features = sorted(df["n_features"].unique())
    methods = ordered_methods(df)

    fig, axes = plt.subplots(
        len(classes),
        len(samples),
        figsize=(7.2, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )

    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")
    for row, n_classes in enumerate(classes):
        for col, n_samples in enumerate(samples):
            ax = axes[row, col]
            sub = df[
                (df["n_classes"] == n_classes)
                & (df["n_samples"] == n_samples)
            ]
            add_method_series(ax, sub, methods)
            apply_axis_style(ax, features)

            ax.set_title(f"{n_samples:,} samples", pad=5)
            ax.text(
                0.02,
                0.97,
                next(panel_letters),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )

            if col == 0:
                ax.text(
                    -0.32,
                    0.5,
                    f"{n_classes} classes",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=False,
        handlelength=1.7,
        columnspacing=1.2,
    )
    fig.supxlabel("Total number of features", y=0.045, fontsize=8)
    fig.supylabel("Runtime (s, log scale)", x=0.035, fontsize=8)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.13,
        top=0.89,
        wspace=0.12,
        hspace=0.25,
    )
    return fig


def main() -> None:
    args = parse_args()
    df = clean_results(args.input)
    fig = make_figure(df)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
