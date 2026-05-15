#!/usr/bin/env python3
"""
Plot grouped Markov-blanket comparison results across all datasets,
in a single grouped boxplot by sample size (250, 1000, 5000).

Usage:
  pixi run python DATA/synthetic/plot_grouped_mb_comparison.py
  pixi run python DATA/synthetic/plot_grouped_mb_comparison.py --results_root /path/to/RESULTS/synthetic
  pixi run python DATA/synthetic/plot_grouped_mb_comparison.py --output_dir /path/to/output

Inputs expected in:
  RESULTS/synthetic/save_<dataset>/<run_folder>/results/markov_blanket_detection_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FOLDER_PATTERN = re.compile(
    r"^(?P<network>.+?)__target_(?P<target>.+?)__n_(?P<n_samples>\d+)__rep_(?P<rep>\d+)$"
)

METHOD_ALIAS = {
    "MS": "Smfs",
    "mRMR": "mRMR",
    "MRMR": "mRMR",
    "mRMR_MS_linear": "mRMR-MS",
}
METHOD_ORDER = ["Smfs", "mRMR", "mRMR-MS"]
METHOD_COLORS = {
    "Smfs": "#E64B35",
    "mRMR": "#4DBBD5",
    "mRMR-MS": "#00A087",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped MB comparison by method and sample size across all datasets."
    )
    parser.add_argument(
        "--results_root",
        default="/mnt/storage/mlopezdecas/MRMR/new_experiments/RESULTS/synthetic",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output dir. Default: <results_root>/mb_grouped_comparison_all",
    )
    return parser.parse_args()


def _collect_summary_rows(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for save_dir in sorted(results_root.glob("save_*")):
        if not save_dir.is_dir():
            continue
        for run_dir in sorted(save_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            match = FOLDER_PATTERN.match(run_dir.name)
            if not match:
                continue
            summary_path = run_dir / "results" / "markov_blanket_detection_summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            if df.empty:
                continue
            for _, row in df.iterrows():
                rows.append(
                    {
                        "dataset_name": save_dir.name.replace("save_", ""),
                        "network_name": match.group("network"),
                        "target_name": match.group("target"),
                        "n_samples": int(match.group("n_samples")),
                        "replication_id": int(match.group("rep")),
                        "method_output_name": row.get("method_output_name"),
                        "mean_recall_mb_grouped": row.get("mean_recall_mb_grouped"),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "network_name",
                "target_name",
                "n_samples",
                "replication_id",
                "method_output_name",
                "mean_recall_mb_grouped",
            ]
        )
    out = pd.DataFrame(rows)
    out["mean_recall_mb_grouped"] = pd.to_numeric(out["mean_recall_mb_grouped"], errors="coerce")
    out["n_samples"] = pd.to_numeric(out["n_samples"], errors="coerce")
    return out


def _holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)

    return adjusted.tolist()


def _sig_text(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _latex_escape(text: str) -> str:
    text = (
        str(text)
        .replace("<=", "LATEXTOKENLEQ")
        .replace(">=", "LATEXTOKENGEQ")
        .replace("<", "LATEXTOKENLT")
        .replace(">", "LATEXTOKENGT")
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    return (
        escaped.replace("LATEXTOKENLEQ", r"$\leq$")
        .replace("LATEXTOKENGEQ", r"$\geq$")
        .replace("LATEXTOKENLT", r"$<$")
        .replace("LATEXTOKENGT", r"$>$")
    )


def _paired_cohens_dz(x: pd.Series, y: pd.Series) -> float:
    diff = y.to_numpy(dtype=float) - x.to_numpy(dtype=float)
    if diff.size < 2:
        return float("nan")
    std_diff = float(np.std(diff, ddof=1))
    if np.isclose(std_diff, 0.0):
        return float("nan")
    return float(np.mean(diff) / std_diff)


def _format_caption_p_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value >= 0.1:
        return f"{value:.3f}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} \\times  10^{exponent}"


def _compute_pairwise_statistics(
    observation_scores: pd.DataFrame,
    sample_sizes: list[int],
) -> pd.DataFrame:
    # Compare mRMR-MS only against the other methods.
    pair_indices = [(0, 2), (1, 2)]
    tests_by_pair: dict[tuple[int, int], list[dict]] = {pair: [] for pair in pair_indices}

    for group_idx, n_samples in enumerate(sample_sizes):
        pivot = (
            observation_scores.loc[observation_scores["n_samples"] == n_samples]
            .pivot(
                index=["dataset_name", "network_name", "target_name", "replication_id"],
                columns="method_display",
                values="mean_recall_mb_grouped",
            )
            .reindex(columns=METHOD_ORDER)
        )
        for method_i, method_j in pair_indices:
            col_i = METHOD_ORDER[method_i]
            col_j = METHOD_ORDER[method_j]
            paired = pivot[[col_i, col_j]].dropna()
            if len(paired) < 2:
                continue
            diff = paired[col_j] - paired[col_i]
            try:
                p_value = float(wilcoxon(paired[col_i], paired[col_j], alternative="two-sided").pvalue)
            except ValueError:
                # All paired differences can be zero in degenerate cases.
                p_value = 1.0
            tests_by_pair[(method_i, method_j)].append(
                {
                    "group_idx": group_idx,
                    "n_samples": n_samples,
                    "method_i": method_i,
                    "method_j": method_j,
                    "method_i_label": col_i,
                    "method_j_label": col_j,
                    "n_pairs": int(len(paired)),
                    "mean_method_i": float(paired[col_i].mean()),
                    "mean_method_j": float(paired[col_j].mean()),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "cohens_dz": _paired_cohens_dz(paired[col_i], paired[col_j]),
                    "p_raw": p_value,
                }
            )

    tests: list[dict] = [t for pair_tests in tests_by_pair.values() for t in pair_tests]
    if not tests:
        return pd.DataFrame(
            columns=[
                "group_idx",
                "n_samples",
                "method_i",
                "method_j",
                "method_i_label",
                "method_j_label",
                "n_pairs",
                "mean_method_i",
                "mean_method_j",
                "mean_diff",
                "median_diff",
                "cohens_dz",
                "p_raw",
                "p_adj",
                "sig",
            ]
        )

    # Holm-Bonferroni per pair across sample sizes (same spirit as reference across percentages).
    for pair_tests in tests_by_pair.values():
        if not pair_tests:
            continue
        adjusted = _holm_bonferroni_adjust([t["p_raw"] for t in pair_tests])
        for t, p_adj in zip(pair_tests, adjusted):
            t["p_adj"] = p_adj
            t["sig"] = _sig_text(p_adj)

    return pd.DataFrame(tests).sort_values(["group_idx", "method_i", "method_j"]).reset_index(drop=True)


def _build_latex_table(pairwise_stats: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Pairwise Wilcoxon summary for grouped Markov blanket recall (mRMR-MS vs baselines).}",
        r"\label{tab:grouped_mb_wilcoxon}",
        r"\begin{tabular}{r l r r r r l}",
        r"\hline",
        r"Samples & Comparison & Pairs & $p_{\mathrm{raw}}$ & $p_{\mathrm{adj}}$ & $d_z$ & Sig. \\",
        r"\hline",
    ]
    for _, row in pairwise_stats.iterrows():
        comparison = _latex_escape(f"{row['method_i_label']} vs {row['method_j_label']}")
        sig = _latex_escape(row["sig"]) if row["sig"] else "-"
        dz = "nan" if pd.isna(row["cohens_dz"]) else f"{row['cohens_dz']:.3f}"
        lines.append(
            f"{int(row['n_samples'])} & {comparison} & {int(row['n_pairs'])} & "
            f"{row['p_raw']:.4g} & {row['p_adj']:.4g} & {dz} & {sig} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _print_comparison_report(observation_scores: pd.DataFrame, pairwise_stats: pd.DataFrame) -> None:
    if pairwise_stats.empty:
        print("No paired comparisons available.")
        return

    def get_stats(method_i_label: str, method_j_label: str, n_samples: int) -> pd.Series:
        matches = pairwise_stats.loc[
            (pairwise_stats["method_i_label"] == method_i_label)
            & (pairwise_stats["method_j_label"] == method_j_label)
            & (pairwise_stats["n_samples"] == n_samples)
        ]
        if matches.empty:
            raise ValueError(
                f"Missing pairwise statistics for {method_i_label} vs {method_j_label} at n={n_samples}."
            )
        return matches.iloc[0]

    smfs_250 = get_stats("Smfs", "mRMR-MS", 250)
    smfs_1000 = get_stats("Smfs", "mRMR-MS", 1000)
    smfs_5000 = get_stats("Smfs", "mRMR-MS", 5000)
    mrmr_250 = get_stats("mRMR", "mRMR-MS", 250)
    mrmr_1000 = get_stats("mRMR", "mRMR-MS", 1000)
    mrmr_5000 = get_stats("mRMR", "mRMR-MS", 5000)

    caption = (
        "Distribution of grouped Markov blanket recall across synthetic experiments for studied "
        "sample sizes, comparing {\\tt Smfs} (red), {\\tt mRMR} (blue), and {\\tt mRMR-MS} "
        "(green). Each boxplot summarizes all paired experimental units available in the synthetic "
        "benchmark. Horizontal brackets indicate statistically significant pairwise differences "
        "involving mRMR-MS, assessed with two-sided paired Wilcoxon signed-rank tests and adjusted "
        "across sample sizes for each pairwise comparison using the Holm-Bonferroni procedure. "
        f"For Smfs vs. mRMR-MS, the raw and adjusted p-values were ${_format_caption_p_value(smfs_250['p_raw'])}$ "
        f"and ${_format_caption_p_value(smfs_250['p_adj'])}$ at $n = 250$, "
        f"${_format_caption_p_value(smfs_1000['p_raw'])}$ and ${_format_caption_p_value(smfs_1000['p_adj'])}$ "
        f"at $n = 1000$, and ${_format_caption_p_value(smfs_5000['p_raw'])}$ and "
        f"${_format_caption_p_value(smfs_5000['p_adj'])}$ at $n = 5000$, with paired Cohen’s $d_z$ "
        f"values of ${smfs_250['cohens_dz']:.3f}$, ${smfs_1000['cohens_dz']:.3f}$, and "
        f"${smfs_5000['cohens_dz']:.3f}$, respectively. For mRMR vs. mRMR-MS, the corresponding "
        f"raw and adjusted p-values were ${_format_caption_p_value(mrmr_250['p_raw'])}$ and "
        f"${_format_caption_p_value(mrmr_250['p_adj'])}$ at $n = 250$, "
        f"${_format_caption_p_value(mrmr_1000['p_raw'])}$ and ${_format_caption_p_value(mrmr_1000['p_adj'])}$ "
        f"at $n = 1000$, and ${_format_caption_p_value(mrmr_5000['p_raw'])}$ and "
        f"${_format_caption_p_value(mrmr_5000['p_adj'])}$ at $n = 5000$, with paired Cohen’s $d_z$ "
        f"values of ${mrmr_250['cohens_dz']:.3f}$, ${mrmr_1000['cohens_dz']:.3f}$, and "
        f"${mrmr_5000['cohens_dz']:.3f}$, respectively."
    )
    print(caption)
    print("LaTeX table:")
    print(_build_latex_table(pairwise_stats=pairwise_stats))


def _add_significance_annotations(
    ax: plt.Axes,
    pairwise_stats: pd.DataFrame,
    centers: np.ndarray,
    offsets: np.ndarray,
    y_data_max: float,
) -> float:
    if pairwise_stats.empty:
        return y_data_max

    significant_by_group: dict[int, list[dict]] = {}
    for test in pairwise_stats.to_dict(orient="records"):
        if not test.get("sig"):
            continue
        significant_by_group.setdefault(int(test["group_idx"]), []).append(test)

    if not significant_by_group:
        return y_data_max

    bar_h = 0.018
    bar_step = 0.045
    y_base = min(0.96, y_data_max + 0.04)
    top_y = y_data_max

    for group_idx, tests_group in significant_by_group.items():
        tests_group = sorted(tests_group, key=lambda x: (x["method_j"] - x["method_i"], x["method_i"]))
        center = centers[group_idx]
        for level, test in enumerate(tests_group):
            x1 = center + offsets[test["method_i"]]
            x2 = center + offsets[test["method_j"]]
            y = y_base + level * bar_step
            ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color="black", linewidth=0.9)
            ax.text((x1 + x2) / 2.0, y + bar_h + 0.006, test["sig"], ha="center", va="bottom", fontsize=13)
            top_y = max(top_y, y + bar_h + 0.03)

    return top_y


def plot_grouped_boxplot(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_paths: list[Path] = []
    if df.empty:
        return out_paths

    plot_df = df.copy()
    plot_df["method_display"] = plot_df["method_output_name"].map(METHOD_ALIAS)
    plot_df = plot_df.loc[plot_df["method_display"].isin(METHOD_ORDER)].copy()
    plot_df["mean_recall_mb_grouped"] = pd.to_numeric(plot_df["mean_recall_mb_grouped"], errors="coerce")
    plot_df = plot_df.dropna(
        subset=[
            "n_samples",
            "dataset_name",
            "network_name",
            "target_name",
            "replication_id",
            "method_display",
            "mean_recall_mb_grouped",
        ]
    )

    if plot_df.empty:
        return out_paths

    # Use each summary CSV as one paired observation instead of averaging by dataset.
    observation_scores = (
        plot_df.groupby(
            [
                "dataset_name",
                "network_name",
                "target_name",
                "n_samples",
                "replication_id",
                "method_display",
            ],
            as_index=False,
        )["mean_recall_mb_grouped"]
        .mean()
        .copy()
    )

    sample_sizes = [n for n in [250, 1000, 5000] if n in observation_scores["n_samples"].unique()]
    if not sample_sizes:
        return out_paths

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    centers = np.arange(len(sample_sizes), dtype=float) * 2.2
    offsets = np.array([-0.42, 0.0, 0.42], dtype=float)
    box_width = 0.34
    pairwise_stats = _compute_pairwise_statistics(
        observation_scores=observation_scores,
        sample_sizes=sample_sizes,
    )
    _print_comparison_report(
        observation_scores=observation_scores,
        pairwise_stats=pairwise_stats,
    )

    y_values_all: list[float] = []

    for method_idx, method_label in enumerate(METHOD_ORDER):
        color = METHOD_COLORS[method_label]
        data_by_sample: list[np.ndarray] = []
        positions: list[float] = []

        for group_idx, n_samples in enumerate(sample_sizes):
            vals = observation_scores.loc[
                (observation_scores["n_samples"] == n_samples)
                & (observation_scores["method_display"] == method_label),
                "mean_recall_mb_grouped",
            ].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data_by_sample.append(vals)
            pos = float(centers[group_idx] + offsets[method_idx])
            positions.append(pos)
            y_values_all.extend(vals.tolist())

        if not data_by_sample:
            continue

        box = ax.boxplot(
            data_by_sample,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
            boxprops={"edgecolor": "black", "linewidth": 1.4},
            zorder=3,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.95)

    if not y_values_all:
        plt.close(fig)
        return out_paths

    y_data_max = max(y_values_all)
    y_top = _add_significance_annotations(
        ax=ax,
        pairwise_stats=pairwise_stats,
        centers=centers,
        offsets=offsets,
        y_data_max=y_data_max,
    )

    ax.set_xlim(centers[0] - 0.85, centers[-1] + 0.85)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=16)
    ax.set_xlabel("Samples", fontsize=20)
    ax.set_ylabel("Recall", fontsize=20)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=17)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=1.0, alpha=1.0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, min(1.08, max(1.0, y_top + 0.02)))

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[name], lw=6, label=name)
        for name in METHOD_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=14,
        ncol=3,
        columnspacing=1.1,
        handlelength=1.8,
    )

    fig.tight_layout()

    out_pdf = out_dir / "boxplot_recall_grouped_smfs_mrmr_mrmrms.pdf"
    out_svg = out_dir / "boxplot_recall_grouped_smfs_mrmr_mrmrms.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)
    out_paths.extend([out_pdf, out_svg])
    return out_paths


def main() -> int:
    args = parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists() or not results_root.is_dir():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (results_root / "mb_grouped_comparison_all").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_summary_rows(results_root=results_root)
    out_paths = plot_grouped_boxplot(df, plots_dir)

    print(f"[mb-plot] results_root={results_root}")
    print(f"[mb-plot] output_dir={output_dir}")
    print(f"[mb-plot] plots_dir={plots_dir}")
    if out_paths:
        print(f"[mb-plot] created {len(out_paths)} files")
    else:
        print("[mb-plot] no plots created (no data found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
