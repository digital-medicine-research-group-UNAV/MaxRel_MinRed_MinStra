#!/usr/bin/env python3
"""
Plot grouped Markov-blanket comparison results across all datasets,
stratified by Markov blanket size into two side-by-side grouped boxplots.

Usage:
  pixi run python DATA/synthetic/plot_mrmr_stratified.py
  pixi run python DATA/synthetic/plot_mrmr_stratified.py --mb_threshold 5
  pixi run python DATA/synthetic/plot_mrmr_stratified.py --results_root /path/to/RESULTS/synthetic
  pixi run python DATA/synthetic/plot_mrmr_stratified.py --output_dir /path/to/output

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
        description=(
            "Plot grouped MB comparison by method and sample size across all datasets, "
            "stratified by Markov blanket size."
        )
    )
    parser.add_argument(
        "--results_root",
        default="/mnt/storage/mlopezdecas/MRMR/new_experiments/RESULTS/synthetic",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output dir. Default: <results_root>/mb_grouped_comparison_stratified",
    )
    parser.add_argument(
        "--mb_threshold",
        type=int,
        default=5,
        help="Threshold for MB size stratification. Left panel: <= threshold, right panel: > threshold.",
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
                        "mb_size": row.get("mb_size"),
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
                "mb_size",
                "mean_recall_mb_grouped",
            ]
        )
    out = pd.DataFrame(rows)
    out["mean_recall_mb_grouped"] = pd.to_numeric(out["mean_recall_mb_grouped"], errors="coerce")
    out["n_samples"] = pd.to_numeric(out["n_samples"], errors="coerce")
    out["mb_size"] = pd.to_numeric(out["mb_size"], errors="coerce")
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

    for pair_tests in tests_by_pair.values():
        if not pair_tests:
            continue
        adjusted = _holm_bonferroni_adjust([t["p_raw"] for t in pair_tests])
        for t, p_adj in zip(pair_tests, adjusted):
            t["p_adj"] = p_adj
            t["sig"] = _sig_text(p_adj)

    return pd.DataFrame(tests).sort_values(["group_idx", "method_i", "method_j"]).reset_index(drop=True)


def _build_latex_table(pairwise_stats: pd.DataFrame, stratum_label: str) -> str:
    caption = _latex_escape(f"{stratum_label}: pairwise Wilcoxon summary (mRMR-MS vs baselines).")
    label_source = (
        stratum_label.lower()
        .replace("<=", "_le_")
        .replace(">=", "_ge_")
        .replace("<", "_lt_")
        .replace(">", "_gt_")
    )
    label = re.sub(r"[^a-z0-9]+", "_", label_source).strip("_")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}_wilcoxon}}",
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


def _print_comparison_report(
    pairwise_stats: pd.DataFrame,
    sample_sizes: list[int],
    stratum_label: str,
) -> None:
    if pairwise_stats.empty:
        print(f"[{stratum_label}] No paired comparisons available.")
        return

    print(f"[{stratum_label}] Pairwise Wilcoxon summary (mRMR-MS vs baselines):")
    for n_samples in sample_sizes:
        chunk = pairwise_stats.loc[pairwise_stats["n_samples"] == n_samples].copy()
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            print(
                f"[{stratum_label}] n={int(n_samples)} | {row['method_i_label']} vs {row['method_j_label']} | "
                f"pairs={int(row['n_pairs'])} | p_raw={row['p_raw']:.4g} | p_adj={row['p_adj']:.4g} | "
                f"d_z={row['cohens_dz']:.3f} | sig='{row['sig']}'"
            )
    print(f"[{stratum_label}] LaTeX table:")
    print(_build_latex_table(pairwise_stats=pairwise_stats, stratum_label=stratum_label))


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


def _prepare_observations(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["method_display"] = plot_df["method_output_name"].map(METHOD_ALIAS)
    plot_df = plot_df.loc[plot_df["method_display"].isin(METHOD_ORDER)].copy()
    plot_df["mean_recall_mb_grouped"] = pd.to_numeric(plot_df["mean_recall_mb_grouped"], errors="coerce")
    plot_df["mb_size"] = pd.to_numeric(plot_df["mb_size"], errors="coerce")
    plot_df = plot_df.dropna(
        subset=[
            "n_samples",
            "dataset_name",
            "network_name",
            "target_name",
            "replication_id",
            "method_display",
            "mb_size",
            "mean_recall_mb_grouped",
        ]
    )

    if plot_df.empty:
        return plot_df

    # Use each summary CSV as one paired observation instead of averaging by dataset.
    return (
        plot_df.groupby(
            [
                "dataset_name",
                "network_name",
                "target_name",
                "n_samples",
                "replication_id",
                "method_display",
                "mb_size",
            ],
            as_index=False,
        )["mean_recall_mb_grouped"]
        .mean()
        .copy()
    )


def _plot_single_stratum(
    ax: plt.Axes,
    observation_scores: pd.DataFrame,
    title: str,
) -> tuple[bool, float]:
    sample_sizes = [n for n in [250, 1000, 5000] if n in observation_scores["n_samples"].unique()]
    if not sample_sizes:
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.set_title(title, fontsize=17, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        return False, 1.0

    centers = np.arange(len(sample_sizes), dtype=float) * 2.2
    offsets = np.array([-0.42, 0.0, 0.42], dtype=float)
    box_width = 0.34

    pairwise_stats = _compute_pairwise_statistics(
        observation_scores=observation_scores,
        sample_sizes=sample_sizes,
    )
    _print_comparison_report(
        pairwise_stats=pairwise_stats,
        sample_sizes=sample_sizes,
        stratum_label=title,
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
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.set_title(title, fontsize=17, pad=12)
        return False, 1.0

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
    ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=15)
    ax.set_xlabel("Samples", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(axis="y", color="#E6E6E6", linewidth=1.0, alpha=1.0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=17, pad=12)

    return True, y_top


def plot_grouped_boxplot_stratified(df: pd.DataFrame, out_dir: Path, mb_threshold: int) -> list[Path]:
    out_paths: list[Path] = []
    if df.empty:
        return out_paths

    observation_scores = _prepare_observations(df)
    if observation_scores.empty:
        return out_paths

    low_mb = observation_scores.loc[observation_scores["mb_size"] <= mb_threshold].copy()
    high_mb = observation_scores.loc[observation_scores["mb_size"] > mb_threshold].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.1), sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    left_title = f"MB size <= {mb_threshold}"
    right_title = f"MB size > {mb_threshold}"

    left_ok, left_top = _plot_single_stratum(
        ax=axes[0],
        observation_scores=low_mb,
        title=left_title,
    )
    right_ok, right_top = _plot_single_stratum(
        ax=axes[1],
        observation_scores=high_mb,
        title=right_title,
    )

    axes[0].set_ylabel("Recall", fontsize=19)

    any_data = left_ok or right_ok
    if not any_data:
        plt.close(fig)
        return out_paths

    common_top = min(1.08, max(1.0, left_top, right_top) + 0.02)
    for ax, ok in zip(axes, [left_ok, right_ok]):
        if ok:
            ax.set_ylim(-0.03, common_top)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[name], lw=6, label=name)
        for name in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        frameon=False,
        fontsize=14,
        ncol=3,
        columnspacing=1.2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
 
    out_pdf = out_dir / f"boxplot_recall_grouped_smfs_mrmr_mrmrms_stratified_mbthr_{mb_threshold}.pdf"
    out_svg = out_dir / f"boxplot_recall_grouped_smfs_mrmr_mrmrms_stratified_mbthr_{mb_threshold}.svg"
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
        else (results_root / "mb_grouped_comparison_stratified").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_summary_rows(results_root=results_root)
    out_paths = plot_grouped_boxplot_stratified(df, plots_dir, args.mb_threshold)

    print(f"[mb-plot] results_root={results_root}")
    print(f"[mb-plot] output_dir={output_dir}")
    print(f"[mb-plot] plots_dir={plots_dir}")
    print(f"[mb-plot] mb_threshold={args.mb_threshold}")
    if out_paths:
        print(f"[mb-plot] created {len(out_paths)} files")
    else:
        print("[mb-plot] no plots created (no data found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
