from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from local_runtime import load_dataset


METRICS = (
    "inefficiency",
    "certainty",
    "inefficiency_k",
    "certainty_k",
    "inefficiency_k_over_p",
    "certainty_k_over_p",
)
METRIC_ORDER = {
    "inefficiency": "ascending",
    "certainty": "descending",
    "inefficiency_k": "ascending",
    "certainty_k": "ascending",
    "inefficiency_k_over_p": "ascending",
    "certainty_k_over_p": "ascending",
}

INEFF_K_CANDIDATES = ("k_inefficiency", "P")
CERT_K_CANDIDATES = ("k_certainty", "P.1")


def holm_adjust_pvalues(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []

    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_pvalues = [float(pvalues[idx]) for idx in order]

    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for i, pvalue in enumerate(sorted_pvalues):
        adjusted = min(1.0, (m - i) * pvalue)
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = running_max

    adjusted = [0.0] * m
    for sorted_pos, original_pos in enumerate(order):
        adjusted[int(original_pos)] = adjusted_sorted[sorted_pos]
    return adjusted


def maximal_nonsignificant_intervals(
    ordered_methods: list[str],
    nonsignificant_pairs: dict[tuple[str, str], bool],
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    n_methods = len(ordered_methods)

    for start in range(n_methods):
        for end in range(start + 1, n_methods):
            all_nonsignificant = True
            for left in range(start, end):
                for right in range(left + 1, end + 1):
                    pair = tuple(sorted((ordered_methods[left], ordered_methods[right])))
                    if not nonsignificant_pairs.get(pair, False):
                        all_nonsignificant = False
                        break
                if not all_nonsignificant:
                    break
            if all_nonsignificant:
                intervals.append((start, end))

    maximal_intervals: list[tuple[int, int]] = []
    for interval in intervals:
        if any(
            other != interval
            and other[0] <= interval[0]
            and other[1] >= interval[1]
            for other in intervals
        ):
            continue
        maximal_intervals.append(interval)
    return maximal_intervals


def save_mean_significance_outputs(
    matrix: pd.DataFrame,
    classifier: str,
    metric: str,
    output_dir: Path,
    lower_is_better: bool,
    alpha: float = 0.05,
) -> None:
    from scipy.stats import ttest_rel

    mean_by_method = matrix.mean(axis=0)
    summary = pd.DataFrame({"mean_value": mean_by_method}).sort_values(
        by="mean_value", ascending=lower_is_better
    )
    ordered_methods = summary.index.tolist()

    raw_pvalues: list[float] = []
    pair_rows: list[dict[str, float | str | bool]] = []
    pair_keys: list[tuple[str, str]] = []

    for method_a, method_b in combinations(ordered_methods, 2):
        values_a = matrix[method_a].to_numpy(dtype=float)
        values_b = matrix[method_b].to_numpy(dtype=float)
        diffs = values_a - values_b

        if np.allclose(diffs, 0.0):
            statistic = 0.0
            pvalue = 1.0
        else:
            statistic, pvalue = ttest_rel(values_a, values_b)
            statistic = float(statistic)
            pvalue = float(pvalue)

        raw_pvalues.append(pvalue)
        pair_keys.append((method_a, method_b))
        pair_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": float(mean_by_method[method_a]),
                "mean_b": float(mean_by_method[method_b]),
                "mean_diff_a_minus_b": float(mean_by_method[method_a] - mean_by_method[method_b]),
                "t_statistic": statistic,
                "pvalue_raw": pvalue,
            }
        )

    adjusted_pvalues = holm_adjust_pvalues(raw_pvalues)
    nonsignificant_pairs: dict[tuple[str, str], bool] = {}
    for idx, adjusted_pvalue in enumerate(adjusted_pvalues):
        pair_rows[idx]["pvalue_holm"] = adjusted_pvalue
        pair_rows[idx]["significant_holm_0_05"] = bool(adjusted_pvalue < alpha)
        nonsignificant_pairs[tuple(sorted(pair_keys[idx]))] = bool(adjusted_pvalue >= alpha)

    pairwise_path = output_dir / f"pairwise_mean_tests_{classifier.lower()}_{metric}.csv"
    pd.DataFrame(pair_rows).to_csv(pairwise_path, index=False)

    intervals = maximal_nonsignificant_intervals(ordered_methods, nonsignificant_pairs)

    fig_height = max(4.8, 0.55 * len(ordered_methods) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y_positions = np.arange(len(ordered_methods))
    x_positions = summary["mean_value"].to_numpy(dtype=float)

    ax.scatter(x_positions, y_positions, color="black", s=42, zorder=3)
    for xpos, ypos, method in zip(x_positions, y_positions, ordered_methods):
        ax.hlines(y=ypos, xmin=x_positions.min(), xmax=xpos, color="#d9d9d9", linewidth=1.1, zorder=1)
        ax.text(xpos, ypos - 0.14, method, fontsize=9, ha="center", va="bottom")

    bar_base = -0.8
    for level, (start, end) in enumerate(intervals):
        ypos = bar_base - 0.22 * level
        ax.hlines(
            y=ypos,
            xmin=x_positions[start],
            xmax=x_positions[end],
            color="#1f77b4",
            linewidth=4,
            zorder=2,
        )

    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(
        f"{classifier} - {metric} mean-significance diagram\npaired t-tests + Holm correction ({direction})"
    )
    ax.set_xlabel(metric)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_ylim(len(ordered_methods) - 0.4, bar_base - 0.22 * max(len(intervals), 1) - 0.35)

    fig_path = output_dir / f"mean_significance_diagram_{classifier.lower()}_{metric}.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def ensure_autorank_importable() -> None:
    try:
        import autorank  # noqa: F401
        return
    except ModuleNotFoundError:
        fallback_python = Path("/opt/miniforge3/bin/python")
        if os.environ.get("AUTORANK_FALLBACK_DONE") == "1" or not fallback_python.exists():
            raise
        env = os.environ.copy()
        env["AUTORANK_FALLBACK_DONE"] = "1"
        cmd = [str(fallback_python), str(Path(__file__).resolve())] + sys.argv[1:]
        result = subprocess.run(cmd, env=env, check=False)
        raise SystemExit(result.returncode)


def discover_best_score_files(results_root: Path) -> list[Path]:
    files: list[Path] = []
    for save_dir in sorted(results_root.iterdir()):
        if not save_dir.is_dir():
            continue
        if "synthetic" in save_dir.name.lower():
            continue
        if not save_dir.name.startswith("save_"):
            continue
        candidates = sorted(save_dir.rglob("Best_conformal_scores*.csv"))
        if candidates:
            files.append(candidates[0])
    return files


def resolve_dataset_feature_count(
    dataset: str,
    script_dir: Path,
    data_root: str | None,
    table: pd.DataFrame,
) -> int | float:
    try:
        df, _ = load_dataset(dataset=dataset, script_dir=script_dir, data_root=data_root)
        return int(df.drop(columns=["Class"]).shape[1])
    except Exception:
        fallback_values = pd.concat(
            [
                pd.to_numeric(table.get("P"), errors="coerce"),
                pd.to_numeric(table.get("P.1"), errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        if fallback_values.empty:
            return float("nan")
        return int(fallback_values.max())


def load_results(
    files: list[Path],
    excluded_methods: set[str],
    script_dir: Path,
    data_root: str | None = None,
) -> pd.DataFrame:
    def first_available_numeric(
        table: pd.DataFrame, candidates: tuple[str, ...], output_name: str
    ) -> pd.Series:
        for candidate in candidates:
            if candidate in table.columns:
                return pd.to_numeric(table[candidate], errors="coerce")
        return pd.Series(float("nan"), index=table.index, name=output_name)

    frames: list[pd.DataFrame] = []
    for csv_path in files:
        dataset = csv_path.parent.parent.name.replace("save_", "", 1)
        table = pd.read_csv(csv_path).rename(
            columns=lambda col: "method_classifier" if str(col).startswith("Unnamed: 0") else col
        )
        table = table[table["method_classifier"].notna()].copy()
        split = table["method_classifier"].str.rsplit("-", n=1, expand=True)
        table["method"] = split[0]
        table["classifier"] = split[1]
        table["dataset"] = dataset
        dataset_n_features = resolve_dataset_feature_count(
            dataset=dataset,
            script_dir=script_dir,
            data_root=data_root,
            table=table,
        )
        table["dataset_n_features"] = pd.to_numeric(dataset_n_features, errors="coerce")
        table["inefficiency"] = pd.to_numeric(table["inefficiency"], errors="coerce")
        table["certainty"] = pd.to_numeric(table["certainty"], errors="coerce")
        table["inefficiency_k"] = first_available_numeric(table, INEFF_K_CANDIDATES, "inefficiency_k")
        table["certainty_k"] = first_available_numeric(table, CERT_K_CANDIDATES, "certainty_k")
        table["inefficiency_k_over_p"] = table["inefficiency_k"] / table["dataset_n_features"]
        table["certainty_k_over_p"] = table["certainty_k"] / table["dataset_n_features"]
        table = table[~table["method"].isin(excluded_methods)]
        frames.append(
            table[
                [
                    "dataset",
                    "classifier",
                    "method",
                    "dataset_n_features",
                    "inefficiency",
                    "certainty",
                    "inefficiency_k",
                    "certainty_k",
                    "inefficiency_k_over_p",
                    "certainty_k_over_p",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset",
                "classifier",
                "method",
                "dataset_n_features",
                "inefficiency",
                "certainty",
                "inefficiency_k",
                "certainty_k",
                "inefficiency_k_over_p",
                "certainty_k_over_p",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def run_autorank_for_metric(
    data: pd.DataFrame,
    classifier: str,
    metric: str,
    output_dir: Path,
) -> None:
    from autorank import autorank, create_report, latex_table, plot_stats

    subset = data[data["classifier"].eq(classifier)][["dataset", "method", metric]].dropna()
    if subset.empty:
        return

    matrix = subset.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    matrix = matrix.dropna(axis=0, how="any")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return

    order = METRIC_ORDER[metric]
    lower_is_better = order == "ascending"
    mean_by_method = matrix.mean(axis=0)
    mean_rank_from_raw = matrix.rank(axis=1, method="average", ascending=lower_is_better).mean(axis=0)
    best_by_mean_value = mean_by_method.idxmin() if lower_is_better else mean_by_method.idxmax()
    best_by_mean_rank = mean_rank_from_raw.idxmin()

    summary = pd.DataFrame(
        {
            "mean_value": mean_by_method,
            "mean_rank_from_raw": mean_rank_from_raw,
        }
    ).sort_values(by="mean_value", ascending=lower_is_better)
    summary_path = output_dir / f"autorank_summary_{classifier.lower()}_{metric}.csv"
    summary.to_csv(summary_path)

    if metric.endswith("_k") or metric.endswith("_k_over_p"):
        save_mean_significance_outputs(
            matrix=matrix,
            classifier=classifier,
            metric=metric,
            output_dir=output_dir,
            lower_is_better=lower_is_better,
            alpha=0.05,
        )

    # Force non-parametric workflow (Friedman + Nemenyi/Wilcoxon)
    # so plot_stats produces the classic critical difference diagram.
    autorank_stdout = io.StringIO()
    with redirect_stdout(autorank_stdout):
        result = autorank(
            matrix,
            alpha=0.05,
            verbose=False,
            order=order,
            approach="frequentist",
            force_mode="nonparametric",
        )
    pvals_shapiro = [float(p) for p in result.pvals_shapiro] if result.pvals_shapiro is not None else []
    shapiro_min = min(pvals_shapiro) if pvals_shapiro else float("nan")
    shapiro_by_method = ", ".join(
        f"{method}={pval:.4g}" for method, pval in zip(matrix.columns.tolist(), pvals_shapiro)
    )
    print(
        (
            f"[autorank] classifier={classifier} metric={metric} datasets={matrix.shape[0]} methods={matrix.shape[1]} | "
            f"omnibus={result.omnibus} p={float(result.pvalue):.6g} alpha={result.alpha} | "
            f"posthoc={result.posthoc} cd={float(result.cd):.6g} | "
            f"normality_all={bool(result.all_normal)} shapiro_min_p={shapiro_min:.6g} | "
            f"homoscedastic={bool(result.homoscedastic)} homogeneity_p={float(result.pval_homogeneity):.6g} | "
            f"best_mean_value={best_by_mean_value} ({float(mean_by_method[best_by_mean_value]):.6g}) | "
            f"best_mean_rank={best_by_mean_rank} ({float(mean_rank_from_raw[best_by_mean_rank]):.6g})"
        )
    )
    if shapiro_by_method:
        print(f"[autorank] shapiro_p_by_method ({classifier}/{metric}): {shapiro_by_method}")

    plt.figure(figsize=(10, 5))
    try:
        # Match the classic autorank behavior from the documentation.
        plot_stats(result)
    except ValueError as exc:
        # autorank can refuse to plot when the omnibus test is not significant.
        if "allow_insignificant" not in str(exc):
            raise
        plot_stats(result, allow_insignificant=True)
    fig_path = output_dir / f"critical_diagram_{classifier.lower()}_{metric}.png"
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()

    rank_path = output_dir / f"autorank_rankdf_{classifier.lower()}_{metric}.csv"
    result.rankdf.to_csv(rank_path)

    latex_buf = io.StringIO()
    with redirect_stdout(latex_buf):
        latex_table(result)
    latex_path = output_dir / f"autorank_table_{classifier.lower()}_{metric}.tex"
    latex_path.write_text(latex_buf.getvalue())

    report_buf = io.StringIO()
    with redirect_stdout(report_buf):
        create_report(result)
    report_path = output_dir / f"autorank_report_{classifier.lower()}_{metric}.txt"
    report_path.write_text(report_buf.getvalue())


def main() -> None:
    ensure_autorank_importable()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("RESULTS"),
        help="Root folder containing save_<dataset> subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_conformal_comparison") / "autorank",
        help="Destination folder for critical diagrams and reports.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=["JMI_MS_linear"],
        help="Method name to exclude from the analysis. Can be repeated.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional dataset root used to resolve p when computing k/p.",
    )
    args = parser.parse_args()

    files = discover_best_score_files(args.results_root)
    if not files:
        raise RuntimeError(f"No Best_conformal_scores files found in {args.results_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(
        files,
        set(args.exclude_method),
        script_dir=Path(__file__).resolve().parent,
        data_root=str(args.data_root) if args.data_root is not None else None,
    )

    classifiers = sorted(results["classifier"].dropna().unique())
    for classifier in classifiers:
        for metric in METRICS:
            run_autorank_for_metric(results, classifier, metric, args.output_dir)

    print(f"Autorank analysis completed. Outputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
