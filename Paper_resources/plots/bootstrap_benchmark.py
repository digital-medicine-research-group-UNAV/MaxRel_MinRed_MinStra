from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGET = "mRMR_MS_linear"
DEFAULT_EXCLUDED_DATASETS = ("parkinson",)
DEFAULT_EXCLUDED_METHODS = ("JMI_MS_linear",)
DEFAULT_METRICS = ("inefficiency", "certainty")
DEFAULT_CLASSIFIERS = ("SVM", "KNN")

METRIC_LOWER_IS_BETTER = {
    "inefficiency": True,
    "certainty": False,
}

METRIC_LABELS = {
    "inefficiency": "Inefficiency",
    "certainty": "Certainty",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap dataset-level benchmark differences against a target method. "
            "The script reads Best_conformal_scores*.csv files, computes paired "
            "method differences and reports percentile 95% confidence intervals."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("RESULTS"),
        help="Root folder containing save_<dataset> subfolders.",
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        default="pruebas_revision",
        help="Run folder inside each save_<dataset> directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_conformal_comparison") / "bootstrap",
        help="Destination folder for bootstrap CSV outputs.",
    )
    parser.add_argument(
        "--target-method",
        type=str,
        default=DEFAULT_TARGET,
        help="Reference method. Positive deltas favor this method.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=list(DEFAULT_EXCLUDED_DATASETS),
        help="Dataset to exclude. Can be repeated. Defaults to parkinson.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=list(DEFAULT_EXCLUDED_METHODS),
        help="Method to exclude. Can be repeated. Defaults to JMI_MS_linear.",
    )
    parser.add_argument(
        "--classifier",
        action="append",
        choices=DEFAULT_CLASSIFIERS,
        default=None,
        help="Classifier to include. Can be repeated. Defaults to SVM and KNN.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=DEFAULT_METRICS,
        default=None,
        help="Metric to include. Can be repeated. Defaults to inefficiency and certainty.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducible resampling.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided interval level. alpha=0.05 gives a 95% interval.",
    )
    return parser.parse_args()


def discover_best_score_files(results_root: Path, run_folder: str) -> list[Path]:
    files: list[Path] = []
    for save_dir in sorted(results_root.iterdir()):
        if not save_dir.is_dir():
            continue
        if not save_dir.name.startswith("save_"):
            continue
        if "synthetic" in save_dir.name.lower():
            continue

        run_dir = save_dir / run_folder
        candidates = sorted(run_dir.glob("Best_conformal_scores*.csv"))
        if candidates:
            files.append(candidates[0])
    return files


def load_score_tables(
    files: list[Path],
    excluded_datasets: set[str],
    excluded_methods: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in files:
        dataset = csv_path.parent.parent.name.replace("save_", "", 1)
        if dataset in excluded_datasets:
            continue

        table = pd.read_csv(csv_path)
        table = table.rename(
            columns=lambda col: "method_classifier" if str(col).startswith("Unnamed") else col
        )
        table = table.loc[table["method_classifier"].notna()].copy()

        split = table["method_classifier"].str.rsplit("-", n=1, expand=True)
        table["method"] = split[0]
        table["classifier"] = split[1]
        table["dataset"] = dataset
        table["k_inefficiency"] = pd.to_numeric(table.get("P"), errors="coerce")
        table["k_certainty"] = pd.to_numeric(table.get("P.1"), errors="coerce")
        table["inefficiency"] = pd.to_numeric(table["inefficiency"], errors="coerce")
        table["certainty"] = pd.to_numeric(table["certainty"], errors="coerce")
        table = table.loc[~table["method"].isin(excluded_methods)].copy()

        frames.append(
            table[
                [
                    "dataset",
                    "classifier",
                    "method",
                    "k_inefficiency",
                    "inefficiency",
                    "k_certainty",
                    "certainty",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset",
                "classifier",
                "method",
                "k_inefficiency",
                "inefficiency",
                "k_certainty",
                "certainty",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, alpha: float, rng: np.random.Generator):
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    sample_indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    bootstrap_values = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (
        float(values.mean()),
        float(lower),
        float(upper),
        float((bootstrap_values > 0.0).mean()),
    )


def win_tie_loss(values: np.ndarray) -> tuple[int, int, int, str]:
    wins = int((values > 0.0).sum())
    ties = int(np.isclose(values, 0.0).sum())
    losses = int((values < 0.0).sum())
    return wins, ties, losses, f"{wins}/{ties}/{losses}"


def summarize_comparison(
    data: pd.DataFrame,
    classifier: str,
    metric: str,
    target_method: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame, pd.DataFrame]:
    lower_is_better = METRIC_LOWER_IS_BETTER[metric]
    pivot = data.loc[data["classifier"].eq(classifier)].pivot_table(
        index="dataset",
        columns="method",
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    pivot = pivot.dropna(axis=0, how="any")

    if target_method not in pivot.columns:
        raise ValueError(f"Target method '{target_method}' is missing for {classifier}/{metric}.")

    ranks = pivot.rank(axis=1, ascending=lower_is_better, method="average")
    rows: list[dict[str, float | int | str]] = []

    baselines = [method for method in pivot.columns if method != target_method]
    for baseline in baselines:
        if lower_is_better:
            score_delta = pivot[baseline] - pivot[target_method]
        else:
            score_delta = pivot[target_method] - pivot[baseline]
        rank_delta = ranks[baseline] - ranks[target_method]

        mean_score, low_score, high_score, p_score = bootstrap_mean(
            score_delta.to_numpy(dtype=float), n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
        )
        mean_rank, low_rank, high_rank, p_rank = bootstrap_mean(
            rank_delta.to_numpy(dtype=float), n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
        )
        wins, ties, losses, wtl = win_tie_loss(score_delta.to_numpy(dtype=float))

        rows.append(
            {
                "classifier": classifier,
                "metric": metric,
                "baseline_method": baseline,
                "target_method": target_method,
                "n_datasets": int(pivot.shape[0]),
                "datasets": ";".join(pivot.index.astype(str).tolist()),
                "mean_delta_score": mean_score,
                "ci_low_delta_score": low_score,
                "ci_high_delta_score": high_score,
                "bootstrap_p_delta_score_gt_0": p_score,
                "mean_delta_rank": mean_rank,
                "ci_low_delta_rank": low_rank,
                "ci_high_delta_rank": high_rank,
                "bootstrap_p_delta_rank_gt_0": p_rank,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_tie_loss": wtl,
                "direction": "positive deltas favor target_method",
            }
        )

    return rows, pivot, ranks


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_float(value: float, digits: int) -> str:
    if pd.isna(value):
        return "--"
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def format_interval(low: float, high: float, digits: int) -> str:
    return f"[{format_float(low, digits)}, {format_float(high, digits)}]"


def render_latex_table(
    table: pd.DataFrame,
    classifier: str,
    metric: str,
    target_method: str,
    confidence_level: float,
) -> str:
    label_classifier = classifier.lower()
    label_metric = metric.lower()
    confidence_text = format_float(confidence_level, 0)
    caption = (
        f"Bootstrap comparison for {classifier} {METRIC_LABELS[metric].lower()}. "
        f"Positive deltas favor {target_method}; W/T/L reports target wins, ties, "
        "and losses across datasets."
    )
    label = f"tab:bootstrap_{label_classifier}_{label_metric}"

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        (
            r"Baseline & "
            r"$\Delta_{\mathrm{score}}$ & "
            rf"{confidence_text}\% BI$_{{\mathrm{{score}}}}$ & "
            r"$\Pr_B(\Delta_{\mathrm{score}}>0)$ & "
            r"$\Delta_{\mathrm{rank}}$ & "
            rf"{confidence_text}\% BI$_{{\mathrm{{rank}}}}$ & "
            r"$\Pr_B(\Delta_{\mathrm{rank}}>0)$ & "
            r"W/T/L \\"
        ),
        r"\midrule",
    ]

    for _, row in table.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["baseline_method"]),
                    format_float(row["mean_delta_score"], 3),
                    format_interval(row["ci_low_delta_score"], row["ci_high_delta_score"], 3),
                    format_float(row["bootstrap_p_delta_score_gt_0"], 3),
                    format_float(row["mean_delta_rank"], 2),
                    format_interval(row["ci_low_delta_rank"], row["ci_high_delta_rank"], 2),
                    format_float(row["bootstrap_p_delta_rank_gt_0"], 3),
                    latex_escape(row["win_tie_loss"]),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_latex_tables(
    summary: pd.DataFrame,
    output_dir: Path,
    classifiers: list[str],
    metrics: list[str],
    target_method: str,
    confidence_level: float,
) -> None:
    combined_tables: list[str] = []
    for classifier in classifiers:
        for metric in metrics:
            table = summary.loc[
                summary["classifier"].eq(classifier) & summary["metric"].eq(metric)
            ].copy()
            if table.empty:
                continue
            latex = render_latex_table(
                table=table,
                classifier=classifier,
                metric=metric,
                target_method=target_method,
                confidence_level=confidence_level,
            )
            filename = f"bootstrap_table_{classifier.lower()}_{metric}.tex"
            (output_dir / filename).write_text(latex)
            combined_tables.append(latex)

    if combined_tables:
        (output_dir / "bootstrap_benchmark_tables.tex").write_text("\n".join(combined_tables))


def main() -> None:
    args = parse_arguments()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    classifiers = args.classifier or list(DEFAULT_CLASSIFIERS)
    metrics = args.metric or list(DEFAULT_METRICS)
    rng = np.random.default_rng(args.seed)

    files = discover_best_score_files(args.results_root, run_folder=args.run_folder)
    if not files:
        raise RuntimeError(f"No Best_conformal_scores*.csv files found in {args.results_root}.")

    data = load_score_tables(
        files=files,
        excluded_datasets=set(args.exclude_dataset),
        excluded_methods=set(args.exclude_method),
    )
    if data.empty:
        raise RuntimeError("No benchmark rows left after applying exclusions.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "bootstrap_input_scores.csv", index=False)

    summary_rows: list[dict[str, float | int | str]] = []
    for classifier in classifiers:
        for metric in metrics:
            rows, pivot, ranks = summarize_comparison(
                data=data,
                classifier=classifier,
                metric=metric,
                target_method=args.target_method,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            summary_rows.extend(rows)
            pivot.to_csv(args.output_dir / f"metric_matrix_{classifier.lower()}_{metric}.csv")
            ranks.to_csv(args.output_dir / f"rank_matrix_{classifier.lower()}_{metric}.csv")

    summary = pd.DataFrame(summary_rows).sort_values(
        ["classifier", "metric", "mean_delta_rank"],
        ascending=[True, True, False],
    )
    summary.to_csv(args.output_dir / "bootstrap_benchmark_summary.csv", index=False)
    write_latex_tables(
        summary=summary,
        output_dir=args.output_dir,
        classifiers=classifiers,
        metrics=metrics,
        target_method=args.target_method,
        confidence_level=100.0 * (1.0 - args.alpha),
    )

    datasets = sorted(data["dataset"].dropna().unique().tolist())
    print(f"Loaded {len(datasets)} datasets: {', '.join(datasets)}")
    print(f"Bootstrap summary saved to: {args.output_dir / 'bootstrap_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
