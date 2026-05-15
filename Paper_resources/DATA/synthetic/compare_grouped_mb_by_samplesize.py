#!/usr/bin/env python3
"""
Compare grouped Markov-blanket detection performance across methods,
stratified by sample size.

Usage:
  pixi run python DATA/synthetic/compare_grouped_mb_by_samplesize.py --dataset_name healthcare

Outputs (default):
  RESULTS/synthetic/save_<dataset_name>/mb_grouped_comparison/
    - per_csv_method_grouped.csv
    - summary_by_samplesize_method.csv
    - summary_overall_method.csv
    - pivot_mean_recall_grouped.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FOLDER_PATTERN = re.compile(
    r"^(?P<network>.+?)__target_(?P<target>.+?)__n_(?P<n_samples>\d+)__rep_(?P<rep>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare grouped MB detection metrics across methods and sample sizes."
    )
    parser.add_argument("--dataset_name", required=True, help="Synthetic dataset name (e.g., healthcare).")
    parser.add_argument(
        "--results_root",
        default="/mnt/storage/mlopezdecas/MRMR/new_experiments/RESULTS/synthetic",
        help="Root containing save_<dataset_name> folders.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output dir. Default: <results_root>/save_<dataset>/mb_grouped_comparison",
    )
    return parser.parse_args()


def discover_rows(save_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

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
                    "network_name": match.group("network"),
                    "target_name": match.group("target"),
                    "n_samples": int(match.group("n_samples")),
                    "replication_id": int(match.group("rep")),
                    "run_folder": run_dir.name,
                    "method_output_name": row.get("method_output_name"),
                    "n_folds": row.get("n_folds"),
                    "mb_size": row.get("mb_size"),
                    "mean_recall_mb_grouped": row.get("mean_recall_mb_grouped"),
                    "std_recall_mb_grouped": row.get("std_recall_mb_grouped"),
                    "mean_precision_mb_grouped": row.get("mean_precision_mb_grouped"),
                    "std_precision_mb_grouped": row.get("std_precision_mb_grouped"),
                    "exact_match_rate_grouped": row.get("exact_match_rate_grouped"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "network_name",
                "target_name",
                "n_samples",
                "replication_id",
                "run_folder",
                "method_output_name",
                "n_folds",
                "mb_size",
                "mean_recall_mb_grouped",
                "std_recall_mb_grouped",
                "mean_precision_mb_grouped",
                "std_precision_mb_grouped",
                "exact_match_rate_grouped",
            ]
        )

    out = pd.DataFrame(rows)
    numeric_cols = [
        "n_folds",
        "mb_size",
        "mean_recall_mb_grouped",
        "std_recall_mb_grouped",
        "mean_precision_mb_grouped",
        "std_precision_mb_grouped",
        "exact_match_rate_grouped",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize_by_samplesize_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "n_samples",
                "method_output_name",
                "n_datasets",
                "mean_recall_mb_grouped",
                "std_recall_mb_grouped",
                "mean_precision_mb_grouped",
                "std_precision_mb_grouped",
                "mean_exact_match_rate_grouped",
                "std_exact_match_rate_grouped",
                "mean_mb_size",
            ]
        )

    grouped = (
        df.groupby(["n_samples", "method_output_name"], dropna=False)
        .agg(
            n_datasets=("run_folder", "count"),
            mean_recall_mb_grouped=("mean_recall_mb_grouped", "mean"),
            std_recall_mb_grouped=("mean_recall_mb_grouped", "std"),
            mean_precision_mb_grouped=("mean_precision_mb_grouped", "mean"),
            std_precision_mb_grouped=("mean_precision_mb_grouped", "std"),
            mean_exact_match_rate_grouped=("exact_match_rate_grouped", "mean"),
            std_exact_match_rate_grouped=("exact_match_rate_grouped", "std"),
            mean_mb_size=("mb_size", "mean"),
        )
        .reset_index()
        .sort_values(["n_samples", "mean_recall_mb_grouped", "mean_exact_match_rate_grouped"], ascending=[True, False, False])
    )
    return grouped


def summarize_overall_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method_output_name",
                "n_datasets",
                "mean_recall_mb_grouped",
                "std_recall_mb_grouped",
                "mean_precision_mb_grouped",
                "std_precision_mb_grouped",
                "mean_exact_match_rate_grouped",
                "std_exact_match_rate_grouped",
            ]
        )

    grouped = (
        df.groupby("method_output_name", dropna=False)
        .agg(
            n_datasets=("run_folder", "count"),
            mean_recall_mb_grouped=("mean_recall_mb_grouped", "mean"),
            std_recall_mb_grouped=("mean_recall_mb_grouped", "std"),
            mean_precision_mb_grouped=("mean_precision_mb_grouped", "mean"),
            std_precision_mb_grouped=("mean_precision_mb_grouped", "std"),
            mean_exact_match_rate_grouped=("exact_match_rate_grouped", "mean"),
            std_exact_match_rate_grouped=("exact_match_rate_grouped", "std"),
        )
        .reset_index()
        .sort_values(["mean_recall_mb_grouped", "mean_exact_match_rate_grouped"], ascending=[False, False])
    )
    return grouped


def main() -> int:
    args = parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    save_dir = results_root / f"save_{args.dataset_name}"
    if not save_dir.exists() or not save_dir.is_dir():
        raise FileNotFoundError(f"save folder not found: {save_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (save_dir / "mb_grouped_comparison").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_csv = discover_rows(save_dir=save_dir)
    by_sample = summarize_by_samplesize_method(per_csv)
    overall = summarize_overall_method(per_csv)

    per_csv_path = output_dir / "per_csv_method_grouped.csv"
    by_sample_path = output_dir / "summary_by_samplesize_method.csv"
    overall_path = output_dir / "summary_overall_method.csv"
    pivot_path = output_dir / "pivot_mean_recall_grouped.csv"

    per_csv.to_csv(per_csv_path, index=False)
    by_sample.to_csv(by_sample_path, index=False)
    overall.to_csv(overall_path, index=False)

    if not by_sample.empty:
        pivot = by_sample.pivot(index="method_output_name", columns="n_samples", values="mean_recall_mb_grouped")
        pivot = pivot.sort_index()
        pivot.to_csv(pivot_path)
    else:
        pd.DataFrame().to_csv(pivot_path, index=False)

    print(f"[mb-compare] dataset={args.dataset_name}")
    print(f"[mb-compare] discovered_rows={len(per_csv)}")
    print(f"[mb-compare] output_dir={output_dir}")
    print(f"[mb-compare] written: {per_csv_path.name}, {by_sample_path.name}, {overall_path.name}, {pivot_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
