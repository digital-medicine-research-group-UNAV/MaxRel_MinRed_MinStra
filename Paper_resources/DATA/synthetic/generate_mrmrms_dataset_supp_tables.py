#!/usr/bin/env python3
"""
Generate supplementary dataset-level LaTeX tables that complement the
mRMR stratified comparison plots.

The script reads:
  RESULTS/synthetic/save_<dataset>/<run_folder>/results/markov_blanket_detection_summary.csv

and produces dataset-level supplementary tables (LaTeX) under:
  <results_root>/mb_grouped_comparison_stratified/supplementary_tables_mrmrms_dataset_level

Usage:
  pixi run python DATA/synthetic/generate_mrmrms_dataset_supp_tables.py
  pixi run python DATA/synthetic/generate_mrmrms_dataset_supp_tables.py --mb_threshold 5
  pixi run python DATA/synthetic/generate_mrmrms_dataset_supp_tables.py --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
PAIRWISE_COMPARISONS = [("Smfs", "mRMR-MS"), ("mRMR", "mRMR-MS")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate supplementary dataset-level LaTeX tables for mRMR-MS advantages "
            "across datasets, sample sizes and MB strata."
        )
    )
    parser.add_argument(
        "--results_root",
        default="/mnt/storage/mlopezdecas/MRMR/new_experiments/RESULTS/synthetic",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional output dir. Default: "
            "<results_root>/mb_grouped_comparison_stratified/supplementary_tables_mrmrms_dataset_level"
        ),
    )
    parser.add_argument(
        "--mb_threshold",
        type=int,
        default=5,
        help="Threshold for MB-size stratification (<= threshold, > threshold).",
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance used to count ties in win/tie/loss summaries.",
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


def _prepare_observations(df: pd.DataFrame, mb_threshold: int) -> pd.DataFrame:
    obs = df.copy()
    obs["method_display"] = obs["method_output_name"].map(METHOD_ALIAS)
    obs = obs.loc[obs["method_display"].isin(METHOD_ORDER)].copy()
    obs["mean_recall_mb_grouped"] = pd.to_numeric(obs["mean_recall_mb_grouped"], errors="coerce")
    obs["mb_size"] = pd.to_numeric(obs["mb_size"], errors="coerce")
    obs = obs.dropna(
        subset=[
            "dataset_name",
            "network_name",
            "target_name",
            "n_samples",
            "replication_id",
            "method_display",
            "mb_size",
            "mean_recall_mb_grouped",
        ]
    )
    if obs.empty:
        return obs

    obs = (
        obs.groupby(
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

    obs["mb_stratum"] = np.where(
        obs["mb_size"] <= mb_threshold,
        f"<= {mb_threshold}",
        f"> {mb_threshold}",
    )
    return obs


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


def _paired_cohens_dz(x: pd.Series, y: pd.Series) -> float:
    diff = y.to_numpy(dtype=float) - x.to_numpy(dtype=float)
    if diff.size < 2:
        return float("nan")
    std_diff = float(np.std(diff, ddof=1))
    if np.isclose(std_diff, 0.0):
        return float("nan")
    return float(np.mean(diff) / std_diff)


def _compute_pairwise_records(observations: pd.DataFrame) -> pd.DataFrame:
    """Unit-level paired differences used by downstream tables."""
    rows: list[dict] = []

    for (dataset_name, n_samples, mb_stratum), chunk in observations.groupby(
        ["dataset_name", "n_samples", "mb_stratum"], sort=False
    ):
        pivot = (
            chunk.pivot(
                index=["network_name", "target_name", "replication_id", "mb_size"],
                columns="method_display",
                values="mean_recall_mb_grouped",
            )
            .reindex(columns=METHOD_ORDER)
            .reset_index()
        )

        for baseline, challenger in PAIRWISE_COMPARISONS:
            paired = pivot[[baseline, challenger]].dropna().copy()
            if paired.empty:
                continue
            diff = paired[challenger] - paired[baseline]
            for value in diff.to_numpy(dtype=float):
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "n_samples": int(n_samples),
                        "mb_stratum": mb_stratum,
                        "comparison": f"{challenger} vs {baseline}",
                        "baseline": baseline,
                        "challenger": challenger,
                        "delta": float(value),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "n_samples",
                "mb_stratum",
                "comparison",
                "baseline",
                "challenger",
                "delta",
            ]
        )

    return pd.DataFrame(rows)


def build_table_s1(observations: pd.DataFrame) -> pd.DataFrame:
    means = (
        observations.groupby(["dataset_name", "n_samples", "mb_stratum", "method_display"], as_index=False)[
            "mean_recall_mb_grouped"
        ]
        .mean()
        .rename(columns={"mean_recall_mb_grouped": "mean_recall"})
    )

    medians = (
        observations.groupby(["dataset_name", "n_samples", "mb_stratum", "method_display"], as_index=False)[
            "mean_recall_mb_grouped"
        ]
        .median()
        .rename(columns={"mean_recall_mb_grouped": "median_recall"})
    )

    means_wide = means.pivot_table(
        index=["dataset_name", "n_samples", "mb_stratum"],
        columns="method_display",
        values="mean_recall",
    ).reset_index()

    medians_wide = medians.pivot_table(
        index=["dataset_name", "n_samples", "mb_stratum"],
        columns="method_display",
        values="median_recall",
    ).reset_index()

    triplets = (
        observations.pivot_table(
            index=[
                "dataset_name",
                "n_samples",
                "mb_stratum",
                "network_name",
                "target_name",
                "replication_id",
                "mb_size",
            ],
            columns="method_display",
            values="mean_recall_mb_grouped",
        )
        .reindex(columns=METHOD_ORDER)
        .dropna()
        .reset_index()
        .groupby(["dataset_name", "n_samples", "mb_stratum"], as_index=False)
        .size()
        .rename(columns={"size": "n_triplets"})
    )

    table = means_wide.merge(
        medians_wide,
        on=["dataset_name", "n_samples", "mb_stratum"],
        suffixes=("_mean", "_median"),
    ).merge(triplets, on=["dataset_name", "n_samples", "mb_stratum"], how="left")

    table["delta_mean_mrmrms_vs_smfs"] = table["mRMR-MS_mean"] - table["Smfs_mean"]
    table["delta_mean_mrmrms_vs_mrmr"] = table["mRMR-MS_mean"] - table["mRMR_mean"]
    table["delta_median_mrmrms_vs_smfs"] = table["mRMR-MS_median"] - table["Smfs_median"]
    table["delta_median_mrmrms_vs_mrmr"] = table["mRMR-MS_median"] - table["mRMR_median"]

    table = table[
        [
            "dataset_name",
            "n_samples",
            "mb_stratum",
            "n_triplets",
            "Smfs_mean",
            "mRMR_mean",
            "mRMR-MS_mean",
            "delta_mean_mrmrms_vs_smfs",
            "delta_mean_mrmrms_vs_mrmr",
            "Smfs_median",
            "mRMR_median",
            "mRMR-MS_median",
            "delta_median_mrmrms_vs_smfs",
            "delta_median_mrmrms_vs_mrmr",
        ]
    ].sort_values(["dataset_name", "n_samples", "mb_stratum"])

    return table.reset_index(drop=True)


def build_table_s2(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (dataset_name, n_samples, mb_stratum), chunk in observations.groupby(
        ["dataset_name", "n_samples", "mb_stratum"], sort=False
    ):
        pivot = (
            chunk.pivot(
                index=["network_name", "target_name", "replication_id", "mb_size"],
                columns="method_display",
                values="mean_recall_mb_grouped",
            )
            .reindex(columns=METHOD_ORDER)
        )

        for baseline, challenger in PAIRWISE_COMPARISONS:
            paired = pivot[[baseline, challenger]].dropna()
            n_pairs = int(len(paired))
            p_raw = np.nan
            dz = np.nan
            mean_delta = np.nan
            median_delta = np.nan

            if n_pairs >= 2:
                diff = paired[challenger] - paired[baseline]
                mean_delta = float(diff.mean())
                median_delta = float(diff.median())
                dz = _paired_cohens_dz(paired[baseline], paired[challenger])
                try:
                    p_raw = float(wilcoxon(paired[baseline], paired[challenger], alternative="two-sided").pvalue)
                except ValueError:
                    p_raw = 1.0

            rows.append(
                {
                    "dataset_name": dataset_name,
                    "n_samples": int(n_samples),
                    "mb_stratum": mb_stratum,
                    "comparison": f"{challenger} vs {baseline}",
                    "n_pairs": n_pairs,
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                    "cohens_dz": dz,
                    "p_raw": p_raw,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "n_samples",
                "mb_stratum",
                "comparison",
                "n_pairs",
                "mean_delta",
                "median_delta",
                "cohens_dz",
                "p_raw",
                "p_adj",
                "sig",
            ]
        )

    table = pd.DataFrame(rows)
    table["p_adj"] = np.nan

    for (_, mb_stratum, comparison), idx in table.groupby(
        ["dataset_name", "mb_stratum", "comparison"]
    ).groups.items():
        sub = table.loc[idx]
        valid = sub["p_raw"].dropna()
        if valid.empty:
            continue
        adjusted = _holm_bonferroni_adjust(valid.tolist())
        table.loc[valid.index, "p_adj"] = adjusted

    table["sig"] = ""
    table.loc[table["p_adj"] < 0.05, "sig"] = "*"
    table.loc[table["p_adj"] < 0.01, "sig"] = "**"

    return table.sort_values(["dataset_name", "n_samples", "mb_stratum", "comparison"]).reset_index(drop=True)


def build_table_s3(pairwise_unit: pd.DataFrame, tie_tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairwise_unit.empty:
        empty_global = pd.DataFrame(
            columns=["dataset_name", "comparison", "wins", "ties", "losses", "win_rate_pct"]
        )
        empty_by_n = pd.DataFrame(
            columns=["dataset_name", "n_samples", "comparison", "wins", "ties", "losses", "win_rate_pct"]
        )
        return empty_global, empty_by_n

    def _summarize(group_cols: list[str]) -> pd.DataFrame:
        out_rows: list[dict] = []
        for keys, chunk in pairwise_unit.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            wins = int((chunk["delta"] > tie_tolerance).sum())
            ties = int((chunk["delta"].abs() <= tie_tolerance).sum())
            losses = int((chunk["delta"] < -tie_tolerance).sum())
            total = wins + ties + losses
            win_rate = 100.0 * wins / total if total else np.nan

            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate_pct": win_rate,
                }
            )
            out_rows.append(row)

        return pd.DataFrame(out_rows)

    global_table = _summarize(["dataset_name", "comparison"]).sort_values(["dataset_name", "comparison"])
    by_n_table = _summarize(["dataset_name", "n_samples", "comparison"]).sort_values(
        ["dataset_name", "n_samples", "comparison"]
    )
    return global_table.reset_index(drop=True), by_n_table.reset_index(drop=True)


def build_table_s4(pairwise_unit: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    if pairwise_unit.empty or s1.empty:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "median_delta_mrmrms_vs_smfs",
                "median_delta_mrmrms_vs_mrmr",
                "mean_delta_mrmrms_vs_smfs",
                "mean_delta_mrmrms_vs_mrmr",
                "best_condition",
                "best_delta_vs_best_baseline",
                "rank_by_median_delta_vs_mrmr",
            ]
        )

    per_dataset = (
        pairwise_unit.groupby(["dataset_name", "comparison"], as_index=False)
        .agg(
            median_delta=("delta", "median"),
            mean_delta=("delta", "mean"),
        )
        .reset_index(drop=True)
    )

    wide_med = per_dataset.pivot(index="dataset_name", columns="comparison", values="median_delta").reset_index()
    wide_mean = per_dataset.pivot(index="dataset_name", columns="comparison", values="mean_delta").reset_index()

    table = wide_med.merge(wide_mean, on="dataset_name", suffixes=("_median", "_mean"))
    table = table.rename(
        columns={
            "mRMR-MS vs Smfs_median": "median_delta_mrmrms_vs_smfs",
            "mRMR-MS vs mRMR_median": "median_delta_mrmrms_vs_mrmr",
            "mRMR-MS vs Smfs_mean": "mean_delta_mrmrms_vs_smfs",
            "mRMR-MS vs mRMR_mean": "mean_delta_mrmrms_vs_mrmr",
        }
    )

    cond = s1[["dataset_name", "n_samples", "mb_stratum", "mRMR-MS_mean", "Smfs_mean", "mRMR_mean"]].copy()
    cond["delta_vs_best_baseline"] = cond["mRMR-MS_mean"] - cond[["Smfs_mean", "mRMR_mean"]].max(axis=1)

    best_idx = cond.groupby("dataset_name")["delta_vs_best_baseline"].idxmax()
    best_rows = cond.loc[best_idx].copy()
    best_rows["best_condition"] = (
        "n="
        + best_rows["n_samples"].astype(int).astype(str)
        + ", MB "
        + best_rows["mb_stratum"].astype(str)
    )

    table = table.merge(
        best_rows[["dataset_name", "best_condition", "delta_vs_best_baseline"]],
        on="dataset_name",
        how="left",
    ).rename(columns={"delta_vs_best_baseline": "best_delta_vs_best_baseline"})

    table["rank_by_median_delta_vs_mrmr"] = (
        table["median_delta_mrmrms_vs_mrmr"].rank(method="dense", ascending=False).astype("Int64")
    )

    return table.sort_values("rank_by_median_delta_vs_mrmr").reset_index(drop=True)


def build_table_s5(pairwise_unit: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    if pairwise_unit.empty:
        return pd.DataFrame(
            columns=[
                "n_samples",
                "mb_stratum",
                "comparison",
                "n_datasets",
                "pct_datasets_positive",
                "median_delta",
                "iqr_q25",
                "iqr_q75",
                "pct_significant_adj",
                "n_significant",
                "n_tested",
            ]
        )

    dataset_condition = (
        pairwise_unit.groupby(["dataset_name", "n_samples", "mb_stratum", "comparison"], as_index=False)[
            "delta"
        ]
        .mean()
        .rename(columns={"delta": "dataset_mean_delta"})
    )

    summary = (
        dataset_condition.groupby(["n_samples", "mb_stratum", "comparison"], as_index=False)["dataset_mean_delta"]
        .agg(
            n_datasets="size",
            median_delta="median",
            iqr_q25=lambda x: float(np.quantile(x, 0.25)),
            iqr_q75=lambda x: float(np.quantile(x, 0.75)),
            pct_datasets_positive=lambda x: float(100.0 * (x > 0.0).mean()),
        )
        .reset_index(drop=True)
    )

    sig_df = s2.dropna(subset=["p_adj"]).copy()
    if sig_df.empty:
        summary["pct_significant_adj"] = np.nan
        summary["n_significant"] = 0
        summary["n_tested"] = 0
        return summary.sort_values(["n_samples", "mb_stratum", "comparison"]).reset_index(drop=True)

    sig_agg = (
        sig_df.groupby(["n_samples", "mb_stratum", "comparison"], as_index=False)
        .agg(
            n_significant=("p_adj", lambda x: int((x < 0.05).sum())),
            n_tested=("p_adj", "size"),
        )
        .reset_index(drop=True)
    )
    sig_agg["pct_significant_adj"] = 100.0 * sig_agg["n_significant"] / sig_agg["n_tested"]

    out = summary.merge(sig_agg, on=["n_samples", "mb_stratum", "comparison"], how="left")
    return out.sort_values(["n_samples", "mb_stratum", "comparison"]).reset_index(drop=True)


def _latex_float(value: float) -> str:
    return f"{value:.4f}"


def _write_latex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str, longtable: bool = True) -> None:
    tex = df.to_latex(
        index=False,
        escape=True,
        na_rep="-",
        longtable=longtable,
        caption=caption,
        label=label,
        float_format=_latex_float,
    )
    out_path.write_text(tex, encoding="utf-8")


def main() -> int:
    args = parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists() or not results_root.is_dir():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            results_root
            / "mb_grouped_comparison_stratified"
            / "supplementary_tables_mrmrms_dataset_level"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _collect_summary_rows(results_root=results_root)
    obs = _prepare_observations(raw_df, mb_threshold=args.mb_threshold)

    if obs.empty:
        print("[supp-tables] No data found. No tables created.")
        return 0

    pairwise_unit = _compute_pairwise_records(obs)

    s1 = build_table_s1(obs)
    s2 = build_table_s2(obs)
    s3_global, s3_by_n = build_table_s3(pairwise_unit, tie_tolerance=args.tie_tolerance)
    s4 = build_table_s4(pairwise_unit, s1)
    s5 = build_table_s5(pairwise_unit, s2)

    tables = [
        (
            s1,
            output_dir / "S1_dataset_level_performance_deltas.tex",
            "Supplementary Table S1. Dataset-level recalls and deltas (mRMR-MS vs baselines) by sample size and MB stratum.",
            "tab:s1_dataset_level_performance_deltas",
        ),
        (
            s2,
            output_dir / "S2_dataset_level_significance_effect_size.tex",
            "Supplementary Table S2. Dataset-level Wilcoxon tests and effect sizes for mRMR-MS comparisons.",
            "tab:s2_dataset_level_significance_effect_size",
        ),
        (
            s3_global,
            output_dir / "S3_dataset_level_win_tie_loss_global.tex",
            "Supplementary Table S3. Dataset-level win/tie/loss counts across all sample sizes and MB strata.",
            "tab:s3_dataset_level_wtl_global",
        ),
        (
            s3_by_n,
            output_dir / "S3b_dataset_level_win_tie_loss_by_samplesize.tex",
            "Supplementary Table S3b. Dataset-level win/tie/loss counts by sample size.",
            "tab:s3b_dataset_level_wtl_by_n",
        ),
        (
            s4,
            output_dir / "S4_dataset_level_magnitude_ranking.tex",
            "Supplementary Table S4. Dataset ranking by mRMR-MS median delta and best-performing condition.",
            "tab:s4_dataset_level_magnitude_ranking",
        ),
        (
            s5,
            output_dir / "S5_conditional_summary_when_advantage_strongest.tex",
            "Supplementary Table S5. Conditional summary of where mRMR-MS advantage is strongest.",
            "tab:s5_conditional_summary",
        ),
    ]

    for df_table, file_path, caption, label in tables:
        _write_latex_table(df_table, file_path, caption=caption, label=label, longtable=True)

    print(f"[supp-tables] results_root={results_root}")
    print(f"[supp-tables] output_dir={output_dir}")
    print(f"[supp-tables] mb_threshold={args.mb_threshold}")
    print(f"[supp-tables] tie_tolerance={args.tie_tolerance}")
    for _, file_path, _, _ in tables:
        print(f"[supp-tables] wrote {file_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
