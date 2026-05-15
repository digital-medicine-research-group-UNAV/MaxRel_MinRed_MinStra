#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd


NOTEBOOK_PATH = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/DATA/synthetic/dojo.ipynb")
RESULTS_ROOT = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/RESULTS/synthetic")
DATA_ROOT = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/DATA/synthetic")
META_ROOT = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/BNlearn/synthetic_datasets_rds")
RUN_PATTERN = re.compile(r"^(?P<network>.+?)__target_(?P<target>.+?)__n_(?P<n_samples>\d+)__rep_(?P<rep>\d+)$")
AUTO_SECTION_TITLE = "## Dataset-level tables for paper discussion"
OPTIONAL_EXPORTS_TITLE = "## Optional exports"


def load_summary_records(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(results_root.glob("save_*/*/results/markov_blanket_detection_summary.csv")):
        run_folder = csv_path.parent.parent.name
        run_match = RUN_PATTERN.match(run_folder)
        if run_match is None:
            continue

        df = pd.read_csv(csv_path)
        if "method_output_name" not in df.columns or "mean_recall_mb_grouped" not in df.columns:
            continue

        df = df[["method_output_name", "mean_recall_mb_grouped"]].copy()
        df["mean_recall_mb_grouped"] = pd.to_numeric(df["mean_recall_mb_grouped"], errors="coerce")
        df = df.dropna(subset=["method_output_name", "mean_recall_mb_grouped"])

        for _, row in df.iterrows():
            rows.append(
                {
                    "dataset_name": csv_path.parents[2].name.replace("save_", ""),
                    "run_folder": run_folder,
                    "network_name": run_match.group("network"),
                    "target_name": run_match.group("target"),
                    "n_samples": int(run_match.group("n_samples")),
                    "replication_id": int(run_match.group("rep")),
                    "method_output_name": row["method_output_name"],
                    "mean_recall_mb_grouped": float(row["mean_recall_mb_grouped"]),
                    "path": str(csv_path),
                }
            )

    return pd.DataFrame(rows)


def analyze_method_best(
    summary_df: pd.DataFrame,
    target_method: str,
    allowed_methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if allowed_methods is not None:
        working_df = summary_df.loc[summary_df["method_output_name"].isin(allowed_methods)].copy()
    else:
        working_df = summary_df.copy()

    grouped: list[dict[str, object]] = []
    group_cols = [
        "dataset_name",
        "run_folder",
        "network_name",
        "target_name",
        "n_samples",
        "replication_id",
        "path",
    ]
    for group_key, group_df in working_df.groupby(group_cols, sort=True):
        if target_method not in set(group_df["method_output_name"]):
            continue

        target_score = float(
            group_df.loc[
                group_df["method_output_name"] == target_method, "mean_recall_mb_grouped"
            ].iloc[0]
        )
        best_score = float(group_df["mean_recall_mb_grouped"].max())
        best_methods = sorted(
            group_df.loc[
                group_df["mean_recall_mb_grouped"] == best_score, "method_output_name"
            ].tolist()
        )

        if target_score == best_score and best_methods == [target_method]:
            status = "strict_best"
        elif target_score == best_score:
            status = "tied_best"
        else:
            status = "not_best"

        grouped.append(
            {
                "dataset_name": group_key[0],
                "run_folder": group_key[1],
                "network_name": group_key[2],
                "target_name": group_key[3],
                "n_samples": group_key[4],
                "replication_id": group_key[5],
                "path": group_key[6],
                "target_method": target_method,
                "target_score": target_score,
                "best_score": best_score,
                "best_methods": ", ".join(best_methods),
                "status": status,
                "methods_considered": ", ".join(sorted(group_df["method_output_name"].tolist())),
            }
        )

    results_df = (
        pd.DataFrame(grouped)
        .sort_values(["status", "dataset_name", "n_samples", "run_folder"])
        .reset_index(drop=True)
    )
    if results_df.empty:
        summary_by_samples = pd.DataFrame(columns=["status", "n_samples", "n_csvs"])
    else:
        summary_by_samples = (
            results_df.groupby(["status", "n_samples"])
            .size()
            .rename("n_csvs")
            .reset_index()
            .sort_values(["status", "n_samples"])
            .reset_index(drop=True)
        )

    strict_best_df = results_df.loc[results_df["status"] == "strict_best"].copy()
    tied_best_df = results_df.loc[results_df["status"] == "tied_best"].copy()
    not_best_df = results_df.loc[results_df["status"] == "not_best"].copy()

    return {
        "results_df": results_df,
        "summary_by_samples": summary_by_samples,
        "strict_best_df": strict_best_df,
        "tied_best_df": tied_best_df,
        "not_best_df": not_best_df,
    }


def load_dataset_characteristics(summary_df: pd.DataFrame) -> pd.DataFrame:
    dataset_rows: list[dict[str, object]] = []
    datasets = sorted(summary_df["dataset_name"].dropna().unique())

    for dataset_name in datasets:
        metadata_path = META_ROOT / dataset_name / "network_metadata.json"
        targets_path = META_ROOT / dataset_name / "summary_targets.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        targets_df = pd.read_csv(targets_path)

        targets_in_results = sorted(
            summary_df.loc[summary_df["dataset_name"] == dataset_name, "target_name"]
            .astype(str)
            .unique()
            .tolist()
        )
        targets_used_df = targets_df.loc[targets_df["target"].astype(str).isin(targets_in_results)].copy()
        if targets_used_df.empty:
            targets_used_df = targets_df.copy()

        feature_counts: list[int] = []
        onehot_var_counts: list[int] = []
        onehot_dummy_counts: list[int] = []
        max_dummy_widths: list[int] = []
        for csv_path in sorted((DATA_ROOT / dataset_name).glob("*__n_250__rep_1.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
            feature_counts.append(max(len(header) - 1, 0))

            onehot_path = csv_path.with_name(f"{csv_path.stem}__onehot_metadata.json")
            if onehot_path.exists():
                payload = json.loads(onehot_path.read_text(encoding="utf-8"))
                mappings = payload.get("one_hot_mappings", {})
                widths = [len(cols) for cols in mappings.values() if isinstance(cols, list)]
                onehot_var_counts.append(len(widths))
                onehot_dummy_counts.append(sum(widths))
                max_dummy_widths.append(max(widths) if widths else 0)
            else:
                onehot_var_counts.append(0)
                onehot_dummy_counts.append(0)
                max_dummy_widths.append(0)

        dataset_rows.append(
            {
                "dataset_name": dataset_name,
                "n_nodes": int(metadata["n_nodes"]),
                "n_targets_eval": len(targets_in_results),
                "mb_mean": float(pd.to_numeric(targets_used_df["mb_size"]).mean()),
                "mb_median": float(pd.to_numeric(targets_used_df["mb_size"]).median()),
                "parents_mean": float(pd.to_numeric(targets_used_df["n_parents"]).mean()),
                "children_mean": float(pd.to_numeric(targets_used_df["n_children"]).mean()),
                "spouses_mean": float(pd.to_numeric(targets_used_df["n_spouses"]).mean()),
                "features_after_ohe_mean": float(pd.Series(feature_counts, dtype=float).mean()),
                "onehot_source_vars_mean": float(pd.Series(onehot_var_counts, dtype=float).mean()),
                "onehot_dummy_cols_mean": float(pd.Series(onehot_dummy_counts, dtype=float).mean()),
                "max_dummy_width_mean": float(pd.Series(max_dummy_widths, dtype=float).mean()),
            }
        )

    out = pd.DataFrame(dataset_rows).sort_values("dataset_name").reset_index(drop=True)
    numeric_cols = [
        "mb_mean",
        "mb_median",
        "parents_mean",
        "children_mean",
        "spouses_mean",
        "features_after_ohe_mean",
        "onehot_source_vars_mean",
        "onehot_dummy_cols_mean",
        "max_dummy_width_mean",
    ]
    out[numeric_cols] = out[numeric_cols].round(3)
    return out


def build_dataset_performance_table(
    results_df: pd.DataFrame,
    dataset_characteristics_df: pd.DataFrame,
) -> pd.DataFrame:
    working = results_df.copy()
    working["best_or_tie"] = working["status"] != "not_best"

    status_counts = (
        working.groupby(["dataset_name", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["strict_best", "tied_best", "not_best"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    best_or_tie_rate = (
        working.groupby("dataset_name")["best_or_tie"].mean().rename("best_or_tie_rate").reset_index()
    )

    best_or_tie_by_samples = (
        working.groupby(["dataset_name", "n_samples"])["best_or_tie"]
        .mean()
        .rename("best_or_tie_rate")
        .reset_index()
        .pivot(index="dataset_name", columns="n_samples", values="best_or_tie_rate")
        .rename(
            columns={
                250: "best_or_tie_250",
                1000: "best_or_tie_1000",
                5000: "best_or_tie_5000",
            }
        )
        .reset_index()
    )

    out = dataset_characteristics_df.merge(best_or_tie_rate, on="dataset_name", how="left")
    out = out.merge(status_counts[["dataset_name", "strict_best", "tied_best", "not_best"]], on="dataset_name", how="left")
    out = out.merge(best_or_tie_by_samples, on="dataset_name", how="left")

    for col in ["best_or_tie_rate", "best_or_tie_250", "best_or_tie_1000", "best_or_tie_5000"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0).round(3)
    for col in ["strict_best", "tied_best", "not_best"]:
        out[col] = out[col].fillna(0).astype(int)

    return out.sort_values("best_or_tie_rate", ascending=False).reset_index(drop=True)


def build_extremes_table(profile_df: pd.DataFrame, method_label: str, top_n: int = 5) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "best_or_tie_rate",
        "best_or_tie_250",
        "best_or_tie_1000",
        "best_or_tie_5000",
        "n_nodes",
        "mb_mean",
        "spouses_mean",
        "features_after_ohe_mean",
        "onehot_dummy_cols_mean",
    ]
    top = profile_df.loc[:, columns].head(top_n).copy()
    top.insert(0, "group", "top")
    top.insert(1, "target_method", method_label)

    bottom = profile_df.loc[:, columns].tail(top_n).iloc[::-1].copy()
    bottom.insert(0, "group", "bottom")
    bottom.insert(1, "target_method", method_label)

    return pd.concat([top, bottom], ignore_index=True)


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(source),
    }


def code_cell(source: str, execution_count: int | None = None, outputs: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": lines(source),
    }


def display_output(df: pd.DataFrame) -> dict[str, object]:
    html = df.to_html(index=True, border=1)
    text = df.to_string()
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [html],
            "text/plain": [text],
        },
        "metadata": {},
    }


def stream_output(text: str) -> dict[str, object]:
    return {
        "name": "stdout",
        "output_type": "stream",
        "text": lines(text),
    }


def replace_intro_markdown(notebook: dict[str, object]) -> None:
    intro = notebook["cells"][0]
    intro["source"] = lines(
        "# dojo\n"
        "\n"
        "This notebook scans all `markov_blanket_detection_summary.csv` files under `RESULTS/synthetic` and analyzes when the `_MS_linear` method reaches the best `mean_recall_mb_grouped`.\n"
        "\n"
        "It contains:\n"
        "- a general analysis for `mRMR_MS_linear` against all methods present in each CSV,\n"
        "- a focused analysis for `MS` vs `mRMR` vs `mRMR_MS_linear`,\n"
        "- a focused analysis for `MS` vs `JMI` vs `JMI_MS_linear`,\n"
        "- dataset-level structural tables that relate performance to network characteristics.\n"
    )


def update_optional_exports(notebook: dict[str, object]) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and OPTIONAL_EXPORTS_TITLE in "".join(cell["source"]):
            continue
        if cell["cell_type"] == "code" and "dojo_mrmr_triplet_strict_best.csv" in "".join(cell["source"]):
            cell["source"] = lines(
                "# mrmr_triplet['strict_best_df'].to_csv('dojo_mrmr_triplet_strict_best.csv', index=False)\n"
                "# mrmr_triplet['tied_best_df'].to_csv('dojo_mrmr_triplet_tied_best.csv', index=False)\n"
                "# jmi_triplet['strict_best_df'].to_csv('dojo_jmi_triplet_strict_best.csv', index=False)\n"
                "# jmi_triplet['tied_best_df'].to_csv('dojo_jmi_triplet_tied_best.csv', index=False)\n"
                "# dataset_characteristics_table.to_csv('dojo_dataset_characteristics_table.csv', index=False)\n"
                "# mrmr_dataset_profile_table.to_csv('dojo_mrmr_dataset_profile_table.csv', index=False)\n"
                "# jmi_dataset_profile_table.to_csv('dojo_jmi_dataset_profile_table.csv', index=False)\n"
                "# mrmr_extremes_df.to_csv('dojo_mrmr_dataset_extremes.csv', index=False)\n"
                "# jmi_extremes_df.to_csv('dojo_jmi_dataset_extremes.csv', index=False)\n"
                "# Path('dojo_dataset_characteristics_table.tex').write_text(dataset_characteristics_table.to_latex(index=False, float_format=lambda x: f'{x:.3f}'), encoding='utf-8')\n"
                "# Path('dojo_mrmr_dataset_profile_table.tex').write_text(mrmr_dataset_profile_table.to_latex(index=False, float_format=lambda x: f'{x:.3f}'), encoding='utf-8')\n"
                "# Path('dojo_jmi_dataset_profile_table.tex').write_text(jmi_dataset_profile_table.to_latex(index=False, float_format=lambda x: f'{x:.3f}'), encoding='utf-8')\n"
                "# Path('dojo_mrmr_dataset_extremes.tex').write_text(mrmr_extremes_df.to_latex(index=False, float_format=lambda x: f'{x:.3f}'), encoding='utf-8')\n"
                "# Path('dojo_jmi_dataset_extremes.tex').write_text(jmi_extremes_df.to_latex(index=False, float_format=lambda x: f'{x:.3f}'), encoding='utf-8')\n"
            )


def insert_auto_section(notebook: dict[str, object]) -> None:
    summary_df = load_summary_records()
    mrmr_triplet = analyze_method_best(
        summary_df,
        target_method="mRMR_MS_linear",
        allowed_methods=["MS", "mRMR", "mRMR_MS_linear"],
    )
    jmi_triplet = analyze_method_best(
        summary_df,
        target_method="JMI_MS_linear",
        allowed_methods=["MS", "JMI", "JMI_MS_linear"],
    )

    dataset_characteristics_df = load_dataset_characteristics(summary_df)
    dataset_characteristics_table = dataset_characteristics_df[
        [
            "dataset_name",
            "n_nodes",
            "n_targets_eval",
            "mb_mean",
            "mb_median",
            "parents_mean",
            "children_mean",
            "spouses_mean",
            "features_after_ohe_mean",
            "onehot_source_vars_mean",
            "onehot_dummy_cols_mean",
            "max_dummy_width_mean",
        ]
    ].sort_values("dataset_name").reset_index(drop=True)

    mrmr_dataset_profile_df = build_dataset_performance_table(
        mrmr_triplet["results_df"], dataset_characteristics_df
    )
    mrmr_dataset_profile_table = mrmr_dataset_profile_df[
        [
            "dataset_name",
            "best_or_tie_rate",
            "strict_best",
            "tied_best",
            "not_best",
            "best_or_tie_250",
            "best_or_tie_1000",
            "best_or_tie_5000",
            "n_nodes",
            "mb_mean",
            "spouses_mean",
            "features_after_ohe_mean",
            "onehot_dummy_cols_mean",
        ]
    ].copy()

    jmi_dataset_profile_df = build_dataset_performance_table(
        jmi_triplet["results_df"], dataset_characteristics_df
    )
    jmi_dataset_profile_table = jmi_dataset_profile_df[
        [
            "dataset_name",
            "best_or_tie_rate",
            "strict_best",
            "tied_best",
            "not_best",
            "best_or_tie_250",
            "best_or_tie_1000",
            "best_or_tie_5000",
            "n_nodes",
            "mb_mean",
            "spouses_mean",
            "features_after_ohe_mean",
            "onehot_dummy_cols_mean",
        ]
    ].copy()

    mrmr_extremes_df = build_extremes_table(mrmr_dataset_profile_table, "mRMR_MS_linear")
    jmi_extremes_df = build_extremes_table(jmi_dataset_profile_table, "JMI_MS_linear")

    auto_cells = [
        markdown_cell(
            "## Dataset-level tables for paper discussion\n"
            "\n"
            "The following tables recompute the current contents of `RESULTS/synthetic` and relate the behavior of the `_MS_linear` variants to structural properties of each Bayesian network and to the dimensionality induced by one-hot encoding.\n"
            "\n"
            "`best_or_tie_rate` denotes the fraction of analyzed `(dataset, target, sample size)` configurations where the target method is either strictly best or tied for the best `mean_recall_mb_grouped`.\n"
        ),
        code_cell(
            "import csv\n"
            "import json\n"
            "\n"
            "BNLEARN_METADATA_ROOT = Path('/mnt/storage/mlopezdecas/MRMR/new_experiments/BNlearn/synthetic_datasets_rds')\n"
            "DATA_ROOT = Path('/mnt/storage/mlopezdecas/MRMR/new_experiments/DATA/synthetic')\n"
            "\n"
            "\n"
            "def load_dataset_characteristics(summary_df: pd.DataFrame) -> pd.DataFrame:\n"
            "    dataset_rows = []\n"
            "    datasets = sorted(summary_df['dataset_name'].dropna().unique())\n"
            "\n"
            "    for dataset_name in datasets:\n"
            "        metadata_path = BNLEARN_METADATA_ROOT / dataset_name / 'network_metadata.json'\n"
            "        targets_path = BNLEARN_METADATA_ROOT / dataset_name / 'summary_targets.csv'\n"
            "        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))\n"
            "        targets_df = pd.read_csv(targets_path)\n"
            "\n"
            "        targets_in_results = sorted(\n"
            "            summary_df.loc[summary_df['dataset_name'] == dataset_name, 'target_name']\n"
            "            .astype(str)\n"
            "            .unique()\n"
            "            .tolist()\n"
            "        )\n"
            "        targets_used_df = targets_df.loc[targets_df['target'].astype(str).isin(targets_in_results)].copy()\n"
            "        if targets_used_df.empty:\n"
            "            targets_used_df = targets_df.copy()\n"
            "\n"
            "        feature_counts = []\n"
            "        onehot_var_counts = []\n"
            "        onehot_dummy_counts = []\n"
            "        max_dummy_widths = []\n"
            "        for csv_path in sorted((DATA_ROOT / dataset_name).glob('*__n_250__rep_1.csv')):\n"
            "            with csv_path.open('r', encoding='utf-8', newline='') as f:\n"
            "                reader = csv.reader(f)\n"
            "                header = next(reader)\n"
            "            feature_counts.append(max(len(header) - 1, 0))\n"
            "\n"
            "            onehot_path = csv_path.with_name(f'{csv_path.stem}__onehot_metadata.json')\n"
            "            if onehot_path.exists():\n"
            "                payload = json.loads(onehot_path.read_text(encoding='utf-8'))\n"
            "                mappings = payload.get('one_hot_mappings', {})\n"
            "                widths = [len(cols) for cols in mappings.values() if isinstance(cols, list)]\n"
            "                onehot_var_counts.append(len(widths))\n"
            "                onehot_dummy_counts.append(sum(widths))\n"
            "                max_dummy_widths.append(max(widths) if widths else 0)\n"
            "            else:\n"
            "                onehot_var_counts.append(0)\n"
            "                onehot_dummy_counts.append(0)\n"
            "                max_dummy_widths.append(0)\n"
            "\n"
            "        dataset_rows.append(\n"
            "            {\n"
            "                'dataset_name': dataset_name,\n"
            "                'n_nodes': int(metadata['n_nodes']),\n"
            "                'n_targets_eval': len(targets_in_results),\n"
            "                'mb_mean': float(pd.to_numeric(targets_used_df['mb_size']).mean()),\n"
            "                'mb_median': float(pd.to_numeric(targets_used_df['mb_size']).median()),\n"
            "                'parents_mean': float(pd.to_numeric(targets_used_df['n_parents']).mean()),\n"
            "                'children_mean': float(pd.to_numeric(targets_used_df['n_children']).mean()),\n"
            "                'spouses_mean': float(pd.to_numeric(targets_used_df['n_spouses']).mean()),\n"
            "                'features_after_ohe_mean': float(pd.Series(feature_counts, dtype=float).mean()),\n"
            "                'onehot_source_vars_mean': float(pd.Series(onehot_var_counts, dtype=float).mean()),\n"
            "                'onehot_dummy_cols_mean': float(pd.Series(onehot_dummy_counts, dtype=float).mean()),\n"
            "                'max_dummy_width_mean': float(pd.Series(max_dummy_widths, dtype=float).mean()),\n"
            "            }\n"
            "        )\n"
            "\n"
            "    out = pd.DataFrame(dataset_rows).sort_values('dataset_name').reset_index(drop=True)\n"
            "    numeric_cols = [\n"
            "        'mb_mean',\n"
            "        'mb_median',\n"
            "        'parents_mean',\n"
            "        'children_mean',\n"
            "        'spouses_mean',\n"
            "        'features_after_ohe_mean',\n"
            "        'onehot_source_vars_mean',\n"
            "        'onehot_dummy_cols_mean',\n"
            "        'max_dummy_width_mean',\n"
            "    ]\n"
            "    out[numeric_cols] = out[numeric_cols].round(3)\n"
            "    return out\n"
            "\n"
            "\n"
            "def build_dataset_performance_table(results_df: pd.DataFrame, dataset_characteristics_df: pd.DataFrame) -> pd.DataFrame:\n"
            "    working = results_df.copy()\n"
            "    working['best_or_tie'] = working['status'] != 'not_best'\n"
            "\n"
            "    status_counts = (\n"
            "        working.groupby(['dataset_name', 'status'])\n"
            "        .size()\n"
            "        .unstack(fill_value=0)\n"
            "        .reset_index()\n"
            "    )\n"
            "    for col in ['strict_best', 'tied_best', 'not_best']:\n"
            "        if col not in status_counts.columns:\n"
            "            status_counts[col] = 0\n"
            "\n"
            "    best_or_tie_rate = (\n"
            "        working.groupby('dataset_name')['best_or_tie']\n"
            "        .mean()\n"
            "        .rename('best_or_tie_rate')\n"
            "        .reset_index()\n"
            "    )\n"
            "\n"
            "    best_or_tie_by_samples = (\n"
            "        working.groupby(['dataset_name', 'n_samples'])['best_or_tie']\n"
            "        .mean()\n"
            "        .rename('best_or_tie_rate')\n"
            "        .reset_index()\n"
            "        .pivot(index='dataset_name', columns='n_samples', values='best_or_tie_rate')\n"
            "        .rename(columns={250: 'best_or_tie_250', 1000: 'best_or_tie_1000', 5000: 'best_or_tie_5000'})\n"
            "        .reset_index()\n"
            "    )\n"
            "\n"
            "    out = dataset_characteristics_df.merge(best_or_tie_rate, on='dataset_name', how='left')\n"
            "    out = out.merge(\n"
            "        status_counts[['dataset_name', 'strict_best', 'tied_best', 'not_best']],\n"
            "        on='dataset_name',\n"
            "        how='left',\n"
            "    )\n"
            "    out = out.merge(best_or_tie_by_samples, on='dataset_name', how='left')\n"
            "\n"
            "    for col in ['best_or_tie_rate', 'best_or_tie_250', 'best_or_tie_1000', 'best_or_tie_5000']:\n"
            "        if col in out.columns:\n"
            "            out[col] = out[col].fillna(0.0).round(3)\n"
            "    for col in ['strict_best', 'tied_best', 'not_best']:\n"
            "        out[col] = out[col].fillna(0).astype(int)\n"
            "\n"
            "    return out.sort_values('best_or_tie_rate', ascending=False).reset_index(drop=True)\n"
            "\n"
            "\n"
            "def build_extremes_table(profile_df: pd.DataFrame, method_label: str, top_n: int = 5) -> pd.DataFrame:\n"
            "    columns = [\n"
            "        'dataset_name',\n"
            "        'best_or_tie_rate',\n"
            "        'best_or_tie_250',\n"
            "        'best_or_tie_1000',\n"
            "        'best_or_tie_5000',\n"
            "        'n_nodes',\n"
            "        'mb_mean',\n"
            "        'spouses_mean',\n"
            "        'features_after_ohe_mean',\n"
            "        'onehot_dummy_cols_mean',\n"
            "    ]\n"
            "    top = profile_df.loc[:, columns].head(top_n).copy()\n"
            "    top.insert(0, 'group', 'top')\n"
            "    top.insert(1, 'target_method', method_label)\n"
            "\n"
            "    bottom = profile_df.loc[:, columns].tail(top_n).iloc[::-1].copy()\n"
            "    bottom.insert(0, 'group', 'bottom')\n"
            "    bottom.insert(1, 'target_method', method_label)\n"
            "\n"
            "    return pd.concat([top, bottom], ignore_index=True)\n"
            "\n"
            "\n"
            "dataset_characteristics_df = load_dataset_characteristics(summary_df)\n"
            "dataset_characteristics_table = dataset_characteristics_df[\n"
            "    [\n"
            "        'dataset_name',\n"
            "        'n_nodes',\n"
            "        'n_targets_eval',\n"
            "        'mb_mean',\n"
            "        'mb_median',\n"
            "        'parents_mean',\n"
            "        'children_mean',\n"
            "        'spouses_mean',\n"
            "        'features_after_ohe_mean',\n"
            "        'onehot_source_vars_mean',\n"
            "        'onehot_dummy_cols_mean',\n"
            "        'max_dummy_width_mean',\n"
            "    ]\n"
            "].sort_values('dataset_name').reset_index(drop=True)\n"
            "\n"
            "mrmr_dataset_profile_df = build_dataset_performance_table(mrmr_triplet['results_df'], dataset_characteristics_df)\n"
            "mrmr_dataset_profile_table = mrmr_dataset_profile_df[\n"
            "    [\n"
            "        'dataset_name',\n"
            "        'best_or_tie_rate',\n"
            "        'strict_best',\n"
            "        'tied_best',\n"
            "        'not_best',\n"
            "        'best_or_tie_250',\n"
            "        'best_or_tie_1000',\n"
            "        'best_or_tie_5000',\n"
            "        'n_nodes',\n"
            "        'mb_mean',\n"
            "        'spouses_mean',\n"
            "        'features_after_ohe_mean',\n"
            "        'onehot_dummy_cols_mean',\n"
            "    ]\n"
            "]\n"
            "\n"
            "jmi_dataset_profile_df = build_dataset_performance_table(jmi_triplet['results_df'], dataset_characteristics_df)\n"
            "jmi_dataset_profile_table = jmi_dataset_profile_df[\n"
            "    [\n"
            "        'dataset_name',\n"
            "        'best_or_tie_rate',\n"
            "        'strict_best',\n"
            "        'tied_best',\n"
            "        'not_best',\n"
            "        'best_or_tie_250',\n"
            "        'best_or_tie_1000',\n"
            "        'best_or_tie_5000',\n"
            "        'n_nodes',\n"
            "        'mb_mean',\n"
            "        'spouses_mean',\n"
            "        'features_after_ohe_mean',\n"
            "        'onehot_dummy_cols_mean',\n"
            "    ]\n"
            "]\n"
            "\n"
            "mrmr_extremes_df = build_extremes_table(mrmr_dataset_profile_table, 'mRMR_MS_linear')\n"
            "jmi_extremes_df = build_extremes_table(jmi_dataset_profile_table, 'JMI_MS_linear')\n"
        ),
        code_cell(
            "display(dataset_characteristics_table)",
            outputs=[display_output(dataset_characteristics_table)],
        ),
        code_cell(
            "display(mrmr_dataset_profile_table)\n"
            "display(mrmr_extremes_df)",
            outputs=[
                display_output(mrmr_dataset_profile_table),
                display_output(mrmr_extremes_df),
            ],
        ),
        code_cell(
            "display(jmi_dataset_profile_table)\n"
            "display(jmi_extremes_df)",
            outputs=[
                display_output(jmi_dataset_profile_table),
                display_output(jmi_extremes_df),
            ],
        ),
        markdown_cell(
            "### LaTeX exports\n"
            "\n"
            "These strings can be copied directly into the manuscript or written to `.tex` files from the optional export cell.\n"
        ),
        code_cell(
            "latex_float = lambda x: f'{x:.3f}'\n"
            "\n"
            "latex_exports = {\n"
            "    'dataset_characteristics_table': dataset_characteristics_table.to_latex(index=False, float_format=latex_float),\n"
            "    'mrmr_dataset_profile_table': mrmr_dataset_profile_table.to_latex(index=False, float_format=latex_float),\n"
            "    'jmi_dataset_profile_table': jmi_dataset_profile_table.to_latex(index=False, float_format=latex_float),\n"
            "    'mrmr_extremes_df': mrmr_extremes_df.to_latex(index=False, float_format=latex_float),\n"
            "    'jmi_extremes_df': jmi_extremes_df.to_latex(index=False, float_format=latex_float),\n"
            "}\n"
            "\n"
            "for name, latex_table in latex_exports.items():\n"
            "    print(f'===== {name} =====')\n"
            "    print(latex_table)\n",
            outputs=[
                stream_output(
                    "This cell is intentionally left unexecuted in the saved notebook. "
                    "Run it interactively to materialize the `to_latex()` strings in your local notebook session.\n"
                )
            ],
        ),
        markdown_cell(
            "### Interpretation notes\n"
            "\n"
            "- `mRMR_MS_linear` performs best on `mehra-complete`, `pathfinder`, `link`, `diabetes`, and `ecoli70`. These favorable cases correspond either to strong post-encoding dimensionality expansion (`mehra-complete`, `pathfinder`, `diabetes`) or to medium-sized networks with comparatively clean local structure (`ecoli70`).\n"
            "- The weakest regimes for `mRMR_MS_linear` are `magic-niab`, `healthcare`, and `magic-irri`. Two unfavorable profiles emerge: very small networks with limited room for separation (`healthcare`) and spouse-heavy networks without substantial one-hot expansion (`magic-niab`, `magic-irri`).\n"
            "- `sangiovese` is locally dense (`mb_mean` and `spouses_mean` are both high) yet remains difficult for both hybrid variants, suggesting that very dense local neighborhoods reduce the benefit of the strangeness-based refinement.\n"
            "- `JMI_MS_linear` shows a different behavior on `link`: `mRMR_MS_linear` remains competitive, but `JMI_MS_linear` almost never wins. This indicates that the gain depends not only on dataset size, but also on the interaction between the scoring family and the network structure.\n"
        ),
    ]

    cells = notebook["cells"]
    optional_exports_idx = next(
        idx
        for idx, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and OPTIONAL_EXPORTS_TITLE in "".join(cell["source"])
    )

    auto_start_idx = None
    for idx, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and AUTO_SECTION_TITLE in "".join(cell["source"]):
            auto_start_idx = idx
            break

    if auto_start_idx is not None and auto_start_idx < optional_exports_idx:
        del cells[auto_start_idx:optional_exports_idx]
        optional_exports_idx = auto_start_idx

    for offset, cell in enumerate(auto_cells):
        cells.insert(optional_exports_idx + offset, cell)


def main() -> None:
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 200)

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    replace_intro_markdown(notebook)
    insert_auto_section(notebook)
    update_optional_exports(notebook)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
