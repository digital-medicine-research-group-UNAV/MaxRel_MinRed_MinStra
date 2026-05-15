#!/usr/bin/env python3

import argparse
import ast
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = {
    "method": None,
    "methods": None,
    "synthetic_dataset": None,
    "n_features": None,
    "kernel": "",
    "base_seed": 2,
    "lambda": 0.5,
    "confidence_level": 0.1,
    "results_root": None,
    "run_folder": None,
    "max_parallel_methods": 1,
    "auto_tune_parallel": False,
    "internal_workers": None,
    "batch_size": None,
    "knn_jobs": -1,
    "blas_threads": None,
    "run_postprocess": True,
    "postprocess_methods": None,
    "postprocess_classifiers": ["SVM", "KNN"],
    "postprocess_reference_method": None,
    "postprocess_n_features": None,
    "postprocess_n_classes": None,
    "verbose": True,
    "csv_limit": None,
}

FILENAME_PATTERN = re.compile(
    r"^(?P<network>.+?)__target_(?P<target>.+?)__n_(?P<n_samples>\d+)__rep_(?P<rep>\d+)$"
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _project_root(script_dir: Path) -> Path:
    # .../new_experiments/DATA/synthetic -> .../new_experiments
    return script_dir.parents[1]


def _parse_dataset_filename(csv_path: Path) -> tuple[str, str]:
    match = FILENAME_PATTERN.match(csv_path.stem)
    if not match:
        raise ValueError(
            f"CSV filename does not match expected pattern '<network>__target_<target>__n_<n>__rep_<rep>.csv': {csv_path.name}"
        )
    return match.group("network"), match.group("target")


def _parse_serialized_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        pass

    return [text.strip('"')]


def _load_markov_boundary(script_dir: Path, csv_path: Path) -> list[str]:
    project_root = _project_root(script_dir)
    network_name, target_name = _parse_dataset_filename(csv_path)
    summary_path = (
        project_root
        / "BNlearn"
        / "synthetic_datasets_rds"
        / network_name
        / "summary_targets.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_targets.csv not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    rows = summary_df.loc[summary_df["target"].astype(str) == str(target_name)]
    if rows.empty:
        raise ValueError(
            f"Target '{target_name}' not found in {summary_path}"
        )
    mb = _parse_serialized_list(rows.iloc[0]["markov_boundary"])
    mb = [str(x) for x in mb if str(x)]
    if len(mb) == 0:
        raise ValueError(f"Empty Markov boundary for {csv_path.name}")
    return mb


def _normalize_method_jobs(config: dict) -> list[dict]:
    methods_cfg = config.get("methods") or []
    jobs: list[dict] = []
    for entry in methods_cfg:
        if isinstance(entry, str):
            jobs.append({"method": entry, "kernel": ""})
        elif isinstance(entry, dict):
            if "method" not in entry:
                raise ValueError("Each methods entry dict must include 'method'.")
            jobs.append({"method": entry["method"], "kernel": entry.get("kernel", "")})
        else:
            raise ValueError("methods entries must be strings or dict objects.")
    if config.get("method"):
        jobs.append({"method": config["method"], "kernel": config.get("kernel", "")})
    return jobs


def _method_output_name(method: str, kernel: str | None) -> str:
    return method if kernel in [None, ""] else f"{method}_{kernel}"


def _force_methods_n_features(base_config: dict, n_features: int) -> None:
    base_config["n_features"] = int(n_features)
    if not base_config.get("methods"):
        return
    forced: list[dict | str] = []
    for entry in base_config["methods"]:
        if isinstance(entry, str):
            forced.append({"method": entry, "n_features": int(n_features)})
        elif isinstance(entry, dict):
            item = dict(entry)
            item["n_features"] = int(n_features)
            forced.append(item)
        else:
            raise ValueError("methods entries must be strings or dict objects.")
    base_config["methods"] = forced


def _load_onehot_reverse_mapping(csv_path: Path) -> dict[str, str]:
    meta_path = csv_path.with_name(f"{csv_path.stem}__onehot_metadata.json")
    if not meta_path.exists():
        return {}
    payload = _load_json(meta_path)
    mappings = payload.get("one_hot_mappings", {})
    if not isinstance(mappings, dict):
        return {}
    reverse: dict[str, str] = {}
    for original, cols in mappings.items():
        if not isinstance(cols, list):
            continue
        for col in cols:
            reverse[str(col)] = str(original)
    return reverse


def _feature_columns_from_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, nrows=0)
    if len(df.columns) < 2:
        return []
    return [str(c) for c in df.columns[1:]]


def _selection_audit_rows_for_method(
    pickle_path: Path,
    method_name: str,
    kernel: str,
    feature_columns: list[str],
    true_mb: list[str],
    dummy_to_original: dict[str, str],
) -> list[dict]:
    with pickle_path.open("rb") as f:
        fold_payload = pickle.load(f)

    true_mb_set = set(true_mb)
    rows: list[dict] = []
    for fold_idx, fold_dict in enumerate(fold_payload, start=1):
        fold_indices_all = fold_dict.get("Index", []) if isinstance(fold_dict, dict) else []
        final_indices = fold_indices_all[-1] if fold_indices_all else []
        final_indices = [int(x) for x in final_indices]
        selected_columns = [
            feature_columns[i]
            for i in final_indices
            if 0 <= int(i) < len(feature_columns)
        ]
        selected_grouped = sorted({dummy_to_original.get(col, col) for col in selected_columns})

        overlap_raw = sorted(set(selected_columns).intersection(true_mb_set))
        overlap_grouped = sorted(set(selected_grouped).intersection(true_mb_set))

        rows.append(
            {
                "method": method_name,
                "kernel": kernel,
                "method_output_name": _method_output_name(method_name, kernel),
                "fold": fold_idx,
                "n_selected": len(selected_columns),
                "selected_feature_indices": json.dumps(final_indices),
                "selected_features": json.dumps(selected_columns),
                "selected_features_grouped": json.dumps(selected_grouped),
                "markov_boundary": json.dumps(true_mb),
                "mb_size": len(true_mb),
                "overlap_raw_features": json.dumps(overlap_raw),
                "n_overlap_raw": len(overlap_raw),
                "overlap_grouped_features": json.dumps(overlap_grouped),
                "n_overlap_grouped": len(overlap_grouped),
            }
        )
    return rows


def _save_selection_vs_mb_audit(
    csv_path: Path,
    config: dict,
    script_dir: Path,
    true_mb: list[str],
) -> None:
    dataset_name = config["synthetic_dataset"]
    run_folder = csv_path.stem
    results_root = Path(config["results_root"]).resolve()
    run_dir = results_root / f"save_{dataset_name}" / run_folder
    pickle_dir = run_dir / "pickle"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = _feature_columns_from_csv(csv_path)
    dummy_to_original = _load_onehot_reverse_mapping(csv_path)
    jobs = _normalize_method_jobs(config)

    all_rows: list[dict] = []
    missing_pickles: list[str] = []

    for job in jobs:
        method = str(job["method"])
        kernel = "" if job.get("kernel") is None else str(job.get("kernel", ""))
        kernel_suffix = "" if kernel == "" else f"_{kernel}"
        pattern = f"{dataset_name}_{method}{kernel_suffix}_conformal_SVM_*.pickle"
        candidates = sorted(pickle_dir.glob(pattern))
        if not candidates:
            missing_pickles.append(pattern)
            continue
        pick_path = candidates[-1]
        rows = _selection_audit_rows_for_method(
            pickle_path=pick_path,
            method_name=method,
            kernel=kernel,
            feature_columns=feature_columns,
            true_mb=true_mb,
            dummy_to_original=dummy_to_original,
        )
        all_rows.extend(rows)

    audit_csv = results_dir / "selected_features_vs_markov_blanket.csv"
    if all_rows:
        new_df = pd.DataFrame(all_rows)
    else:
        new_df = pd.DataFrame(
            columns=[
                "method",
                "kernel",
                "method_output_name",
                "fold",
                "n_selected",
                "selected_feature_indices",
                "selected_features",
                "selected_features_grouped",
                "markov_boundary",
                "mb_size",
                "overlap_raw_features",
                "n_overlap_raw",
                "overlap_grouped_features",
                "n_overlap_grouped",
            ]
        )

    if audit_csv.exists():
        prev_df = pd.read_csv(audit_csv)
        merged = pd.concat([prev_df, new_df], ignore_index=True)
        if {"method_output_name", "fold"}.issubset(merged.columns):
            merged = merged.drop_duplicates(
                subset=["method_output_name", "fold"],
                keep="last",
            )
        merged.to_csv(audit_csv, index=False)
    else:
        new_df.to_csv(audit_csv, index=False)

    summary_json = results_dir / "markov_blanket_info.json"
    summary_payload = {
        "dataset_csv": str(csv_path.resolve()),
        "markov_boundary": true_mb,
        "mb_size": len(true_mb),
        "n_feature_columns_in_csv": len(feature_columns),
        "onehot_grouping_used": bool(dummy_to_original),
        "missing_method_pickles": missing_pickles,
        "audit_csv": str(audit_csv.resolve()),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def _save_mb_detection_summary(results_dir: Path) -> None:
    source_csv = results_dir / "selected_features_vs_markov_blanket.csv"
    out_csv = results_dir / "markov_blanket_detection_summary.csv"

    if not source_csv.exists():
        pd.DataFrame(
            columns=[
                "method_output_name",
                "n_folds",
                "mb_size",
                "mean_recall_mb_grouped",
                "std_recall_mb_grouped",
                "mean_precision_mb_grouped",
                "std_precision_mb_grouped",
                "exact_match_rate_grouped",
                "mean_recall_mb_raw",
                "std_recall_mb_raw",
                "exact_match_rate_raw",
            ]
        ).to_csv(out_csv, index=False)
        return

    df = pd.read_csv(source_csv)
    if df.empty:
        pd.DataFrame(
            columns=[
                "method_output_name",
                "n_folds",
                "mb_size",
                "mean_recall_mb_grouped",
                "std_recall_mb_grouped",
                "mean_precision_mb_grouped",
                "std_precision_mb_grouped",
                "exact_match_rate_grouped",
                "mean_recall_mb_raw",
                "std_recall_mb_raw",
                "exact_match_rate_raw",
            ]
        ).to_csv(out_csv, index=False)
        return

    work = df.copy()
    work["mb_size"] = pd.to_numeric(work["mb_size"], errors="coerce")
    work["n_selected"] = pd.to_numeric(work["n_selected"], errors="coerce")
    work["n_overlap_grouped"] = pd.to_numeric(work["n_overlap_grouped"], errors="coerce")
    work["n_overlap_raw"] = pd.to_numeric(work["n_overlap_raw"], errors="coerce")

    work["recall_mb_grouped"] = work["n_overlap_grouped"] / work["mb_size"].replace(0, pd.NA)
    work["recall_mb_raw"] = work["n_overlap_raw"] / work["mb_size"].replace(0, pd.NA)
    work["precision_mb_grouped"] = work["n_overlap_grouped"] / work["n_selected"].replace(0, pd.NA)

    work["exact_match_grouped"] = (work["n_overlap_grouped"] == work["mb_size"]).astype(float)
    work["exact_match_raw"] = (work["n_overlap_raw"] == work["mb_size"]).astype(float)

    grouped = work.groupby("method_output_name", dropna=False)
    summary = grouped.agg(
        n_folds=("fold", "count"),
        mb_size=("mb_size", "first"),
        mean_recall_mb_grouped=("recall_mb_grouped", "mean"),
        std_recall_mb_grouped=("recall_mb_grouped", "std"),
        mean_precision_mb_grouped=("precision_mb_grouped", "mean"),
        std_precision_mb_grouped=("precision_mb_grouped", "std"),
        exact_match_rate_grouped=("exact_match_grouped", "mean"),
        mean_recall_mb_raw=("recall_mb_raw", "mean"),
        std_recall_mb_raw=("recall_mb_raw", "std"),
        exact_match_rate_raw=("exact_match_raw", "mean"),
    ).reset_index()

    summary.to_csv(out_csv, index=False)


def _latest_pickle_for(
    pickle_dir: Path,
    dataset_name: str,
    method: str,
    kernel: str,
    family: str,
    classifier: str,
) -> Path | None:
    kernel_suffix = "" if kernel == "" else f"_{kernel}"
    pattern = f"{dataset_name}_{method}{kernel_suffix}_{family}_{classifier}_*.pickle"
    candidates = sorted(pickle_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _safe_last(values: object) -> object:
    if isinstance(values, list) and values:
        return values[-1]
    return None


def _safe_json(value: object) -> str:
    return json.dumps(value, default=str)


def _as_float_or_nan(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
        return out
    except (TypeError, ValueError):
        return float("nan")


def _save_mb_cardinality_metrics(
    csv_path: Path,
    config: dict,
    true_mb: list[str],
) -> None:
    dataset_name = config["synthetic_dataset"]
    run_folder = csv_path.stem
    results_root = Path(config["results_root"]).resolve()
    run_dir = results_root / f"save_{dataset_name}" / run_folder
    pickle_dir = run_dir / "pickle"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    mb_size = len(true_mb)
    method_jobs = _normalize_method_jobs(config)
    rows: list[dict] = []
    missing_pickles: list[str] = []

    for job in method_jobs:
        method = str(job["method"])
        kernel = "" if job.get("kernel") is None else str(job.get("kernel", ""))
        method_name = _method_output_name(method, kernel)

        for classifier in ["SVM", "KNN"]:
            conf_path = _latest_pickle_for(
                pickle_dir=pickle_dir,
                dataset_name=dataset_name,
                method=method,
                kernel=kernel,
                family="conformal",
                classifier=classifier,
            )
            classic_path = _latest_pickle_for(
                pickle_dir=pickle_dir,
                dataset_name=dataset_name,
                method=method,
                kernel=kernel,
                family="classic",
                classifier=classifier,
            )

            if conf_path is None:
                missing_pickles.append(f"missing conformal {classifier} for {method_name}")
                continue
            if classic_path is None:
                missing_pickles.append(f"missing classic {classifier} for {method_name}")
                continue

            with conf_path.open("rb") as f:
                conf_payload = pickle.load(f)
            with classic_path.open("rb") as f:
                classic_payload = pickle.load(f)

            n_folds = min(len(conf_payload), len(classic_payload))
            for fold_idx in range(n_folds):
                conf_fold = conf_payload[fold_idx] if isinstance(conf_payload[fold_idx], dict) else {}
                classic_fold = (
                    classic_payload[fold_idx] if isinstance(classic_payload[fold_idx], dict) else {}
                )

                idx_last = _safe_last(conf_fold.get("Index", []))
                idx_last = idx_last if isinstance(idx_last, list) else []
                n_selected = len(idx_last)

                row = {
                    "method": method,
                    "kernel": kernel,
                    "method_output_name": method_name,
                    "classifier": classifier,
                    "fold": fold_idx + 1,
                    "mb_size": mb_size,
                    "n_selected_final": n_selected,
                    "matches_mb_cardinality": bool(n_selected == mb_size),
                    "selected_feature_indices_final": _safe_json(idx_last),
                    "coverage": _as_float_or_nan(_safe_last(conf_fold.get("coverage", []))),
                    "inefficiency": _as_float_or_nan(_safe_last(conf_fold.get("inefficiency", []))),
                    "certainty": _as_float_or_nan(_safe_last(conf_fold.get("certainty", []))),
                    "uncertainty": _as_float_or_nan(_safe_last(conf_fold.get("uncertainty", []))),
                    "mistrust": _as_float_or_nan(_safe_last(conf_fold.get("mistrust", []))),
                    "S_score": _as_float_or_nan(_safe_last(conf_fold.get("S_score", []))),
                    "F_score": _as_float_or_nan(_safe_last(conf_fold.get("F_score", []))),
                    "Creditibily": _as_float_or_nan(_safe_last(conf_fold.get("Creditibily", []))),
                    "accuracy": _as_float_or_nan(_safe_last(classic_fold.get("accuracy", []))),
                    "precision": _as_float_or_nan(_safe_last(classic_fold.get("precision", []))),
                    "recall": _as_float_or_nan(_safe_last(classic_fold.get("recall", []))),
                    "f1_micro": _as_float_or_nan(_safe_last(classic_fold.get("f1_micro", []))),
                    "f1_macro": _as_float_or_nan(_safe_last(classic_fold.get("f1_macro", []))),
                    "f1_weighted": _as_float_or_nan(_safe_last(classic_fold.get("f1_weighted", []))),
                    "per_class_json": _safe_json(_safe_last(classic_fold.get("per_class", []))),
                }
                rows.append(row)

    per_fold_path = results_dir / "mb_cardinality_metrics_per_fold.csv"
    if rows:
        new_per_fold_df = pd.DataFrame(rows)
    else:
        new_per_fold_df = pd.DataFrame(
            columns=[
                "method",
                "kernel",
                "method_output_name",
                "classifier",
                "fold",
                "mb_size",
                "n_selected_final",
                "matches_mb_cardinality",
                "selected_feature_indices_final",
                "coverage",
                "inefficiency",
                "certainty",
                "uncertainty",
                "mistrust",
                "S_score",
                "F_score",
                "Creditibily",
                "accuracy",
                "precision",
                "recall",
                "f1_micro",
                "f1_macro",
                "f1_weighted",
                "per_class_json",
            ]
        )

    if per_fold_path.exists():
        prev_per_fold_df = pd.read_csv(per_fold_path)
        per_fold_df = pd.concat([prev_per_fold_df, new_per_fold_df], ignore_index=True)
        if {"method_output_name", "classifier", "fold"}.issubset(per_fold_df.columns):
            per_fold_df = per_fold_df.drop_duplicates(
                subset=["method_output_name", "classifier", "fold"],
                keep="last",
            )
    else:
        per_fold_df = new_per_fold_df

    per_fold_df.to_csv(per_fold_path, index=False)

    numeric_cols = [
        "coverage",
        "inefficiency",
        "certainty",
        "uncertainty",
        "mistrust",
        "S_score",
        "F_score",
        "Creditibily",
        "accuracy",
        "precision",
        "recall",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
    ]
    summary_path = results_dir / "mb_cardinality_metrics_summary.csv"
    if per_fold_df.empty:
        pd.DataFrame(
            columns=[
                "method_output_name",
                "classifier",
                "n_folds",
                "mb_size",
                "mean_n_selected_final",
                "all_folds_match_mb_cardinality",
            ]
            + [f"mean_{col}" for col in numeric_cols]
            + [f"std_{col}" for col in numeric_cols]
        ).to_csv(summary_path, index=False)
    else:
        grouped = per_fold_df.groupby(
            ["method_output_name", "classifier"], dropna=False
        )
        summary = grouped.agg(
            n_folds=("fold", "count"),
            mb_size=("mb_size", "first"),
            mean_n_selected_final=("n_selected_final", "mean"),
            all_folds_match_mb_cardinality=("matches_mb_cardinality", "all"),
        ).reset_index()
        for col in numeric_cols:
            summary[f"mean_{col}"] = grouped[col].mean().values
            summary[f"std_{col}"] = grouped[col].std().values
        summary.to_csv(summary_path, index=False)

    info_payload = {
        "dataset_csv": str(csv_path.resolve()),
        "mb_size": mb_size,
        "per_fold_metrics_csv": str(per_fold_path.resolve()),
        "summary_metrics_csv": str(summary_path.resolve()),
        "missing_method_pickles": missing_pickles,
    }
    (results_dir / "mb_cardinality_metrics_info.json").write_text(
        json.dumps(info_payload, indent=2), encoding="utf-8"
    )


def _resolve_config(cli_args: argparse.Namespace, script_dir: Path) -> dict:
    config = dict(DEFAULT_CONFIG)

    if cli_args.config:
        config_path = Path(cli_args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config JSON not found: {config_path}")
        config = _deep_merge(config, _load_json(config_path))

    if cli_args.synthetic_dataset is not None:
        config["synthetic_dataset"] = cli_args.synthetic_dataset
    if cli_args.csv_limit is not None:
        config["csv_limit"] = cli_args.csv_limit
    if cli_args.results_root is not None:
        config["results_root"] = cli_args.results_root

    # Backward/alias compatibility.
    if config.get("synthetic_dataset") is None and config.get("dataset") is not None:
        config["synthetic_dataset"] = config["dataset"]

    if config.get("synthetic_dataset") is None:
        raise ValueError(
            "Missing required parameter: synthetic_dataset (CLI --synthetic_dataset or JSON 'synthetic_dataset')."
        )
    if config.get("method") is None and not config.get("methods"):
        raise ValueError("Missing required parameter: method or methods.")

    if config.get("csv_limit") is not None:
        config["csv_limit"] = int(config["csv_limit"])
        if config["csv_limit"] < 1:
            raise ValueError("csv_limit must be >= 1 when provided.")

    if config.get("results_root") is None:
        config["results_root"] = str((_project_root(script_dir) / "RESULTS" / "synthetic").resolve())
    else:
        config["results_root"] = str(Path(config["results_root"]).expanduser().resolve())

    config["synthetic_dataset"] = str(config["synthetic_dataset"])
    return config


def _discover_csvs(config: dict, script_dir: Path) -> list[Path]:
    synthetic_root = script_dir
    dataset_dir = synthetic_root / config["synthetic_dataset"]
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Synthetic dataset folder not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    if config.get("csv_limit") is not None:
        csv_files = csv_files[: int(config["csv_limit"])]
    return csv_files


def _build_child_config(base_config: dict, dataset_name: str, csv_stem: str, data_root: Path) -> dict:
    child = dict(base_config)

    # Remove synthetic-only keys.
    child.pop("synthetic_dataset", None)
    child.pop("csv_limit", None)

    # launcher_local expects these.
    child["dataset"] = dataset_name
    child["data_root"] = str(data_root.resolve())
    child["results_root"] = base_config["results_root"]
    child["run_postprocess"] = False

    # Requirement: subfolder per analyzed CSV under save_<dataset>.
    child["run_folder"] = csv_stem
    return child


def _run_for_csv(
    csv_path: Path,
    config: dict,
    script_dir: Path,
    launcher_local_path: Path,
) -> None:
    dataset_name = config["synthetic_dataset"]
    csv_stem = csv_path.stem
    true_mb = _load_markov_boundary(script_dir=script_dir, csv_path=csv_path)
    mb_size = len(true_mb)

    print(
        f"[synthetic] processing csv={csv_path.name} -> run_folder={csv_stem} mb_size={mb_size}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix=f"synthetic_{dataset_name}_{csv_stem}_") as tmp:
        tmp_root = Path(tmp)
        dataset_dir = tmp_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Adapt data reading to launcher_local expected layout.
        staged_csv = dataset_dir / f"data_{dataset_name}.csv"
        shutil.copy2(csv_path, staged_csv)

        child_cfg = _build_child_config(
            base_config=config,
            dataset_name=dataset_name,
            csv_stem=csv_stem,
            data_root=tmp_root,
        )
        _force_methods_n_features(child_cfg, n_features=mb_size)
        child_cfg_path = tmp_root / f"cfg_{csv_stem}.json"
        with child_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(child_cfg, f, indent=2)

        cmd = [sys.executable, str(launcher_local_path), "--config", str(child_cfg_path)]
        subprocess.run(cmd, cwd=str(_project_root(script_dir)), check=True)

    _save_selection_vs_mb_audit(
        csv_path=csv_path,
        config=config,
        script_dir=script_dir,
        true_mb=true_mb,
    )
    results_dir = (
        Path(config["results_root"]).resolve()
        / f"save_{dataset_name}"
        / csv_stem
        / "results"
    )
    _save_mb_detection_summary(results_dir=results_dir)
    _save_mb_cardinality_metrics(
        csv_path=csv_path,
        config=config,
        true_mb=true_mb,
    )


def run_all(config: dict, script_dir: Path) -> None:
    launcher_local_path = (_project_root(script_dir) / "launcher_local.py").resolve()
    if not launcher_local_path.exists():
        raise FileNotFoundError(f"launcher_local.py not found: {launcher_local_path}")

    csv_files = _discover_csvs(config=config, script_dir=script_dir)
    print(f"[synthetic] dataset={config['synthetic_dataset']} csv_files={len(csv_files)}", flush=True)
    print(f"[synthetic] results_root={config['results_root']}", flush=True)

    failures: list[tuple[str, str]] = []
    for csv_path in csv_files:
        try:
            _run_for_csv(
                csv_path=csv_path,
                config=config,
                script_dir=script_dir,
                launcher_local_path=launcher_local_path,
            )
        except subprocess.CalledProcessError as exc:
            failures.append((csv_path.name, f"exit={exc.returncode}"))
            print(f"[synthetic] FAILED csv={csv_path.name} exit={exc.returncode}", flush=True)
        except Exception as exc:
            failures.append((csv_path.name, str(exc)))
            print(f"[synthetic] FAILED csv={csv_path.name} error={exc}", flush=True)

    if failures:
        lines = "\n".join(f"{name}: {err}" for name, err in failures)
        raise RuntimeError(f"Synthetic launcher failed for one or more CSV files:\n{lines}")

    print("[synthetic] all CSV runs completed successfully", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=False, help="Path to JSON config.")
    parser.add_argument(
        "--synthetic_dataset",
        type=str,
        default=None,
        required=False,
        help="Synthetic dataset name under DATA/synthetic.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        required=False,
        help="Optional results root. Default: RESULTS/synthetic",
    )
    parser.add_argument(
        "--csv_limit",
        type=int,
        default=None,
        required=False,
        help="Optional limit for number of CSVs to run (debug/testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config = _resolve_config(args, script_dir=script_dir)
    run_all(config=config, script_dir=script_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
