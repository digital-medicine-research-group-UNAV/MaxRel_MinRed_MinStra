#!/usr/bin/env python3
"""
Transform synthetic CSV datasets for a given dataset name.

Usage:
  python prepare_synthetic_dataset.py --dataset asia
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.preprocessing import LabelEncoder


INPUT_ROOT = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/BNlearn/synthetic_datasets_rds")
OUTPUT_ROOT = Path("/mnt/storage/mlopezdecas/MRMR/new_experiments/DATA/synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode non-numeric columns from synthetic CSV datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (folder under synthetic_datasets_rds).",
    )
    return parser.parse_args()


def is_numeric_column(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)


def encode_binary_column(series: pd.Series) -> pd.Series:
    values = list(pd.unique(series.dropna()))
    mapping = {values[0]: -1, values[1]: 1}
    return series.map(mapping)


def encode_multiclass_column(series: pd.Series) -> pd.Series:
    mask = series.notna()
    encoded = pd.Series([pd.NA] * len(series), index=series.index, dtype="Int64")
    if mask.sum() == 0:
        return encoded

    le = LabelEncoder()
    encoded_values = le.fit_transform(series.loc[mask].astype(str))
    encoded.loc[mask] = encoded_values
    return encoded


def one_hot_encode_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[str]]:
    """Apply one-hot encoding to a non-numeric multi-class feature."""
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
    new_columns = dummies.columns.tolist()
    df = df.drop(columns=[col])
    df = pd.concat([df, dummies], axis=1)
    return df, new_columns


def transform_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if df.shape[1] == 0:
        raise ValueError("CSV without columns.")

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "target"})

    one_hot_metadata: dict[str, list[str]] = {}

    # Process target separately: keep label semantics for class labels.
    if not is_numeric_column(df["target"]):
        n_unique_target = df["target"].dropna().nunique()
        if n_unique_target == 2:
            df["target"] = encode_binary_column(df["target"])
        else:
            df["target"] = encode_multiclass_column(df["target"])

    # Drop constant/near-empty features (<=1 distinct non-null value).
    feature_columns = [col for col in df.columns if col != "target"]
    drop_constant_features = [
        col for col in feature_columns if df[col].dropna().nunique() <= 1
    ]
    if drop_constant_features:
        df = df.drop(columns=drop_constant_features)

    # Process remaining features.
    feature_columns = [col for col in df.columns if col != "target"]
    for col in feature_columns:
        if is_numeric_column(df[col]):
            continue

        n_unique = df[col].dropna().nunique()
        if n_unique == 2:
            df[col] = encode_binary_column(df[col])
        elif n_unique > 2:
            df, new_columns = one_hot_encode_column(df, col)
            one_hot_metadata[col] = new_columns
        else:
            # Encode 0/1-category non-numeric columns to numeric (0 or NA)
            df[col] = encode_multiclass_column(df[col])

    # Keep target in first position.
    ordered_cols = ["target"] + [c for c in df.columns if c != "target"]
    df = df[ordered_cols]

    return df, one_hot_metadata


def save_one_hot_metadata(
    metadata_path: Path,
    dataset_name: str,
    source_csv: str,
    output_csv: str,
    one_hot_mapping: dict[str, list[str]],
) -> None:
    payload = {
        "dataset": dataset_name,
        "source_csv": source_csv,
        "output_csv": output_csv,
        "one_hot_applied": True,
        "drop_first": True,
        "one_hot_mappings": one_hot_mapping,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_name = args.dataset

    dataset_dir = INPUT_ROOT / dataset_name / "datasets"
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Input: {dataset_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] CSV files: {len(csv_files)}")

    for csv_path in csv_files:
        # Preserve literal category labels such as "NA"/"None" from BN datasets.
        # Only empty cells are treated as missing values.
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        df_transformed, one_hot_mapping = transform_dataframe(df)
        out_path = output_dir / csv_path.name
        df_transformed.to_csv(out_path, index=False)
        if one_hot_mapping:
            metadata_path = output_dir / f"{csv_path.stem}__onehot_metadata.json"
            save_one_hot_metadata(
                metadata_path=metadata_path,
                dataset_name=dataset_name,
                source_csv=str(csv_path),
                output_csv=str(out_path),
                one_hot_mapping=one_hot_mapping,
            )
        print(f"[OK] {csv_path.name}")

    print("[DONE] Transformation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
