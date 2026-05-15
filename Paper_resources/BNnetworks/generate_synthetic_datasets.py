#!/usr/bin/env python3
"""
Generate synthetic datasets from Bayesian networks downloaded from bnlearn.

Usage examples:
  python generate_synthetic_datasets.py \
    --input_dir ./networks \
    --output_dir ./synthetic_datasets \
    --n_samples 250,1000,5000

  python generate_synthetic_datasets.py \
    --input_dir ./networks \
    --network alarm \
    --output_dir ./synthetic_datasets \
    --n_samples 250 \
    --n_replications 3 \
    --seed 42

Dependencies:
  - pandas
  - pgmpy
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install with `pip install pandas`."
    ) from exc

try:
    from pgmpy.readwrite import BIFReader
    from pgmpy.sampling import BayesianModelSampling
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pgmpy. Install with `pip install pgmpy`."
    ) from exc


LOGGER = logging.getLogger("synthetic-generator")

SUPPORTED_SUFFIXES = (".bif", ".bif.gz", ".dsc", ".net")
SUMMARY_COLUMNS = [
    "network_name",
    "target",
    "n_nodes",
    "n_parameters",
    "n_parents",
    "n_children",
    "n_spouses",
    "mb_size",
    "parents",
    "children",
    "spouses",
    "markov_boundary",
]


@dataclass(frozen=True)
class TargetInfo:
    """Structural information for a valid target node."""

    target: str
    parents: list[str]
    children: list[str]
    spouses: list[str]

    @property
    def markov_boundary(self) -> list[str]:
        return sorted(set(self.parents) | set(self.children) | set(self.spouses))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create synthetic CSV datasets from Bayesian networks."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing downloaded Bayesian network files.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Specific network name or filename to sample (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where datasets and metadata will be written.",
    )
    parser.add_argument(
        "--n_samples",
        type=str,
        required=True,
        help="Samples per dataset. Supports one value (e.g. 250) or many (e.g. 250,1000,5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n_replications",
        type=int,
        default=1,
        help="Number of datasets to generate per target (default: 1).",
    )
    return parser.parse_args()


def parse_sample_sizes(raw: str) -> list[int]:
    """Parse --n_samples into a validated list of positive integers."""
    sizes = [token.strip() for token in raw.split(",")]
    out: list[int] = []
    for token in sizes:
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Invalid n_samples value: {value}. Must be > 0.")
        out.append(value)
    if not out:
        raise ValueError("No valid n_samples values provided.")
    return sorted(set(out))


def network_base_name(network_file: Path) -> str:
    """Return network logical name without format suffixes."""
    lower_name = network_file.name.lower()
    if lower_name.endswith(".bif.gz"):
        return network_file.name[:-7]
    return network_file.stem


def discover_network_files(input_dir: Path) -> list[Path]:
    """Discover supported network files in input_dir."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = [p for p in input_dir.iterdir() if p.is_file()]
    candidates = [p for p in files if any(p.name.lower().endswith(ext) for ext in SUPPORTED_SUFFIXES)]
    candidates.sort(key=lambda p: p.name.lower())

    if not candidates:
        raise FileNotFoundError(
            f"No supported network files found in {input_dir}. "
            f"Expected one of: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    return candidates


def filter_networks(network_files: list[Path], requested_network: Optional[str]) -> list[Path]:
    """Filter detected network files by --network if provided."""
    if requested_network is None:
        return network_files

    requested = requested_network.strip().lower()
    matched = [
        p
        for p in network_files
        if p.name.lower() == requested
        or network_base_name(p).lower() == requested
        or p.stem.lower() == requested
    ]
    if not matched:
        available = ", ".join(network_base_name(p) for p in network_files)
        raise FileNotFoundError(
            f"Network '{requested_network}' not found in input_dir. "
            f"Available: {available}"
        )
    return matched


def load_network(network_file: Path) -> Any:
    """
    Load a Bayesian network file.

    Current implementation supports BIF as the primary format and keeps an explicit
    dispatch layer to extend loaders for DSC/NET without touching downstream logic.
    """
    lower_name = network_file.name.lower()

    if lower_name.endswith(".bif"):
        return load_bif_network(network_file)
    if lower_name.endswith(".bif.gz"):
        return load_bif_gz_network(network_file)
    if lower_name.endswith(".dsc"):
        raise NotImplementedError("DSC loader not implemented yet.")
    if lower_name.endswith(".net"):
        raise NotImplementedError("NET loader not implemented yet.")

    raise ValueError(f"Unsupported network format: {network_file.name}")


def load_bif_network(network_file: Path) -> Any:
    """Load a .bif Bayesian network."""
    reader = BIFReader(str(network_file))
    return reader.get_model()


def load_bif_gz_network(network_file: Path) -> Any:
    """Load a .bif.gz Bayesian network via temporary decompression."""
    with gzip.open(network_file, "rb") as f_in:
        bif_bytes = f_in.read()

    with tempfile.NamedTemporaryFile(suffix=".bif", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(bif_bytes)

    try:
        reader = BIFReader(str(temp_path))
        model = reader.get_model()
    finally:
        temp_path.unlink(missing_ok=True)

    return model


def validate_dag(model: Any, network_name: str) -> None:
    """Validate that the loaded network structure is a DAG."""
    if not is_dag_from_edges(model.nodes(), model.edges()):
        raise ValueError(f"Network '{network_name}' is not a valid DAG.")

    if hasattr(model, "check_model"):
        model.check_model()


def is_dag_from_edges(nodes: Any, edges: Any) -> bool:
    """Check acyclicity using Kahn's algorithm without external dependencies."""
    node_list = [str(node) for node in nodes]
    adjacency: dict[str, set[str]] = {node: set() for node in node_list}
    indegree: dict[str, int] = {node: 0 for node in node_list}

    for source, target in edges:
        source_name = str(source)
        target_name = str(target)
        if source_name not in adjacency:
            adjacency[source_name] = set()
            indegree[source_name] = 0
        if target_name not in adjacency:
            adjacency[target_name] = set()
            indegree[target_name] = 0
        if target_name not in adjacency[source_name]:
            adjacency[source_name].add(target_name)
            indegree[target_name] += 1

    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return visited == len(indegree)


def get_parents(model: Any, node: Any) -> list[str]:
    """Return sorted parent list for a node."""
    return sorted(str(parent) for parent in model.get_parents(node))


def get_children(model: Any, node: Any) -> list[str]:
    """Return sorted child list for a node."""
    return sorted(str(child) for child in model.get_children(node))


def get_spouses(model: Any, node: Any) -> list[str]:
    """Return sorted spouse list (other parents of node's children)."""
    node_name = str(node)
    spouses: set[str] = set()
    for child in model.get_children(node):
        for parent in model.get_parents(child):
            parent_name = str(parent)
            if parent_name != node_name:
                spouses.add(parent_name)
    return sorted(spouses)


def get_markov_boundary(parents: list[str], children: list[str], spouses: list[str]) -> list[str]:
    """Return Markov boundary as union(parents, children, spouses)."""
    return sorted(set(parents) | set(children) | set(spouses))


def find_valid_targets(model: Any) -> list[TargetInfo]:
    """Find target nodes with at least one parent, one child and one spouse."""
    valid: list[TargetInfo] = []

    for node in model.nodes():
        target_name = str(node)
        parents = get_parents(model, node)
        children = get_children(model, node)
        spouses = get_spouses(model, node)

        if len(parents) < 1 or len(children) < 1 or len(spouses) < 1:
            continue

        target = TargetInfo(
            target=target_name,
            parents=parents,
            children=children,
            spouses=spouses,
        )

        if not target.parents or not target.children or not target.spouses:
            continue

        valid.append(target)

    return sorted(valid, key=lambda item: item.target)


def compute_n_parameters(model: Any) -> Optional[int]:
    """Compute total number of free parameters from CPDs if available."""
    try:
        cpds = model.get_cpds()
    except Exception:
        return None

    if not cpds:
        return None

    total = 0
    for cpd in cpds:
        values = cpd.get_values()
        if values.ndim == 1:
            variable_card = int(values.shape[0])
            parent_configs = 1
        else:
            variable_card = int(values.shape[0])
            parent_configs = int(values.shape[1])
        total += (variable_card - 1) * parent_configs
    return total


def sample_dataset(model: Any, target: str, n_samples: int, seed_rep: int) -> pd.DataFrame:
    """Sample from network joint distribution and reorder columns."""
    sampler = BayesianModelSampling(model)
    sampled = sampler.forward_sample(size=n_samples, show_progress=False, seed=seed_rep)
    df = sampled.copy()
    df.columns = [str(col) for col in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not present in sampled dataframe columns.")

    ordered_columns = [target] + sorted(col for col in df.columns if col != target)
    return df[ordered_columns]


def validate_dataset_shape(df: pd.DataFrame, n_samples: int, n_expected_columns: int) -> None:
    """Validate generated dataset dimensions."""
    if df.shape[0] != n_samples:
        raise ValueError(
            f"Dataset has {df.shape[0]} rows but expected {n_samples}."
        )
    if df.shape[1] != n_expected_columns:
        raise ValueError(
            f"Dataset has {df.shape[1]} columns but expected {n_expected_columns}."
        )


def sanitize_name(name: str) -> str:
    """Sanitize arbitrary node names for safe filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Write sampled dataset to CSV with header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def serialize_list(values: list[str]) -> str:
    """Serialize lists for CSV cells in a deterministic format."""
    return json.dumps(values, ensure_ascii=False)


def save_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save summary_targets.csv (with header even if empty)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(output_path, index=False)


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Save network-level metadata JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_summary_rows(
    network_name: str,
    n_nodes: int,
    n_parameters: Optional[int],
    targets: list[TargetInfo],
) -> list[dict[str, Any]]:
    """Build rows for summary_targets.csv."""
    rows: list[dict[str, Any]] = []

    for target in targets:
        mb = target.markov_boundary
        rows.append(
            {
                "network_name": network_name,
                "target": target.target,
                "n_nodes": n_nodes,
                "n_parameters": n_parameters,
                "n_parents": len(target.parents),
                "n_children": len(target.children),
                "n_spouses": len(target.spouses),
                "mb_size": len(mb),
                "parents": serialize_list(target.parents),
                "children": serialize_list(target.children),
                "spouses": serialize_list(target.spouses),
                "markov_boundary": serialize_list(mb),
            }
        )

    return rows


def process_network(
    network_file: Path,
    output_root: Path,
    sample_sizes: list[int],
    seed: int,
    n_replications: int,
) -> None:
    """Process one network end-to-end: load, target discovery, sample, and save outputs."""
    network_name = network_base_name(network_file)
    network_dir = output_root / network_name
    datasets_dir = network_dir / "datasets"
    summary_path = network_dir / "summary_targets.csv"
    metadata_path = network_dir / "network_metadata.json"

    network_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading network: %s", network_file)
    model = load_network(network_file)
    validate_dag(model, network_name)

    node_names = sorted(str(node) for node in model.nodes())
    n_nodes = len(node_names)
    n_parameters = compute_n_parameters(model)

    LOGGER.info("Network '%s' loaded with %d nodes", network_name, n_nodes)
    valid_targets = find_valid_targets(model)
    LOGGER.info("Valid targets found in '%s': %d", network_name, len(valid_targets))

    summary_rows = build_summary_rows(
        network_name=network_name,
        n_nodes=n_nodes,
        n_parameters=n_parameters,
        targets=valid_targets,
    )
    save_summary(summary_rows, summary_path)

    generated_files = 0
    expected_columns = n_nodes

    for target_index, target in enumerate(valid_targets):
        if not (target.parents and target.children and target.spouses):
            raise ValueError(f"Invalid target structure validation failed: {target.target}")

        safe_target = sanitize_name(target.target)

        for n_samples in sample_sizes:
            for replication_id in range(1, n_replications + 1):
                seed_rep = seed + target_index * 1000 + replication_id + n_samples * 10
                df = sample_dataset(
                    model=model,
                    target=target.target,
                    n_samples=n_samples,
                    seed_rep=seed_rep,
                )
                validate_dataset_shape(df, n_samples=n_samples, n_expected_columns=expected_columns)

                file_name = (
                    f"{network_name}__target_{safe_target}__n_{n_samples}__rep_{replication_id}.csv"
                )
                save_dataset(df, datasets_dir / file_name)
                generated_files += 1

    metadata = {
        "network_name": network_name,
        "source_file": str(network_file.resolve()),
        "n_nodes": n_nodes,
        "node_names": node_names,
        "valid_targets": [target.target for target in valid_targets],
        "n_valid_targets": len(valid_targets),
        "sampling_method": "pgmpy.BayesianModelSampling.forward_sample",
        "seed": seed,
        "n_samples": sample_sizes if len(sample_sizes) > 1 else sample_sizes[0],
        "n_replications": n_replications,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_metadata(metadata, metadata_path)

    LOGGER.info(
        "Network '%s' completed. Datasets generated: %d. Output: %s",
        network_name,
        generated_files,
        network_dir,
    )


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    try:
        sample_sizes = parse_sample_sizes(args.n_samples)
    except ValueError as exc:
        LOGGER.error("Invalid --n_samples: %s", exc)
        return 2

    if args.n_replications < 1:
        LOGGER.error("--n_replications must be >= 1.")
        return 2

    try:
        discovered = discover_network_files(args.input_dir)
        selected = filter_networks(discovered, args.network)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Detected %d network file(s) in %s", len(discovered), args.input_dir)
    LOGGER.info("Processing %d network file(s)", len(selected))

    errors = 0
    for network_file in selected:
        try:
            process_network(
                network_file=network_file,
                output_root=args.output_dir,
                sample_sizes=sample_sizes,
                seed=args.seed,
                n_replications=args.n_replications,
            )
        except NotImplementedError as exc:
            errors += 1
            LOGGER.error("Skipping %s: %s", network_file.name, exc)
        except Exception as exc:
            errors += 1
            LOGGER.error("Failed processing %s: %s", network_file.name, exc)

    if errors > 0:
        LOGGER.warning("Finished with %d error(s).", errors)
        return 1

    LOGGER.info("All requested networks processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
