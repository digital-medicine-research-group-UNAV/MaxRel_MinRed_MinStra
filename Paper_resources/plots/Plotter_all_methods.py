from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from local_runtime import default_results_root, load_dataset


RUN_FOLDER = "pruebas_revision"
PLOTTED_METRICS = ["coverage", "inefficiency", "certainty"]
METHOD_SPECS = [
    (("mRMR_MS_linear",), "mRMR-MS"),
    (("MS_linear", "MS"), "SMFS"),
    (("mRMR",), "mRMR"),
    (("JMI",), "JMI"),
    (("relax_mRMR",), "relax-mRMR"),
    (("CONMI_FS",), "CONMI-FS"),
    (("greedy",), "greedy"),
]
COLORS = {
    "coverage": "blue",
    "inefficiency": "red",
    "certainty": "green",
}
DASHES = {
    "coverage": (1, 0),
    "inefficiency": (6, 1),
    "certainty": (1, 2),
}
LINEWIDTHS = {
    "coverage": 1.2,
    "inefficiency": 1.9,
    "certainty": 1.5,
}
ALPHAS = {
    "coverage": 0.40,
    "inefficiency": 0.50,
    "certainty": 0.20,
}


def resolve_pickle_path(path_like: Path) -> Path:
    path = Path(path_like)
    if path.exists():
        return path

    candidates = sorted(path.parent.glob(f"{path.stem}_*.pickle"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"Pickle file not found: {path}")


def pick_pickle_path(run_dir: Path, method_names: tuple[str, ...], classifier: str) -> tuple[Path, str]:
    pickle_dir = run_dir / "pickle"
    dataset = run_dir.parent.name.replace("save_", "", 1)

    checked: list[Path] = []
    for method in method_names:
        filename = f"{dataset}_{method}_conformal_{classifier}.pickle"
        for base_dir in (pickle_dir, run_dir):
            candidate = base_dir / filename
            checked.append(candidate)
            try:
                return resolve_pickle_path(candidate), method
            except FileNotFoundError:
                continue

    checked_text = "\n".join(str(path) for path in checked)
    raise FileNotFoundError(f"No pickle found for {dataset} {method_names} {classifier}. Checked:\n{checked_text}")


def min_max_scale(values: list[float], x_min: float, x_max: float) -> list[float]:
    if x_max == x_min:
        return [0.0 for _ in values]
    return [(value - x_min) / (x_max - x_min) for value in values]


def load_metric_curves(pickle_path: Path, n_classes: int) -> tuple[dict[str, list[float]], dict[str, list[float]], list[int]]:
    with open(resolve_pickle_path(pickle_path), "rb") as file:
        out = pickle.load(file)

    if not out:
        raise ValueError(f"Empty pickle content: {pickle_path}")

    raw: dict[str, list[list[float]]] = {metric: [] for metric in PLOTTED_METRICS}
    x: list[int] | None = None

    for fold_result in out:
        if x is None:
            x = list(range(1, len(fold_result["inefficiency"]) + 1))

        for metric in PLOTTED_METRICS:
            values = fold_result[metric]
            if metric == "inefficiency":
                values = min_max_scale(values, 0, n_classes)
            raw[metric].append(values)

    data_mean: dict[str, list[float]] = {}
    data_std: dict[str, list[float]] = {}
    for metric, values in raw.items():
        values_array = np.array(values)
        data_mean[metric] = np.mean(values_array, axis=0).tolist()
        data_std[metric] = np.std(values_array, axis=0).tolist()

    return data_mean, data_std, x or []


def infer_n_classes(dataset: str, script_dir: Path, data_root: str | None) -> int:
    df, _ = load_dataset(dataset=dataset, script_dir=script_dir, data_root=data_root)
    return len(sorted(df["Class"].dropna().unique().tolist()))


def infer_n_classes_from_pickles(run_dir: Path, classifier: str) -> int:
    max_inefficiency = 0.0
    pickle_paths = sorted((run_dir / "pickle").glob(f"*_conformal_{classifier}*.pickle"))
    pickle_paths.extend(sorted(run_dir.glob(f"*_conformal_{classifier}*.pickle")))

    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as file:
            out = pickle.load(file)
        for fold_result in out:
            values = fold_result.get("inefficiency", [])
            if values:
                max_inefficiency = max(max_inefficiency, max(values))

    if max_inefficiency <= 0:
        raise ValueError(f"Could not infer n_classes from pickles under {run_dir}")
    return max(1, math.ceil(max_inefficiency))


def plot_dataset(
    dataset: str,
    results_root: Path,
    run_folder: str,
    classifier: str,
    n_classes: int,
) -> list[Path]:
    run_dir = results_root / f"save_{dataset}" / run_folder
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_cols = 2
    n_rows = math.ceil(len(METHOD_SPECS) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 3.45 * n_rows), sharey=True)
    axes_flat = np.array(axes).reshape(-1)

    legend_handles = []
    legend_labels = []

    for ax, (method_candidates, display_name) in zip(axes_flat, METHOD_SPECS):
        pickle_path, resolved_method = pick_pickle_path(run_dir, method_candidates, classifier)
        data_mean, data_std, x = load_metric_curves(pickle_path, n_classes=n_classes)

        for metric in PLOTTED_METRICS:
            mean = np.array(data_mean[metric])
            std = np.array(data_std[metric])
            line = ax.plot(
                x,
                mean,
                label=metric,
                dashes=DASHES[metric],
                markersize=2.5,
                color=COLORS[metric],
                linestyle="--",
                linewidth=LINEWIDTHS[metric],
            )[0]
            ax.fill_between(x, mean + std, mean - std, color=COLORS[metric], alpha=ALPHAS[metric])

            if metric not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(metric)

        title = display_name
        if resolved_method not in method_candidates[:1]:
            title = f"{display_name}"
        ax.set_title(title)
        ax.set_xlabel("Num. of Features")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([i / 10 for i in range(11)])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes_flat[len(METHOD_SPECS) :]:
        ax.axis("off")

    fig.suptitle(f"{dataset} - {classifier}", y=0.995)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(PLOTTED_METRICS),
        frameon=False,
        handlelength=3.8,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.006),
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))

    stem = f"conformal_scores_{dataset}_{classifier}_all_methods"
    pdf_path = plots_dir / f"{stem}.pdf"
    svg_path = plots_dir / f"{stem}.svg"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [pdf_path, svg_path]


def has_classifier_pickles(run_dir: Path, classifier: str) -> bool:
    return any((run_dir / "pickle").glob(f"*_conformal_{classifier}*.pickle")) or any(
        run_dir.glob(f"*_conformal_{classifier}*.pickle")
    )


def discover_datasets(results_root: Path, run_folder: str, classifier: str) -> list[str]:
    datasets = []
    for save_dir in sorted(results_root.glob("save_*")):
        if not save_dir.is_dir():
            continue
        if "synthetic" in save_dir.name.lower():
            continue
        if not has_classifier_pickles(save_dir / run_folder, classifier):
            print(f"[skip] no {classifier} conformal pickles found under {save_dir / run_folder}", file=sys.stderr)
            continue
        datasets.append(save_dir.name.replace("save_", "", 1))
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name. If omitted, all non-synthetic save_* folders are processed.")
    parser.add_argument("--classifier", type=str, default="SVM", help="Classifier suffix to read from conformal pickles.")
    parser.add_argument("--n_classes", type=int, default=None, help="Class count. Required only when --dataset is used and data cannot be loaded.")
    parser.add_argument("--results_root", type=str, default=None, help="Optional results root. If empty, RESULTS/ is used.")
    parser.add_argument("--run_folder", type=str, default=RUN_FOLDER, help="Subfolder inside save_<dataset>.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional dataset root for automatic class-count inference.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = default_results_root(script_dir=script_dir, results_root=args.results_root)
    datasets = [args.dataset] if args.dataset else discover_datasets(results_root, args.run_folder, args.classifier)

    if not datasets:
        raise FileNotFoundError(f"No non-synthetic save_* folders found under {results_root}")

    for dataset in datasets:
        n_classes = args.n_classes
        if n_classes is None:
            try:
                n_classes = infer_n_classes(dataset=dataset, script_dir=script_dir, data_root=args.data_root)
            except FileNotFoundError:
                run_dir = results_root / f"save_{dataset}" / args.run_folder
                n_classes = infer_n_classes_from_pickles(run_dir=run_dir, classifier=args.classifier)
                print(
                    f"[warn] inferred n_classes={n_classes} from pickles for dataset '{dataset}'",
                    file=sys.stderr,
                )

        out_paths = plot_dataset(
            dataset=dataset,
            results_root=results_root,
            run_folder=args.run_folder,
            classifier=args.classifier,
            n_classes=n_classes,
        )
        for path in out_paths:
            print(path)


if __name__ == "__main__":
    main()
