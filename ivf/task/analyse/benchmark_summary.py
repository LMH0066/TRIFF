"""Summarize saved benchmark predictions without fitting models or choosing thresholds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = {"LR": ".lr", "HGB": ".hgb", "RF": ".rf", "TabPFN": ".tab", "TRIFF": ""}
METRICS = {"roc_auc": "ROC-AUC", "average_precision": "AP"}
DATASETS = ["Validation set", "2025 test set", "Validation set - 2025 test set"]


def compute_metrics(source, seeds, include_combined=False, training_manifest=None):
    """Check score/label integrity; scores need not be calibrated probabilities."""
    rows = []
    manifest = None
    if training_manifest is not None:
        import json
        manifest = json.loads(Path(training_manifest).read_text())
    for seed in seeds:
        train_ids = None
        train_targets = None
        if manifest is not None:
            import anndata
            adata = anndata.read_h5ad(manifest[str(seed)]["dataset"])
            train_mask = adata.obs["split_key"].to_numpy() == "train"
            train_ids = adata.obs_names[train_mask].tolist()
            train_targets = np.asarray(adata.obsm["target"]).reshape(-1)[train_mask]
            if not train_ids or len(set(train_ids)) != len(train_ids):
                raise ValueError("Training sample IDs must be nonempty and unique")
        datasets = list(zip(DATASETS[:2], ["", ".2025"], strict=True))
        if include_combined:
            datasets.insert(0, ("Training + validation set", ".combined"))
        if train_ids is not None:
            datasets.insert(0, ("Training set", ".combined"))
        for dataset, suffix in datasets:
            reference = None
            for model, tag in MODELS.items():
                path = Path(source) / f"predictions.{seed}{tag}{suffix}.csv"
                frame = pd.read_csv(path, dtype={"sample_id": str})
                if dataset == "Training set":
                    if train_ids is None or train_targets is None:
                        raise ValueError("Training manifest is required")
                    if "sample_id" not in frame or frame.sample_id.duplicated().any():
                        raise ValueError(f"Missing or duplicate sample IDs: {path}")
                    indexed = frame.set_index("sample_id")
                    if not set(train_ids).issubset(indexed.index):
                        raise ValueError(f"Training sample IDs missing: {path}")
                    frame = indexed.loc[train_ids]
                    if not np.array_equal(frame.targets.to_numpy(), train_targets):
                        raise ValueError(f"Training labels differ from original data: {path}")
                if not {"targets", "predictions"}.issubset(frame.columns):
                    raise ValueError(f"Missing targets/predictions: {path}")
                target = frame.targets.to_numpy(dtype=float)
                score = frame.predictions.to_numpy(dtype=float)
                if not np.isfinite(score).all() or set(np.unique(target)) != {0, 1}:
                    raise ValueError(f"Expected finite scores and both binary classes: {path}")
                if reference is not None and not np.array_equal(target, reference):
                    raise ValueError(f"Target order differs across models: {path}")
                reference = target
                rows.append({"seed": seed, "model": model, "dataset": dataset,
                             "roc_auc": roc_auc_score(target, score),
                             "average_precision": average_precision_score(target, score)})
    frame = pd.DataFrame(rows)
    validation = frame[frame.dataset == DATASETS[0]].set_index(["seed", "model"])
    external = frame[frame.dataset == DATASETS[1]].set_index(["seed", "model"])
    delta = validation[list(METRICS)] - external[list(METRICS)]
    delta["dataset"] = DATASETS[2]
    return pd.concat([frame, delta.reset_index()], ignore_index=True)


def summarize(frame):
    long = frame.melt(id_vars=["seed", "model", "dataset"], value_vars=list(METRICS),
                      var_name="metric", value_name="value")
    return long.groupby(["dataset", "metric", "model"], sort=False).value.agg(
        ["mean", "std", "count"]).reset_index()


def make_table(summary):
    rows, labels = [], []
    datasets = [name for name in ["Training set", "Training + validation set"] if name in summary.dataset.values] + DATASETS
    for dataset in datasets:
        for metric, label in METRICS.items():
            subset = summary[(summary.dataset == dataset) & (summary.metric == metric)].set_index("model")
            rows.append([f"{subset.loc[model, 'mean']:.4f} ± {subset.loc[model, 'std']:.4f}"
                         for model in MODELS])
            labels.append(f"{dataset}\n{label}")
    return pd.DataFrame(rows, index=pd.Index(labels), columns=pd.Index(MODELS))


def plot_boxplots(frame, output):
    """Plot per-split metrics, not the mean/SD cells in the summary table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    datasets = DATASETS[:2]
    data = frame.loc[frame.dataset.isin(datasets)]
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        palette = dict(zip(datasets, sns.color_palette("colorblind", len(datasets)), strict=True))
        for ax, (metric, label) in zip(axes[0], METRICS.items(), strict=True):
            sns.boxplot(data=data, x="model", y=metric, order=list(MODELS),
                        hue="dataset", hue_order=datasets, palette=palette,
                        dodge=True, showfliers=False, showmeans=True, width=0.8,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 4}, ax=ax)
            sns.stripplot(data=data, x="model", y=metric, order=list(MODELS),
                          hue="dataset", hue_order=datasets, palette=palette,
                          dodge=True, legend=False, size=2.5, alpha=0.55, jitter=False, ax=ax)
            ax.set(xlabel="", ylabel=label, title=label)
            if metric == "roc_auc":
                ax.set_ylim(0.65, 1)
            else:
                lower = max(0, np.floor((data[metric].min() - 0.01) * 20) / 20)
                upper = min(1, np.ceil((data[metric].max() + 0.01) * 20) / 20)
                ax.set_ylim(lower, upper)
            ax.get_legend().remove()
        delta = frame.loc[frame.dataset == DATASETS[2]]
        for ax, (metric, label) in zip(axes[1], METRICS.items(), strict=True):
            sns.boxplot(data=delta, x="model", y=metric, order=list(MODELS),
                        color="#b9c9d6", showfliers=False, showmeans=True, width=0.55,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 4}, ax=ax)
            sns.stripplot(data=delta, x="model", y=metric, order=list(MODELS),
                          color="black", size=2.5, alpha=0.55, jitter=False, ax=ax)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            bound = max(0.02, float(delta[metric].abs().max()) * 1.15)
            ax.set(xlabel="", ylabel=f"Δ {label}",
                   title=f"Validation set − 2025 test set — {label}", ylim=(-bound, bound))
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=palette[name], label=name) for name in datasets],
                   loc="upper center", bbox_to_anchor=(0.5, 1), ncol=len(datasets),
                   title="Data source", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=root / "benchmark")
    parser.add_argument("--output", type=Path, default=root / "benchmark_summary")
    parser.add_argument("--seeds", default=",".join(map(str, range(15))))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-combined", action="store_true", help="Read .combined.csv predictions (includes training samples)")
    parser.add_argument("--training-manifest", type=Path, help="Extract Training set from combined predictions using original split IDs")
    args = parser.parse_args()
    if args.include_combined and args.training_manifest is not None:
        parser.error("Choose training-only or combined, not both")
    if not args.include_combined and args.training_manifest is None:
        args.training_manifest = root / "ablation_manifest.json"
    seeds = list(map(int, args.seeds.split(",")))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("At least two unique seeds are required for sample SD")
    names = ["metrics.csv", "summary.csv", "performance_table.csv", "performance_table.pdf", "performance_boxplots.pdf"]
    if not args.overwrite and any((args.output / name).exists() for name in names):
        raise FileExistsError("Summary outputs exist; use --overwrite")
    frame = compute_metrics(args.source, seeds, args.include_combined, args.training_manifest)
    summary = summarize(frame)
    table = make_table(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / names[0], index=False)
    summary.to_csv(args.output / names[1], index=False)
    table.to_csv(args.output / names[2], index_label="Dataset / metric")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 8 if args.include_combined or args.training_manifest else 6))
    ax.axis("off")
    artist = ax.table(cellText=table.values.tolist(), colLabels=table.columns.tolist(),
                      rowLabels=table.index.tolist(), cellLoc="center", loc="center")
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    artist.scale(1, 3)
    ax.set_title("Performance of the Models", fontsize=18, fontweight="bold", pad=25)
    fig.text(0.02, 0.03,
             f"Mean ± sample SD across {len(seeds)} repeated splits (not independent cohorts).\n"
             "AP: sklearn average_precision_score; positive class: abnormal fertilization. Training/combined scores describe in-sample fitting, not held-out performance.\n"
             "Difference: validation minus 2025 test, paired by split. LR: LinearRegression(fit_intercept=False, positive=True).",
             fontsize=9)
    fig.savefig(args.output / names[3], bbox_inches="tight")
    plt.close(fig)
    plot_boxplots(frame, args.output / "performance_boxplots.pdf")
    print(f"Summarized {len(seeds)} splits, {len(MODELS)} models: {args.output}")


if __name__ == "__main__":
    main()
