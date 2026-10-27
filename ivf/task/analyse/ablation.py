"""SHAP ranking and fixed-model feature-masking experiments.

Run from the repository root: python -m ivf.task.analyse.ablation --help
Only trusted local scvi model checkpoints should be loaded.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ivf import IVF
def reference_values(train, categorical):
    """Training-only median/mode; all-missing columns remain missing."""
    result = np.full(train.shape[1], np.nan, dtype=np.float32)
    for j in range(train.shape[1]):
        values = train[:, j][np.isfinite(train[:, j])]
        if not len(values):
            continue
        if j in categorical:
            unique, counts = np.unique(values, return_counts=True)
            result[j] = unique[np.argmax(counts)]
        else:
            result[j] = np.median(values)
    return result


def feature_orders(importance, seed, random_repeats):
    ranked = np.argsort(-np.asarray(importance), kind="stable")
    orders = {"important": ranked, "least": ranked[::-1]}
    rng = np.random.default_rng(seed)
    for r in range(random_repeats):
        orders[f"random_{r}"] = rng.permutation(len(ranked))
    return orders


def load_importance(path, names):
    ranking = pd.read_csv(path).sort_values("feature_index")
    if ranking.feature_index.tolist() != list(range(len(names))) or ranking.feature.tolist() != names:
        raise ValueError("Ranking feature indices/names do not match inputs")
    importance = ranking.importance.to_numpy(dtype=float)
    if not np.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("Ranking importance must be finite and nonnegative")
    return importance


def mask_features(data, dropped, reference, mode):
    masked = data.copy()
    masked[:, dropped] = np.nan if mode == "nan" else reference[dropped]
    return masked


def metrics(target, prediction):
    if len(np.unique(target)) != 2:
        raise ValueError("Evaluation set must contain both classes")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite model predictions")
    return dict(roc_auc=float(roc_auc_score(target, prediction)),
                average_precision=float(average_precision_score(target, prediction)))


def predict(net, data, device, batch_size=64):
    net.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.as_tensor(data[start:start + batch_size],
                                    dtype=torch.float32, device=device)
            predictions.append(torch.sigmoid(net.inference(batch)).cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def compute_shap(net, train, names, args, out, seed):
    import shap

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(train), min(args.background_size, len(train)), replace=False)
    explain_idx = rng.choice(len(train), min(args.explain_size, len(train)), replace=False)
    background, samples = train[background_idx], train[explain_idx]
    # NaN equals NaN for masking purposes. SHAP's default numeric comparison
    # otherwise treats missing values as varying even when both sides are missing.
    def equal(a, b):
        return np.isclose(a, b, equal_nan=True)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    masker.invariants = lambda x: equal(x, background)
    explainer = shap.PermutationExplainer(
        lambda x: predict(net, np.asarray(x, dtype=np.float32), args.device, args.batch_size),
        masker, feature_names=names, seed=seed)
    explanation = explainer(samples, max_evals=(2 * train.shape[1] + 1) * args.shap_permutations,
                            batch_size=args.batch_size)
    values = np.asarray(explanation.values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite SHAP values")
    predictions = predict(net, samples, args.device, args.batch_size)
    residual = predictions - (np.asarray(explanation.base_values).reshape(-1) + values.sum(axis=1))
    importance = np.abs(values).mean(axis=0)
    ranked = np.argsort(-importance, kind="stable")
    pd.DataFrame(dict(feature_index=ranked, feature=np.asarray(names)[ranked],
                      importance=importance[ranked], rank=np.arange(1, len(names) + 1))).to_csv(
                          out / "ranking.csv", index=False)
    np.savez_compressed(out / "shap.npz", values=values, samples=samples,
                        base_values=explanation.base_values, background_indices=background_idx,
                        explain_indices=explain_idx, additivity_residual=residual)
    (out / "shap_metadata.json").write_text(json.dumps(dict(
        shap_version=shap.__version__, output="abnormal_fertilization_probability",
        ranking_source="training_only", max_abs_additivity_residual=float(np.abs(residual).max()),
        feature_names=names), indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    display_names = [re.sub(r"\s*\([^)]*\)", "", name).strip() for name in names]
    shap.summary_plot(values, samples, feature_names=display_names, show=False)
    plt.savefig(out / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.close("all")
    shap.summary_plot(values, samples, feature_names=names, plot_type="bar", show=False)
    plt.savefig(out / "shap_importance.png", bbox_inches="tight")
    plt.close("all")
    return importance


def evaluate(net, datasets, device, batch_size):
    return {name: metrics(y, predict(net, x, device, batch_size))
            for name, (x, y) in datasets.items()}


def result_rows(scores, baseline, seed, order, count, dropped, names):
    return [dict(seed=seed, order=order, removed=count, dataset=dataset,
                 dropped_features=json.dumps([names[i] for i in dropped]),
                 **score,
                 delta_roc_auc=score["roc_auc"] - baseline[dataset]["roc_auc"],
                 delta_average_precision=score["average_precision"] - baseline[dataset]["average_precision"])
            for dataset, score in scores.items()]


def plot_results(frame, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dataset in frame.dataset.unique():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for order, group in frame[frame.dataset == dataset].groupby("order"):
            group = group.sort_values("removed")
            for ax, (metric, label) in zip(axes, [("roc_auc", "ROC-AUC"), ("average_precision", "Average precision")], strict=True):
                ax.plot(group.removed, group[metric], label=order)
                ax.set(xlabel="Number of masked features", ylabel=label)
                ax.legend()
        fig.suptitle(f"{title}: {dataset}")
        fig.savefig(out / f"{title}_{dataset}.png", bbox_inches="tight")
        plt.close(fig)


def run_seed(seed, entry, args, names, categorical, external):
    out = Path(args.output) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    dataset_path, model_path = Path(entry["dataset"]), Path(entry["model"])
    adata = anndata.read_h5ad(dataset_path)
    if adata.n_vars != len(names):
        raise ValueError("Dataset feature count differs from Excel headers")
    x = np.asarray(adata.X, dtype=np.float32)
    y = np.asarray(adata.obsm["target"]).reshape(-1).astype(np.float32)
    if not np.isin(y, [0, 1]).all() or np.isinf(x).any():
        raise ValueError("Expected binary target and finite-or-NaN features")
    train_idx = np.flatnonzero(adata.obs[args.split_key].to_numpy() == "train")
    val_idx = np.flatnonzero(adata.obs[args.split_key].to_numpy() == "validate")
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Missing train or validate split")
    if args.smoke_samples:
        def subset(idx):
            if len(idx) <= args.smoke_samples:
                return idx
            selected, _ = train_test_split(idx, train_size=args.smoke_samples,
                                           stratify=y[idx], random_state=seed)
            return selected
        train_idx, val_idx = subset(train_idx), subset(val_idx)
    train_x = x[train_idx]
    val_x, val_y = x[val_idx], y[val_idx]
    datasets = {"validate": (val_x, val_y), "external": external}
    if args.smoke_samples:
        ex, ey = external
        if len(ey) > args.smoke_samples:
            idx, _ = train_test_split(np.arange(len(ey)), train_size=args.smoke_samples,
                                      stratify=ey, random_state=seed)
            datasets["external"] = (ex[idx], ey[idx])
    # Disallow mixing rankings/results from incompatible data or settings.
    config = {k: v for k, v in vars(args).items() if k not in {"stages", "overwrite", "seeds", "output", "no_summary", "device", "mask_results", "plot_counts"}}
    config.update(seed=seed, entry=entry, feature_names=names,
                  train_indices=train_idx.tolist(), validate_indices=val_idx.tolist())
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    config["input_hashes"] = {str(p): digest(p) for p in [dataset_path, model_path / "model.pt",
                                                       Path(args.data), Path(args.external)]}
    if args.ranking_file:
        config["input_hashes"][args.ranking_file] = digest(args.ranking_file)
    config_path = out / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        # Older SHAP/mask runs also recorded unused training-only settings.
        for key in ("device", "model_seed", "one_cycle_total_steps", "epochs", "lr", "weight_decay"):
            previous.pop(key, None)
        if previous != config:
            raise ValueError(f"Configuration differs in {out}; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2))
    IVF.setup_anndata(adata)
    accelerator = "cpu" if args.device == "cpu" else "gpu"
    device = "auto" if accelerator == "cpu" else [int(args.device.split(":")[-1])]
    model = IVF.load(str(model_path), adata, accelerator=accelerator, device=device)
    if model.module is None:
        raise ValueError("Loaded model has no network module")
    net = model.module.to(args.device)
    ranking_path = out / "ranking.csv"
    if args.ranking_file:
        importance = load_importance(args.ranking_file, names)
        pd.read_csv(args.ranking_file).to_csv(ranking_path, index=False)
    elif "shap" in args.stages:
        if ranking_path.exists() and not args.overwrite:
            raise FileExistsError(f"{ranking_path} exists; use --overwrite to rerun")
        importance = compute_shap(net, train_x, names, args, out, seed)
    else:
        importance = load_importance(ranking_path, names)
    orders = feature_orders(importance, args.order_seed if args.ranking_file else seed, args.random_repeats)
    if args.orders != ["all"]:
        orders = {key: value for key, value in orders.items()
                  if key in args.orders or (key.startswith("random_") and "random" in args.orders)}
        if not orders:
            raise ValueError("No selected feature orders; random requires --random-repeats > 0")
    (out / "feature_orders.json").write_text(json.dumps(
        {key: [names[i] for i in order] for key, order in orders.items()}, indent=2))
    reference = reference_values(train_x, categorical)
    pd.DataFrame({"feature": names, "reference": reference,
                  "categorical": [i in categorical for i in range(len(names))]}).to_csv(
                          out / "mask_reference.csv", index=False)
    counts = list(range(len(names))) if args.counts == "all" else sorted(set(map(int, args.counts.split(","))))
    if any(k < 0 or k >= len(names) for k in counts):
        raise ValueError("Counts must be between 0 and n_features-1")
    counts = sorted(set([0] + counts))
    if "mask" in args.stages:
        path = out / "mask_metrics.csv"
        if path.exists() and not args.overwrite:
            raise FileExistsError(str(path))
        baseline = evaluate(net, datasets, args.device, args.batch_size)
        rows = []
        for order_name, order in orders.items():
            for k in counts:
                dropped = order[:k]
                masked = {name: (mask_features(a, dropped, reference, args.mask_mode), b)
                          for name, (a, b) in datasets.items()}
                scores = evaluate(net, masked, args.device, args.batch_size)
                rows.extend(result_rows(scores, baseline, seed, order_name, k, dropped, names))
                pd.DataFrame(rows).to_csv(path, index=False)
        plot_results(pd.DataFrame(rows), out, "mask")


def load_plot_data(path, seeds, counts):
    """Load mask aggregate metrics only; never touches models or clinical samples."""
    required = {"seed", "order", "removed", "dataset", "roc_auc", "average_precision",
                "delta_roc_auc", "delta_average_precision"}
    frame = pd.read_csv(path)
    if "pr_auc" in frame.columns:
        raise ValueError("PR-AUC results cannot be converted to AP; restore AP results or rerun the mask stage")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing metric columns in {path}")
    if frame.duplicated(["seed", "order", "removed", "dataset"]).any():
        raise ValueError(f"Duplicate experiment rows in {path}")
    frame = frame[frame.seed.isin(seeds) & frame.removed.isin(counts)].copy()
    if frame.empty:
        raise ValueError(f"No selected rows in {path}")
    if not np.isfinite(frame[list(required - {"order", "dataset"})].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite metrics in {path}")
    for (order, dataset), group in frame.groupby(["order", "dataset"]):
        expected = {(s, k) for s in seeds for k in counts}
        if set(zip(group.seed, group.removed, strict=True)) != expected:
            raise ValueError(f"Incomplete selected grid: mask/{order}/{dataset}")
    return frame


def plot_analysis(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = list(range(45)) if args.plot_counts == "all" else sorted(set(map(int, args.plot_counts.split(","))))
    seeds = list(map(int, args.seeds.split(",")))
    if not counts or len(set(seeds)) != len(seeds):
        raise ValueError("Plot counts must be nonempty and seeds unique")
    frame = load_plot_data(args.mask_results, seeds, counts)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [name for name in ["validate", "external"] if name in frame.dataset.unique()]
    if set(datasets) != set(frame.dataset.unique()):
        raise ValueError("Expected validate/external datasets")
    targets = [out / "mask_boxplots.pdf"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Plot outputs exist; use --overwrite to regenerate")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    dataset_labels = {"validate": "Validation set", "external": "2025 test set"}
    metrics_to_plot = [("roc_auc", "ROC-AUC"), ("average_precision", "AP")]
    for kind in ["mask"]:
        figure_width = max(12, len(counts) * 0.65)
        fig, axes = plt.subplots(len(datasets), 2, squeeze=False,
                                 figsize=(figure_width, 4.5 * len(datasets)))
        for row, dataset in enumerate(datasets):
            data = frame[frame.dataset == dataset].copy()
            if data.empty:
                raise ValueError(f"No {kind} data for {dataset}")
            hue = "order"
            for ax, (metric, label) in zip(axes[row], metrics_to_plot, strict=True):
                sns.boxplot(data=data, x="removed", y=metric, hue=hue,
                            order=counts, ax=ax, showmeans=True,
                            legend=(row == 0 and metric == "roc_auc"),
                            width=0.8,
                            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                       "markersize": 3},
                            fliersize=2, linewidth=0.8)
                ax.set(xlabel="Number of masked features", ylabel=label,
                       title=f"{dataset_labels[dataset]} — {label}")
                if metric == "roc_auc":
                    ax.set_ylim(0, 1)
                if len(counts) > 15:
                    ax.tick_params(axis="x", rotation=90, labelsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{kind}_boxplots.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Boxplots written to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="JSON mapping seed to dataset/model paths")
    parser.add_argument("--data", default="ivf/task/data.xlsx", help="Feature schema; not used for ranking")
    parser.add_argument("--external", default="ivf/task/data.2025.xlsx")
    parser.add_argument("--output", help="Default: analyse/ablation_outputs/<stage or experiments>")
    parser.add_argument("--seeds", default=",".join(map(str, range(15))))
    parser.add_argument("--stages", nargs="+", choices=["shap", "mask", "plot"], default=["shap", "mask"])
    root = Path(__file__).resolve().parent / "ablation_outputs"
    parser.add_argument("--mask-results", default=str(root / "mask/dataset2/mask_all.csv"))
    parser.add_argument("--plot-counts", default="0,1,3,5,10,15,20,30,40,44",
                        help="Mask displayed counts; all displays 0..44")
    parser.add_argument("--ranking-file", help="Shared ranking.csv from the designated baseline; no per-split SHAP")
    parser.add_argument("--order-seed", type=int, default=2, help="Shared random-control seed with --ranking-file")
    parser.add_argument("--orders", nargs="+", choices=["all", "important", "least", "random"],
                        default=["all"], help="Select masking orders")
    parser.add_argument("--counts", default="all", help="all = 0..44, or comma-separated counts")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--split-key", default="split_key")
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--explain-size", type=int, default=128)
    parser.add_argument("--shap-permutations", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--categorical-indices", default="auto", help="auto: training columns with <10 unique values; or indices")
    parser.add_argument("--mask-mode", choices=["reference", "nan"], default="reference")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-samples", type=int, default=0, help="Stratified subsample per split; diagnostic only")
    parser.add_argument("--no-summary", action="store_true", help="Workers write only their own seed directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if "all" in args.orders and args.orders != ["all"]:
        parser.error("--orders all cannot be combined with other orders")
    if args.output is None:
        category = args.stages[0] if len(args.stages) == 1 else "experiments"
        args.output = str(Path(__file__).resolve().parent / "ablation_outputs" / category)
    if "plot" in args.stages:
        if args.stages != ["plot"]:
            parser.error("Run --stages plot separately from computation stages")
        plot_analysis(args)
        return
    if not args.manifest:
        parser.error("--manifest is required for computation stages")
    if args.ranking_file and "shap" in args.stages:
        parser.error("Use --stages mask with --ranking-file; generate the baseline SHAP separately")
    for field in ["batch_size", "background_size", "explain_size", "shap_permutations"]:
        if getattr(args, field) < 1:
            parser.error(f"{field} must be positive")
    if args.random_repeats < 0 or (args.smoke_samples and args.smoke_samples < 4):
        parser.error("Invalid random repeat count or smoke sample size")
    if args.device != "cpu" and not args.device.startswith("cuda:"):
        parser.error("device must be cpu or cuda:N")
    schema = pd.read_excel(args.data)
    names = schema.columns[2:].tolist()
    test = pd.read_excel(args.external)
    if test.columns[2:].tolist() != names:
        raise ValueError("External feature schema/order does not match")
    labels = test.iloc[:, 1].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("External labels must be binary without missing values")
    external = (test.iloc[:, 2:].to_numpy(dtype=np.float32), 1 - labels.astype(np.float32))
    manifest = json.loads(Path(args.manifest).read_text())
    for seed in map(int, args.seeds.split(",")):
        entry = manifest[str(seed)]
        # Infer categories from this split's training data, not the external set.
        if args.categorical_indices == "auto":
            adata = anndata.read_h5ad(entry["dataset"])
            train = np.asarray(adata.X)[adata.obs[args.split_key].to_numpy() == "train"]
            categorical = {i for i in range(len(names)) if len(np.unique(train[:, i][np.isfinite(train[:, i])])) < 10}
        else:
            categorical = set(map(int, args.categorical_indices.split(","))) if args.categorical_indices else set()
            if not categorical.issubset(range(len(names))):
                raise ValueError("Categorical index out of bounds")
        run_seed(seed, entry, args, names, categorical, external)
    # Summary from this invocation's seeds only, not stale results from other runs.
    if args.no_summary:
        return
    for stage in ["mask"]:
        if stage not in args.stages:
            continue
        frames = [pd.read_csv(Path(args.output) / f"seed_{s}" / f"{stage}_metrics.csv")
                  for s in map(int, args.seeds.split(","))]
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(Path(args.output) / f"{stage}_all.csv", index=False)
        frame.groupby(["order", "removed", "dataset"])[
            ["roc_auc", "average_precision", "delta_roc_auc", "delta_average_precision"]
        ].agg(["mean", "std", "count"]).to_csv(Path(args.output) / f"{stage}_summary.csv")


if __name__ == "__main__":
    main()
