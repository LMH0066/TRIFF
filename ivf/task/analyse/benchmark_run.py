"""Fit benchmark baselines on training splits; refuse to overwrite existing outputs.

Run with SCIPY_ARRAY_API=1 for installed TabPFN/sklearn compatibility.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from tabpfn import TabPFNRegressor

from ivf import IVF
from ivf.task.analyse.ablation import predict


def save_prediction(path, scores, targets, ids):
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(targets) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid predictions: {path}")
    pd.DataFrame({"sample_id": ids, "predictions": scores, "targets": targets}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=root / "ablation_manifest.json")
    parser.add_argument("--external", type=Path, default=root.parent / "data.2025.xlsx")
    parser.add_argument("--output", type=Path, default=root / "benchmark")
    parser.add_argument("--seeds", default=",".join(map(str, range(15))))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--tabpfn-model", type=Path, default=Path.home() / ".cache/tabpfn/tabpfn-v2.6-regressor-v2.6_default.ckpt")
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(",")))
    if len(set(seeds)) != len(seeds) or not seeds:
        parser.error("Seeds must be unique and nonempty")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if not args.tabpfn_model.is_file():
        raise FileNotFoundError(args.tabpfn_model)
    external = pd.read_excel(args.external)
    labels = external.iloc[:, 1].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("External labels must be binary")
    ex = external.iloc[:, 2:].to_numpy(dtype=np.float32)
    ey = 1 - labels
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True)
    provenance = {"settings": {k: str(v) for k, v in vars(args).items()},
                  "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "tabpfn", "torch", "scvi-tools"]},
                  "tabpfn_sha256": hashlib.sha256(args.tabpfn_model.read_bytes()).hexdigest(),
                  "note": "Baselines fitted only on train. Combined predictions include training samples. Historical predictions are not mixed into this run.",
                  "completed_seeds": []}
    for seed in seeds:
        print(f"Seed {seed}: loading data", flush=True)
        entry = manifest[str(seed)]
        adata = anndata.read_h5ad(entry["dataset"])
        train = np.flatnonzero(adata.obs.split_key.to_numpy() == "train")
        val = np.flatnonzero(adata.obs.split_key.to_numpy() == "validate")
        if not len(train) or not len(val):
            raise ValueError("Missing train/validate partition")
        combined = np.concatenate([train, val])
        x = np.asarray(adata.X, dtype=np.float32)
        y = np.asarray(adata.obsm["target"]).reshape(-1)
        if x.shape[1] != ex.shape[1] or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid feature count or labels")
        sets = {"": (x[val], y[val], adata.obs_names[val].tolist()),
                ".combined": (x[combined], y[combined], adata.obs_names[combined].tolist()),
                ".2025": (ex, ey, [f"external_row_{i}" for i in range(len(ey))])}
        imputer = SimpleImputer(strategy="mean")
        clean_train = imputer.fit_transform(x[train])
        baselines = {"lr": LinearRegression(fit_intercept=False, positive=True),
                     "hgb": HistGradientBoostingRegressor(random_state=42),
                     "rf": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)}
        for name, estimator in baselines.items():
            print(f"Seed {seed}: fit/predict {name}", flush=True)
            estimator.fit(clean_train, y[train])
            for suffix, (features, target, ids) in sets.items():
                save_prediction(args.output / f"predictions.{seed}.{name}{suffix}.csv",
                                estimator.predict(imputer.transform(features)), target, ids)
        print(f"Seed {seed}: fit/predict TabPFN (local v2.6 weights)", flush=True)
        tab = TabPFNRegressor(model_path=str(args.tabpfn_model), device=args.device, random_state=0)
        tab.fit(x[train], y[train])
        for suffix, (features, target, ids) in sets.items():
            # Bounded prediction chunks control GPU memory; fitted model is unchanged.
            scores = np.concatenate([tab.predict(features[start:start + 512])
                                     for start in range(0, len(features), 512)])
            save_prediction(args.output / f"predictions.{seed}.tab{suffix}.csv", scores, target, ids)
        del tab
        torch.cuda.empty_cache()
        print(f"Seed {seed}: predict existing Effristor checkpoint", flush=True)
        IVF.setup_anndata(adata)
        model = IVF.load(entry["model"], adata, accelerator="gpu", device=[int(args.device.split(":")[-1])])
        if model.module is None:
            raise ValueError("No Effristor network")
        net = model.module.to(args.device)
        for suffix, (features, target, ids) in sets.items():
            save_prediction(args.output / f"predictions.{seed}{suffix}.csv",
                            predict(net, features, args.device, batch_size=64), target, ids)
        del net, model
        torch.cuda.empty_cache()
        provenance["completed_seeds"].append(seed)
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
        print(f"Seed {seed}: complete", flush=True)


if __name__ == "__main__":
    main()
