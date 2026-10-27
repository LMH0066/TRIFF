import numpy as np
import pandas as pd
import pytest

from ivf.task.analyse.benchmark_summary import (
    DATASETS,
    MODELS,
    compute_metrics,
    make_table,
    plot_boxplots,
    summarize,
)


def save_predictions(path):
    for seed in [0, 1]:
        for tag in MODELS.values():
            for suffix in ["", ".2025"]:
                scores = [0.1, 0.4, 0.35, 0.8] if not suffix else [0.5] * 4
                pd.DataFrame({"targets": [0, 0, 1, 1], "predictions": scores}).to_csv(
                    path / f"predictions.{seed}{tag}{suffix}.csv", index=False)


def test_ap_and_paired_difference(tmp_path):
    save_predictions(tmp_path)
    frame = compute_metrics(tmp_path, [0, 1])
    assert len(frame) == 30
    assert set(frame.columns) == {"seed", "model", "dataset", "roc_auc", "average_precision"}
    for dataset, expected in zip(DATASETS, [5 / 6, 0.5, 1 / 3], strict=True):
        np.testing.assert_allclose(frame.loc[frame.dataset == dataset, "average_precision"], expected)
    summary = summarize(frame)
    assert (summary["count"] == 2).all()
    assert (summary["std"] == 0).all()
    assert make_table(summary).shape == (6, 5)
    output = tmp_path / "performance_boxplots.pdf"
    plot_boxplots(frame, output)
    assert output.read_bytes().startswith(b"%PDF")


def test_combined_is_scored_from_predictions(tmp_path):
    save_predictions(tmp_path)
    for seed in [0, 1]:
        for tag in MODELS.values():
            pd.DataFrame({"targets": [0, 0, 1, 1], "predictions": [0.1, 0.2, 0.8, 0.9]}).to_csv(
                tmp_path / f"predictions.{seed}{tag}.combined.csv", index=False)
    frame = compute_metrics(tmp_path, [0, 1], include_combined=True)
    assert len(frame) == 40
    assert (frame.loc[frame.dataset == "Training + validation set", "average_precision"] == 1).all()
    assert make_table(summarize(frame)).shape == (8, 5)


def test_training_extraction_uses_original_ids(tmp_path):
    import json

    import anndata

    save_predictions(tmp_path)
    adata = anndata.AnnData(np.zeros((4, 1)))
    adata.obs_names = ["a", "b", "c", "d"]
    adata.obs["split_key"] = ["train", "train", "validate", "validate"]
    adata.obsm["target"] = np.array([[0], [1], [0], [1]])
    data_path = tmp_path / "data.h5ad"
    adata.write_h5ad(data_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({str(seed): {"dataset": str(data_path)} for seed in [0, 1]}))
    for seed in [0, 1]:
        for tag in MODELS.values():
            pd.DataFrame({"sample_id": ["d", "a", "c", "b"],
                          "targets": [1, 0, 0, 1], "predictions": [0.1, 0.1, 0.9, 0.9]}).to_csv(
                tmp_path / f"predictions.{seed}{tag}.combined.csv", index=False)
    frame = compute_metrics(tmp_path, [0, 1], training_manifest=manifest)
    assert (frame.loc[frame.dataset == "Training set", "average_precision"] == 1).all()
    assert "Training + validation set" not in frame.dataset.values
    assert make_table(summarize(frame)).shape == (8, 5)
    path = tmp_path / "predictions.0.lr.combined.csv"
    data = pd.read_csv(path)
    data.loc[data.sample_id == "b", "sample_id"] = "missing"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Training sample IDs missing"):
        compute_metrics(tmp_path, [0, 1], training_manifest=manifest)


def test_reject_invalid_scores_and_label_order(tmp_path):
    save_predictions(tmp_path)
    path = tmp_path / "predictions.0.hgb.csv"
    pd.DataFrame({"targets": [1, 0, 0, 1], "predictions": [0.1] * 4}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Target order"):
        compute_metrics(tmp_path, [0, 1])
    pd.DataFrame({"targets": [0, 0, 1, 1], "predictions": [np.nan] * 4}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite scores"):
        compute_metrics(tmp_path, [0, 1])
