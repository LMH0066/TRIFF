import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from ivf._module import NET
from ivf.task.analyse.ablation import (
    feature_orders,
    load_importance,
    load_plot_data,
    mask_features,
    metrics,
    plot_analysis,
    reference_values,
    result_rows,
)


def test_plot_data_rejects_incomplete_and_duplicate_rows(tmp_path):
    rows = [{"seed": seed, "order": "important", "removed": k, "dataset": d,
             "roc_auc": 0.8, "average_precision": 0.3,
             "delta_roc_auc": 0, "delta_average_precision": 0}
            for seed in [0, 1] for k in [0, 1] for d in ["validate", "external"]]
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(load_plot_data(path, [0, 1], [0, 1])) == 8
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        load_plot_data(path, [0, 1], [0, 1])
    pd.DataFrame(rows + rows[:1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_plot_data(path, [0, 1], [0, 1])


def test_plot_uses_only_mask_results(tmp_path):
    path = tmp_path / "mask_all.csv"
    pd.DataFrame([
        {"seed": 0, "order": "important", "removed": 0, "dataset": dataset,
         "roc_auc": 0.8, "average_precision": 0.3,
         "delta_roc_auc": 0, "delta_average_precision": 0}
        for dataset in ["validate", "external"]
    ]).to_csv(path, index=False)
    args = argparse.Namespace(mask_results=path, seeds="0", plot_counts="0",
                              output=tmp_path / "plots", overwrite=False)
    plot_analysis(args)
    assert [p.name for p in args.output.iterdir()] == ["mask_boxplots.pdf"]
    with pytest.raises(FileExistsError):
        plot_analysis(args)


def test_plot_rejects_pr_auc_results(tmp_path):
    path = tmp_path / "pr_auc.csv"
    pd.DataFrame([{"pr_auc": 0.3}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rerun the mask stage"):
        load_plot_data(path, [0], [0])


@pytest.mark.parametrize("prediction, expected", [
    ([0.1, 0.4, 0.35, 0.8], 5 / 6),
    ([0.5, 0.5, 0.5, 0.5], 0.5),
    ([0.8, 0.35, 0.4, 0.1], 0.5),
])
def test_ap_uses_average_precision(prediction, expected):
    # Non-perfect and tied scores distinguish AP from trapezoidal PR-AUC.
    score = metrics(np.array([0, 0, 1, 1]), np.array(prediction))
    assert score["average_precision"] == pytest.approx(expected)
    assert "pr_auc" not in score
    row = result_rows({"external": score},
                      {"external": {"roc_auc": 0.5, "average_precision": 0.5}},
                      0, "important", 1, [0], ["age"])[0]
    assert row["delta_average_precision"] == pytest.approx(expected - 0.5)


def test_shared_ranking_schema_and_values(tmp_path):
    path = tmp_path / "ranking.csv"
    path.write_text("feature_index,feature,importance,rank\n1,b,0.9,1\n0,a,0.2,2\n")
    np.testing.assert_equal(load_importance(path, ["a", "b"]), [0.2, 0.9])
    with pytest.raises(ValueError):
        load_importance(path, ["b", "a"])
    path.write_text("feature_index,feature,importance,rank\n0,a,nan,1\n1,b,0.2,2\n")
    with pytest.raises(ValueError):
        load_importance(path, ["a", "b"])


def test_reference_training_only_and_missing():
    train = np.array([[1, 1, np.nan], [9, 2, np.nan], [5, 2, np.nan]], dtype=float)
    reference = reference_values(train, {1})
    np.testing.assert_equal(reference, [5, 2, np.nan])


def test_mask_preserves_other_features_and_input():
    data = np.array([[1., 2., 3.]])
    masked = mask_features(data, [1], np.array([0., 7., 0.]), "reference")
    np.testing.assert_equal(masked, [[1, 7, 3]])
    np.testing.assert_equal(data, [[1, 2, 3]])
    assert np.isnan(mask_features(data, [1], np.zeros(3), "nan")[0, 1])


def test_orders_are_complete_and_reproducible():
    orders = feature_orders([2, 9, 1], 42, 2)
    np.testing.assert_equal(orders["important"], [1, 0, 2])
    np.testing.assert_equal(orders["least"], [2, 0, 1])
    for name, order in orders.items():
        assert sorted(order) == [0, 1, 2]
        np.testing.assert_equal(order, feature_orders([2, 9, 1], 42, 2)[name])


@pytest.mark.parametrize("shape", [(4, 1), (1, 1), (1, 45), (1, 4, 1)])
def test_inference_preserves_batch_with_one_feature(shape):
    net = NET("X", "target", d_model=8, nhead=2,
              num_encoder_layers=1, dim_feedforward=8, dropout=0)
    net.eval()
    batch = shape[1] if len(shape) == 3 else shape[0]
    with torch.no_grad():
        assert net.inference(torch.ones(shape)).shape == (batch, 1)


def test_metrics_and_delta():
    score = metrics(np.array([0, 1]), np.array([0.1, 0.9]))
    assert score == {"roc_auc": 1., "average_precision": 1.}
    row = result_rows({"external": score}, {"external": score}, 0,
                      "important", 1, [0], ["age"])[0]
    assert row["delta_roc_auc"] == 0
    assert row["delta_average_precision"] == 0
    assert row["dropped_features"] == '["age"]'
    with pytest.raises(ValueError):
        metrics([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        metrics([0, 1], [np.nan, 0.2])
