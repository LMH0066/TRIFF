import numpy as np
import pandas as pd
import pytest

from ivf.task.analyse.benchmark_summary import DATASETS, MODELS, statistical_comparisons


def example_frame():
    return pd.DataFrame([
        {"seed": seed, "model": model, "dataset": dataset,
         "roc_auc": 0.8, "average_precision": 0.3}
        for seed in range(15) for model in MODELS for dataset in DATASETS
    ])


def test_all_comparisons_zero_difference_and_bh():
    frame = example_frame()
    result = statistical_comparisons(frame.sample(frac=1, random_state=5))
    assert len(result) == 70
    assert (result.n_pairs == 15).all()
    assert (result.p_value == 1).all()
    assert (result.p_adjusted_bh == 1).all()
    assert not result.reject_bh_0_05.any()
    assert set(result.family_size) == {5, 10}


def test_pairing_direction_and_invalid_grid():
    frame = example_frame()
    frame.loc[frame.model == "TRIFF", "roc_auc"] += 0.05
    result = statistical_comparisons(frame)
    row = result[(result.dataset == DATASETS[0]) & (result.metric == "ROC-AUC")
                 & (result.group_a == "LR") & (result.group_b == "TRIFF")].iloc[0]
    assert row.mean_difference_a_minus_b == pytest.approx(-0.05)
    assert row.p_value < 0.05
    assert row.p_adjusted_bh >= row.p_value
    with pytest.raises(ValueError, match="Incomplete"):
        statistical_comparisons(frame.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        statistical_comparisons(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "roc_auc"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        statistical_comparisons(frame)
