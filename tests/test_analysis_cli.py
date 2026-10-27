import numpy as np
import torch
from click.testing import CliRunner

from ivf import IVF
from ivf.task.analyse.run import cli


def test_analysis_command_surface():
    assert set(cli.commands) == {"predict-and-evaluate", "interpret"}
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_prediction_and_interpretation_preserved():
    from ivf import setup_data

    features = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    data = setup_data(features, [0, 1])
    IVF.setup_anndata(data)
    model = IVF(data, module_params={"d_model": 8, "nhead": 2,
                                   "num_encoder_layers": 1, "dim_feedforward": 8,
                                   "dropout": 0})
    # This synthetic instance checks inference mechanics without fitting parameters.
    model.is_trained_ = True
    prediction = model.predict(data, batch_size=2)
    assert prediction.shape == (2, 1)
    assert np.isfinite(np.asarray(prediction.X)).all()
    attribution = model.interpret(features[:1], internal_batch_size=8)
    assert attribution.shape == (1, 2)
    assert torch.isfinite(attribution).all()
