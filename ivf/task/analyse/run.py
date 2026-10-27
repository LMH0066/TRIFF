import anndata
import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import ivf
from ivf import IVF


@click.command()
@click.option("--model_path", type=str)
@click.option("--adata_path", type=str)
@click.option("--xlsx_path", type=str, default="")
@click.option("--output_path", type=str, default="predictions.csv")
def predict_and_evaluate(
    model_path: str,
    adata_path: str,
    xlsx_path: str = "",
    output_path: str = "predictions.csv",
):
    adata = anndata.read_h5ad(adata_path)
    model = IVF.load(model_path, adata)
    if xlsx_path == "":
        val_indices = np.where(adata.obs["split_key"] == "validate")[0]
        predictions = model.predict(adata, val_indices, batch_size=64)
    else:
        data = pd.read_excel(xlsx_path)
        dataset = ivf.setup_data(
            data.iloc[:, 2:].values,
            (~data.iloc[:, 1].values.astype(bool)).astype(int),
        )
        IVF.setup_anndata(dataset)
        predictions = model.predict(dataset, batch_size=64)
    predictions.X = torch.sigmoid(torch.tensor(predictions.X)).numpy()
    pd.DataFrame({
        "predictions": predictions.X.flatten(),
        "targets": predictions.obsm["target"].flatten(),
    }).to_csv(output_path, index=False)


@click.command()
@click.option("--model_path", type=str)
@click.option("--adata_path", type=str)
@click.option("--output_path", type=str, default="interpret_result.csv")
def interpret(model_path: str, adata_path: str, output_path: str = "interpret_result.csv"):
    adata = anndata.read_h5ad(adata_path)
    model = IVF.load(model_path, adata)
    interpretations = []
    for i in tqdm(adata.obs_names, desc="Generating interpretations", unit="sample"):
        interpretations.append(model.interpret(adata[i].X))
    interpretations = torch.stack(interpretations).squeeze()
    pd.DataFrame(interpretations.numpy()).to_csv(output_path, index=False)


@click.group()
def cli():
    pass


cli.add_command(predict_and_evaluate)
cli.add_command(interpret)

if __name__ == "__main__":
    cli()
