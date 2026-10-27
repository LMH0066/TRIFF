# TRIFF

**A Temporally Robust Interpretable model for total Fertilization Failure Risk in IVF/ICSI**

## Project Structure

The core components of the IVF project are located in the `ivf/` directory:

- `ivf/_data.py`: Handles data setup and splitting into training, validation, and test sets using `AnnData` objects.
- `ivf/_model.py`: Defines the main `IVF` model class, which orchestrates data processing, model training, prediction, and interpretation.
- `ivf/_module.py`: Contains the neural network architecture, including `PositionEncoding`, `TransformerEmbedding`, `FocalLoss`, and the `NET` (Transformer Encoder-based) module.
- `ivf/_train.py`: Implements the `IVFTrainingPlan` for managing the training lifecycle, including optimizers, learning rate schedulers, and epoch-wise logging.
- `ivf/_utils.py`: Provides utility functions and classes, such as `LOSS_KEYS` for defining loss function names and `DefaultMetric` for custom metric computation during training.

The `ivf/task/` directory contains examples and scripts for specific tasks:

- `ivf/task/run.py`: A script for creating datasets and training models.
- `ivf/task/data.xlsx`: Example data file.
- `ivf/task/README.md`: Instructions for using `run.py`.

## Installation

```bash
conda create -n TRIFF python=3.10
conda activate TRIFF
pip install poetry
```

Then, clone the repository and install the dependencies:

```bash
git clone https://github.com/LMH0066/TRIFF.git
cd TRIFF
poetry install
```

The Python package (`ivf`) and model class (`IVF`) retain their names for checkpoint compatibility. Existing installations may keep their original directory and Conda environment names; these do not affect the TRIFF model name.

## Usage

### Data Preparation

The `run.py` script in `ivf/task/` can be used to create a dataset from an Excel file.

```bash
poetry run python ivf/task/run.py create-dataset --input_file ivf/task/data.xlsx --split_key split_key --output_file ivf/task/dataset
```

- `--input_file`: Path to your input Excel data.
- `--split_key`: Column name in your Excel file used for splitting the data (e.g., into train/validation/test sets).
- `--output_file`: Base name for the output dataset files.

### Model Training

After preparing the dataset, you can train the model using the `run.py` script:

```bash
poetry run python ivf/task/run.py train --dataset_path ivf/task/dataset --split_key split_key
```

- `--dataset_path`: Path to the prepared dataset.
- `--split_key`: The same split key used during dataset creation.

### Example Walkthrough

Refer to `ivf/task/README.md` for a quick example of how to use the `run.py` script for data preparation and model training.

```bash
python run.py create-dataset --input_file data.xlsx --split_key split_key --output_file dataset
python run.py train --dataset_path dataset --split_key split_key
```

## License

This project is licensed under the MIT License.
