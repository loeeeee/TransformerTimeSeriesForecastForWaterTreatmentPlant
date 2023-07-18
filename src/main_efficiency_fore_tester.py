"""
This script tests the efficiency of a transformer forecaster,
The tests is to read in the validation dataset, and run the model on those data for inference,
However, instead of input the original water quality as an input, it runs the national standard
for water quality as an input for its output water quality.
In this case, I am making a big assumption that the water quality of the output can be regulated by 
the transformer model.
To compare the model against the original PID control system, I would add up the total run speed of
the pump as it is an indication of amount of chemical drug added into the pool.
"""
import settings # Get config
import utils # TODO: recode this

from helper import to_numeric_and_downcast_data
from transformer import TimeSeriesTransformer, TransformerDataset, transformer_collate_fn, GREEN, BLACK, generate_square_subsequent_mask, TransformerForecastPlotter, NotEnoughData

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from termcolor import colored, cprint
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer

import os
import re
import sys
import json

"""
The preferred folder structure for the testing is,
range/
├─ transformer_forecast_512_0/
│  ├─ train_prediction_trend/
│  ├─ validation_data/
│  ├─ val_prediction_trend/
│  ├─ args.json
│  ├─ kwargs.json
│  ├─ trans_for_23-07-07-10-02.pt
│  ├─ trans_for_23-07-07-10-02_best_trained.pt
│  ├─ trans_for_23-07-07-10-02_train_loss_history.png
│  ├─ trans_for_23-07-07-10-02_val_loss_history.png
├─ transformer_forecast_512_1/
...
"""
"""
The input data should be in folder "range" inside the model dir that needs to be tested,
So that it can be reproduced later, even if the data processing changed
The data needs to be split into train and validation folders *manually*
"""
# The model suppose to be store in the folder "range"
MODEL_DIR = sys.argv[1]
RANGE_DIR = settings.RANGE_DIR
ROOT_DIR = settings.ROOT_DIR
DEVICE = settings.DEVICE

print(colored(f"Read from {MODEL_DIR}", "black", "on_green"))
print()

# HYPERPARAMETER
HYPERPARAMETER = None # Load from model

# Helper
def _apply_quantiles_transformer(
        n_quantiles: int, 
        quantiles: np.array, 
        references: np.array, 
        n_features_in: int, 
        value: float,
        ) -> float:
    """Apply quantiles transform on individual values

    Args:
        n_quantiles (int): _description_
        quantiles (np.array): _description_
        references (np.array): _description_
        n_features_in (int): _description_
        value (float): Value to be transformed

    Returns:
        float: Transform result
    """
    # Resume Transformer
    scaler = QuantileTransformer()
    scaler.n_quantiles_ = n_quantiles
    scaler.quantiles_ = quantiles
    scaler.references_ = references
    scaler.n_features_in_ = n_features_in
    # Do the transform
    scaled_value = scaler.transform(value)

    return scaled_value

def _apply_standard_scaler(
        mean: float,
        stddev: float,
        value: float,
        ) -> float:
    """Apply standard scaler on individual values

    Args:
        mean (float): _description_
        stddev (float): _description_
        value (float): Value to be transformed

    Returns:
        float: Transform result
    """
    # Resume transformer
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = stddev
    # Do the transform
    scaled_value = scaler.transform(value)
    return scaled_value

def _get_scaled_national_standards(
        og_national_standards: dict, 
        scaling_factors: dict,
        name_mapping: dict,
        ) -> dict:
    """Generate scaled national standards based on the scale of the data

    Args:
        og_national_standards (dict): original national standards GB18918-2002
        scaling_factors (dict): scaling factors for scaling the data
        name_mapping (dict): maps column name in data to name in national standards

    Returns:
        dict: scaled national standards with its original name
    """
    """
    scaled_national_standards = {}
    for name in name_mapping:
        scaler = StandardScaler()
        scaler.mean_ = scaling_factors[name][0]
        scaler.scale_ = scaling_factors[name][1]
        scaled_national_standards[name_mapping[name]] = scaler.transform(
            np.asarray(
                og_national_standards[name_mapping[name]]
            ).reshape(1, -1)
            ).reshape(-1)[0]
    """
    scaled_national_standards = {}
    # cprint(f"{scaling_factors}", "green")
    for name in name_mapping:
        scaler = QuantileTransformer()
        # cprint(f"{}", "green")
        scaler = scaler.set_params(**scaling_factors[name][4])
        scaler.n_quantiles_ = scaling_factors[name][0]
        scaler.quantiles_ = np.array(scaling_factors[name][1])
        scaler.references_ = np.array(scaling_factors[name][2])
        scaler.n_features_in_ = scaling_factors[name][3]

        scaled_national_standards[name_mapping[name]] = scaler.transform(
            np.asarray(
                og_national_standards[name_mapping[name]]
            ).reshape(1, -1)
        ).reshape(-1)[0]
    return scaled_national_standards

# Subprocess

def efficiency_test_loader(
        csv_dir: str,
        scaling_factors: dict,
        national_standards: dict,
        ) -> torch.utils.data.DataLoader:
    # Read csv
    data = pd.read_csv(
        csv_dir,
        low_memory=False,
        index_col=0,
        parse_dates=["timestamp"],
    )
    
    # Scale the national standard
    name_mapping = {
        "outlet COD": "COD",
        "outlet ammonia nitrogen": "ammonia nitrogen",
        "outlet total nitrogen": "total nitrogen",
        "outlet phosphorus": "total phosphorus",
    }
    scaled_national_standards = _get_scaled_national_standards(
        national_standards,
        scaling_factors,
        name_mapping,
    )
    
    # Apply national standards to the data
    for data_name, std_name  in name_mapping.items():
        data[data_name] = scaled_national_standards[std_name]

    # Downcast data
    data = to_numeric_and_downcast_data(data)
    
    # Make sure data is in ascending order by timestamp
    data.sort_values(by=["timestamp"], inplace=True)

    # Split data
    src = data[HYPERPARAMETER["src_columns"]]
    tgt = data[HYPERPARAMETER["tgt_columns"]]
    
    # Drop data that is too short for the prediction
    if len(tgt.values) < HYPERPARAMETER["forecast_length"] + HYPERPARAMETER["knowledge_length"]:
        raise NotEnoughData
    
    src = torch.tensor(src.values, dtype=torch.float32, device=DEVICE)
    tgt = torch.tensor(tgt.values, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    dataset = TransformerDataset(
        src,
        tgt,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"],
        DEVICE,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        HYPERPARAMETER["batch_size"],
        drop_last=False,
        collate_fn=transformer_collate_fn,
    )
    return loader

def load_data(
        path: str, 
        scaling_factors: dict, 
        national_standards: dict,
        train_validation_ratio: float,
        ) -> list[DataLoader]:
    csv_files = []
    pattern = re.compile(r'\d+')

    # Iterate over all files in the directory
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            csv_files.append(filename)

    # Sort the CSV files based on the numbers in their filenames
    csv_files.sort(key=lambda x: int(pattern.search(x).group()))
    
    if len(csv_files) == 0:
        raise FileNotFoundError
    
    # Compose validation loader
    val = []
    for csv_file in csv_files: #[int(len(csv_files) * train_validation_ratio):]:
        current_csv = os.path.join(path, csv_file)
        try:
            val.append(
                efficiency_test_loader(
                    current_csv, 
                    scaling_factors, 
                    national_standards,
                    )
                )
        except NotEnoughData:
            continue
    
    return val

def load_hyper_parameters(dir: str) -> Tuple[list, dict]:
    args_dir = os.path.join(dir, "args.json")
    kwargs_dir = os.path.join(dir, "kwargs.json")
    with open(args_dir, "r", encoding="utf-8") as f:
        args = json.load(f)

    with open(kwargs_dir, "r", encoding="utf-8") as f:
        kwargs = json.load(f)
    return args, kwargs

def load_model(dir: str, args: list, kwargs: dict) -> Tuple[TimeSeriesTransformer, TimeSeriesTransformer]:
    """Load model from given directory

    Args:
        dir (str): Where to load from

    Returns:
        TimeSeriesTransformer: A trained model
    """
    for filename in os.listdir(dir):
        if filename.endswith('_best_trained.pt'):
            model_dir = os.path.join(
                dir,
                filename
            )
            best_trained_model = TimeSeriesTransformer(*args, **kwargs).to(DEVICE)
            best_trained_model.load_state_dict(torch.load(model_dir))
        elif filename.endswith('.pt'):
            model_dir = os.path.join(
                dir,
                filename
            )
            best_validated_model = TimeSeriesTransformer(*args, **kwargs).to(DEVICE)
            best_validated_model.load_state_dict(torch.load(model_dir))
    return best_trained_model, best_validated_model

def evaluation(model: TimeSeriesTransformer, dataloaders: DataLoader, working_dir: str) -> None:
    # Run inference
    total_batches = sum([len(dataloader) for dataloader in dataloaders])

    cprint(f"Testing the {model.model_name} model", "green")
    model.eval()
    loss_fn = nn.L1Loss()
    plotter = TransformerForecastPlotter(
        f"{model.model_name}_test",
        working_dir,
        runtime_plotting = False,
        which_to_plot = [0, HYPERPARAMETER["forecast_length"]-1],
        in_one_figure = False,
        format = "svg",
    )
    with torch.no_grad():
        test_loss = 0
        test_saving = 0
        correct = 0
        bar = tqdm(
            total       = total_batches, 
            position    = 1,
            colour      = GREEN,
            )
        batch_cnt = 0
        for dataloader in dataloaders:
            for (src, tgt, tgt_y) in dataloader:
                src, tgt, tgt_y = src.to(DEVICE), tgt.to(DEVICE), tgt_y.to(DEVICE)
                                
                with torch.autocast(device_type=DEVICE):
                    pred = model(src, tgt)
                    # TODO: Check accuracy calculation
                    temp_loss = loss_fn(pred, tgt_y).item()
                    temp_saving = torch.sub(tgt_y, pred).sum()
                test_loss += temp_loss # Multiply by batch count
                test_saving += temp_saving

                correct += (pred == tgt_y).type(torch.int8).sum().item()
                plotter.append(tgt_y, pred)
                bar.set_description(desc=f"Loss: {(test_loss/(1+batch_cnt)):.3f}, Saving: {temp_saving}", refresh=True)
                batch_cnt += 1
                bar.update()
            plotter.signal_new_dataloader()
        bar.colour = BLACK
        bar.close()
    note = f"{str(type(loss_fn))[7:-2].split('.')[-1]}: {test_loss}, Saving: {test_saving}"
    plotter.signal_finished(note=note)
    return

# Main
def main():
    # Load data
    working_dir = os.path.join(
        ROOT_DIR,
        MODEL_DIR
    )

    # Load hyper parameters
    args, kwargs = load_hyper_parameters(working_dir)

    # Load model
    best_trained_model, best_validated_model = load_model(working_dir, args, kwargs)
    
    # Metadata
    metadata = best_validated_model.get_metadata()
    global HYPERPARAMETER
    HYPERPARAMETER = metadata
    
    # Load data
    data_dir = os.path.join(
        working_dir,
        "data"
    )
    vals = load_data(
        data_dir, 
        metadata["x_scaling_factors"], 
        metadata["national_standards"],
        metadata["train_val_split_ratio"],
        )
    cprint(f"Number of datasets: {len(vals)}", "green")

    # Start evaluation
    cprint("Start evaluation", "green")
    evaluation(best_validated_model, vals, working_dir)
    evaluation(best_trained_model, vals, working_dir)
    return

if __name__ == "__main__":
    main()