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

from helper import to_numeric_and_downcast_data, get_best_device
from transformer import TimeSeriesTransformer, TransformerDataset, transformer_collate_fn, GREEN, BLACK, generate_square_subsequent_mask, TransformerForecastPlotter

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from termcolor import colored, cprint
from sklearn.preprocessing import StandardScaler

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
DEVICE = get_best_device()

print(colored(f"Read from {MODEL_DIR}", "black", "on_green"))
print()

# HYPERPARAMETER
HYPERPARAMETER = None # Load from model

Y_COLUMNS = [
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]
def generate_skip_columns():
    """
    Skip the one-hot label columns
    """
    skip_columns = []
    for column in Y_COLUMNS:
        for i in range(11):
            skip_columns.append(f"{column} {i}")
    return skip_columns
SKIP_COLUMNS = generate_skip_columns()
TGT_COLUMNS = "line 1 pump speed"

# Helper

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
    return scaled_national_standards

# Subprocess
def csv_to_loader(
        csv_dir: str,
        scaling_factors: dict,
        national_standards: dict,
        skip_columns: list = [],
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
    
    # Remove skip columns, skipping those information for the classifier
    data = data.drop(
        columns=skip_columns,
    )

    # Split data
    src = data.drop(
        columns=Y_COLUMNS
    )
    src[TGT_COLUMNS] = data[TGT_COLUMNS]
    tgt = data[TGT_COLUMNS]

    print(src.head(10))
    # Drop data that is too short for the prediction
    if len(tgt.values) < HYPERPARAMETER["forecast_length"] + HYPERPARAMETER["knowledge_length"]:
        print(f"Drop {colored(csv_dir, 'red' )}")
        raise Exception
    
    src = torch.tensor(src.values, dtype=torch.float32)
    tgt = torch.tensor(tgt.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TransformerDataset(
        src,
        tgt,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"]
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        HYPERPARAMETER["batch_size"],
        drop_last=False,
        collate_fn=transformer_collate_fn
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
                csv_to_loader(
                    current_csv, 
                    scaling_factors, 
                    national_standards,
                    skip_columns=SKIP_COLUMNS,
                    )
                )
        except Exception:
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

def load_model(dir: str, args: list, kwargs: dict, device: str) -> Tuple[TimeSeriesTransformer, TimeSeriesTransformer]:
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
            best_trained_model = TimeSeriesTransformer(*args, **kwargs).to(device)
            best_trained_model.load_state_dict(torch.load(model_dir))
        elif filename.endswith('.pt'):
            model_dir = os.path.join(
                dir,
                filename
            )
            best_validated_model = TimeSeriesTransformer(*args, **kwargs).to(device)
            best_validated_model.load_state_dict(torch.load(model_dir))
    return best_trained_model, best_validated_model

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
    best_trained_model, best_validated_model = load_model(working_dir, args, kwargs, DEVICE)
    
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
        metadata["scaling_factors"], 
        metadata["national_standards"],
        metadata["train_val_split_ratio"],
        )
    cprint(f"{len(vals)}", "green")

    # Run inference
    total_batches = sum([len(dataloader) for dataloader in vals])

    # Generate masks
    src_mask = generate_square_subsequent_mask(
        dim1=metadata["forecast_length"],
        dim2=metadata["knowledge_length"]
        ).to(DEVICE)
    
    tgt_mask = generate_square_subsequent_mask(
        dim1=metadata["forecast_length"],
        dim2=metadata["forecast_length"]
        ).to(DEVICE)
    
    # Start evaluation
    cprint("Start evaluation", "green")
    best_validated_model.eval()
    loss_fn = nn.L1Loss()
    plotter = TransformerForecastPlotter(
        "efficiency_test",
        working_dir,
        runtime_plotting = True,
        which_to_plot = [0, HYPERPARAMETER["forecast_length"]-1],
        in_one_figure = False,
    )
    with torch.no_grad():
        test_loss = 0
        correct = 0
        bar = tqdm(
            total       = total_batches, 
            position    = 1,
            colour      = GREEN,
            )
        batch_cnt = 0
        for dataloader in vals:
            for (src, tgt, tgt_y) in dataloader:
                src, tgt, tgt_y = src.to(DEVICE), tgt.to(DEVICE), tgt_y.to(DEVICE)

                pred = best_validated_model(src, tgt, src_mask, tgt_mask)
                # TODO: Check accuracy calculation
                test_loss += loss_fn(pred, tgt_y).item() * tgt_y.shape[1] # Multiply by batch count
                # tqdm.write(f"{loss_fn(pred, tgt_y).item()} {tgt_y.shape}")
                correct += (pred == tgt_y).type(torch.int8).sum().item()
                plotter.append(tgt_y, pred)
                bar.set_description(desc=f"Loss: {(test_loss/(1+batch_cnt)):.3f}", refresh=True)
                batch_cnt += 1
                bar.update()
            plotter.signal_new_dataloader()
        bar.colour = BLACK
        bar.close()

    plotter.signal_new_epoch()
    test_loss /= total_batches
    correct /= total_batches
    tqdm.write("")

    # Track the performance

    # Visualize the performance

    return

if __name__ == "__main__":
    main()