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

from helper import *
from transformer import TimeSeriesTransformer, TransformerDataset, TransformerForecasterVisualLogger, transformer_collate_fn

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd

from tqdm import tqdm
from typing import Tuple
from termcolor import colored, cprint

import os
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
"""
"""
The input data should be in folder "range" inside the model dir that needs to be tested,
So that it can be reproduced later, even if the data processing changed
The data needs to be split into train and validation folders *manually*
"""
INPUT_DATA = sys.argv[1]
# The model suppose to be store in the folder "range"
MODEL_DIR = sys.argv[2]

RANGE_DIR = settings.RANGE_DIR

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print()

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":     24,     # 4 hours
    "forecast_length":      6,      # 1 hour
    "batch_size":           128,    # 32 is pretty small
    "train_val_split_ratio":0.667,
}
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
INPUT_FEATURE_SIZE = 16 + 1
FORECAST_FEATURE_SIZE = 1

# Subprocess
def csv_to_loader(
        csv_dir: str,
        skip_columns: list = [],
        ) -> torch.utils.data.DataLoader:
    # Read csv
    data = pd.read_csv(
        csv_dir,
        low_memory=False,
        index_col=0,
        parse_dates=["timestamp"],
    )
    
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

def load(path: str) -> list:
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
    for csv_file in csv_files:
        current_csv = os.path.join(path, csv_file)
        try:
            val.append(csv_to_loader(current_csv, skip_columns=SKIP_COLUMNS))
        except Exception:
            continue
    
    return val

def load_hyper_parameters(dir: str) -> Tuple[list, dict]:
    args_dir = os.path.join(dir, "args.json")
    kwargs_dir = os.path.join(dir, "kwargs.json")
    with open(args_dir, "r", encoding="utf-8") as f:
        args = json.load(f)

    with open(kwargs_dir, "w", encoding="utf-8") as f:
        kwargs = json.load(f)
    return args, kwargs

# Main

def main():
    # Load data
    vals = load(INPUT_DATA)

    # Load hyper parameters
    args, kwargs = load_hyper_parameters(MODEL_DIR)

    # Load model
    model = TimeSeriesTransformer()
    # Run inference

    # Track the performance

    # Visualize the performance

    return