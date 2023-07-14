"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config

from helper import *
from transformer import TimeSeriesTransformer, TransformerDataset, TransformerForecasterVisualLogger, transformer_collate_fn, NotEnoughData

import torch
from torch import nn

import pandas as pd

from tqdm import tqdm
from typing import Literal
from termcolor import colored, cprint
from datetime import datetime

import os
import sys
import json
import shutil
import random

INPUT_DATA = sys.argv[1]
try:
    MODEL_NAME = sys.argv[2]
except IndexError:
    print("No model name specified, using current time stamp")
    # Get the current timestamp
    current_time = datetime.now()
    # Format the timestamp as YY-MM-DD-HH-MM
    formatted_time = current_time.strftime("%y-%m-%d-%H-%M")
    MODEL_NAME = f"trans_for_{formatted_time}"
VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
DEVICE = settings.DEVICE

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
create_folder_if_not_exists(WORKING_DIR)

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

RAW_COLUMNS = [
    "inlet flow",
    "inlet COD",
    "inlet ammonia nitrogen",
    "inlet total nitrogen",
    "inlet phosphorus",
    "outlet COD",
    "outlet ammonia nitrogen",
    "outlet total nitrogen",
    "outlet phosphorus",
    "line 1 nitrate nitrogen",
    "line 2 nitrate nitrogen",
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]

TIME_COLUMNS = [
    "year",
    "date_x",
    "date_y",
    "time_x",
    "time_y",
]

X_COLUMNS = RAW_COLUMNS[:-4]
Y_COLUMNS = RAW_COLUMNS[-4:]

TGT_COLUMNS = RAW_COLUMNS[-4]

# Read scaling factors
def load_scaling_factors(x_or_y: Literal['x', 'y']) -> dict:
    """Load scaling factors from data/processed

    Returns:
        dict: scaling factors in dict format 
            result:
                column: (scaling factors, stddev)\n
                column: (scaling factors, stddev)\n
                column: (scaling factors, stddev)
    """
    scaling_factors_path = os.path.join(
        DATA_DIR, f"{x_or_y}_scaling_factors.json"
        )
    with open(scaling_factors_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_national_standards() -> dict:
    """Load national standard from data/GB

    Returns:
        dict: scaling factors in dict format 
            result:
                chemical: standard\n
                chemical: standard\n
                chemical: standard
    """
    data = pd.read_csv(
        os.path.join(
        DATA_DIR, "GB18918-2002.csv"
        ),
        index_col=0,
    )
    result = {}
    for index, row in data.iterrows():
        result[row[0]] = row[1]
    return result

def generate_src_columns() -> None:
    _ = X_COLUMNS.copy()
    _.extend(TIME_COLUMNS)
    return _

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":     12,     # 4 hours
    "forecast_length":      2,      # 1 hour
    "embedding_dimension":  512,
    "batch_size":           128,    # 32 is pretty small
    "train_val_split_ratio":0.7,
    "x_scaling_factors":    load_scaling_factors('x'),
    "y_scaling_factors":    load_scaling_factors('y'),
    "national_standards":   load_national_standards(),
    "src_columns":          generate_src_columns(),
    "tgt_columns":          TGT_COLUMNS,
    "tgt_y_columns":        TGT_COLUMNS,
    "random_seed":          42,
}

INPUT_FEATURE_SIZE = len(HYPERPARAMETER["src_columns"])
FORECAST_FEATURE_SIZE = len(TGT_COLUMNS)

cprint(f"Source columns: {HYPERPARAMETER['src_columns']}", "green")
cprint(f"Target columns: {HYPERPARAMETER['tgt_columns']}", "green")

# Helper

# Subprocess
def csv_to_loader(
        csv_dir: str,
        device: str,
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
    
    # Split data
    src = data[HYPERPARAMETER["src_columns"]]
    tgt = data[HYPERPARAMETER["tgt_columns"]]

    # Drop data that is too short for the prediction
    if len(tgt.values) < HYPERPARAMETER["forecast_length"] + HYPERPARAMETER["knowledge_length"]:
        print(f"Drop {colored(csv_dir, 'red' )}")
        raise NotEnoughData
    
    src = torch.tensor(src.values, dtype=torch.float32, device=device)
    tgt = torch.tensor(tgt.values, dtype=torch.float32, device=device).unsqueeze(1)
    
    dataset = TransformerDataset(
        src,
        tgt,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"],
        device,
        )

    loader = torch.utils.data.DataLoader(
        dataset,    
        HYPERPARAMETER["batch_size"],
        drop_last=False,
        collate_fn=transformer_collate_fn,
    )
    return loader

def load(path: str, device: str, train_val_split: float=0.8) -> list:
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
    
    # Compose train loader
    train_val = []
    for csv_file in csv_files:
        current_csv = os.path.join(path, csv_file)
        try:
            train_val.append(csv_to_loader(current_csv, device))
        except NotEnoughData:
            continue
    random.seed(HYPERPARAMETER["random_seed"])
    random.shuffle(train_val)
    train = train_val[:int(len(csv_files)*train_val_split)]
    val = train_val[int(len(csv_files)*train_val_split):]

    cprint(f"{len(train)}, {len(val)}", "green")
    return train, val

def save_data(
        src: str, 
        dst: str, 
        symlinks: bool=False, 
        ignore: Union[list, None]=None
        ) -> None:
    """Copy data to model_dir/data

    Args:
        src (str): source_dir
        dst (str): destination_dir
        symlinks (bool, optional): if copy symbolic link. Defaults to False.
        ignore (Union[list, None], optional): what to ignore. Defaults to None.
    """
    dst = os.path.join(dst, "data")
    create_folder_if_not_exists(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
    return

# Main

def main() -> None:
    print(colored(f"Using {DEVICE} for training", "black", "on_green"), "\n")

    train_loaders, val_loaders = load(
        INPUT_DATA, 
        DEVICE,
        train_val_split=HYPERPARAMETER["train_val_split_ratio"]
        )
    # Model
    model: TimeSeriesTransformer = TimeSeriesTransformer(
        INPUT_FEATURE_SIZE,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"],
        DEVICE,
        HYPERPARAMETER,
        model_name = MODEL_NAME,
        embedding_dimension = HYPERPARAMETER["embedding_dimension"],
        num_of_decoder_layers = 8,
        num_of_encoder_layers = 8,
    ).to(DEVICE)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Training
    loss_fn = nn.MSELoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    metrics.append(mae)
    ## Optimizer
    lr = 0.01  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer = optimizer,
        T_0 = 5,
    )
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        patience = 2,
    )
    t_epoch = TrackerEpoch(300)
    t_loss = TrackerLoss(10, model)
    t_train_loss = TrackerLoss(-1, model)
    # Validation logger
    train_logger = TransformerForecasterVisualLogger(
        "train",
        WORKING_DIR,
        runtime_plotting = True,
        plot_interval = 20,
        format="svg",
    )
    val_logger = TransformerForecasterVisualLogger(
        "val",
        WORKING_DIR,
        runtime_plotting = True,
        in_one_figure = True,
        plot_interval = 5,
        format="svg",
    )
    print(colored("Training:", "black", "on_green"), "\n")
    with tqdm(total=t_epoch.max_epoch, unit="epoch", position=0) as bar:
        while True:
            try:
                lr = scheduler_0.get_last_lr()[0]
                tqdm.write("----------------------------------")
                tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
                tqdm.write(colored(f"Learning rate {lr}", "green"))
                tqdm.write("----------------------------------")

                train_loss = model.learn(
                    train_loaders, 
                    loss_fn, 
                    optimizer, 
                    vis_logger = train_logger,
                )
                note = f"{str(type(loss_fn))[7:-2].split('.')[-1]}: {train_loss}"
                train_logger.signal_new_epoch(note=note)
                bar.refresh()

                val_loss = model.val(
                    val_loaders, 
                    loss_fn, 
                    metrics,
                    vis_logger = val_logger,
                    )
                note = f"{str(type(loss_fn))[7:-2].split('.')[-1]}: {val_loss}"
                val_logger.signal_new_epoch(note=note)

                scheduler_0.step()
                scheduler_1.step()
                # scheduler_2.step(train_loss)

                if not t_loss.check(val_loss, model):
                    tqdm.write(colored("Validation loss no longer decrease, finish training", "green", "on_red"))
                    break
                if not t_epoch.check():
                    tqdm.write(colored("Maximum epoch reached. Finish training", "green", "on_red"))
                    break
                if not t_train_loss.check(train_loss, model):
                    tqdm.write(colored("Training loss no longer decrease, finish training", "green", "on_red"))
                    break
                bar.update()
            except KeyboardInterrupt:
                tqdm.write(colored("Early stop triggered by Keyboard Input", "green", "on_red"))
                break
        bar.close()

    print(colored("Done!", "black", "on_green"), "\n")
    
    # Bring back the best known model
    model = t_loss.get_best_model()
    print(colored(f"The best model has the validation loss of {t_loss.lowest_loss}", "cyan"))
    model_best_train = t_train_loss.get_best_model()
    model_best_train.model_name += "_best_trained"
    cprint(f"Best trained model has an train loss of {t_train_loss.lowest_loss}", "cyan")

    # Dump hyper parameters
    model.dump_hyper_parameters(WORKING_DIR)

    # Save model
    save_model(model, WORKING_DIR)
    save_model(model_best_train, WORKING_DIR)
    visualize_loss(t_loss, WORKING_DIR, f"{MODEL_NAME}_val")
    visualize_loss(t_train_loss, WORKING_DIR, f"{MODEL_NAME}_train")

    # Save data
    save_data(INPUT_DATA, WORKING_DIR)

    # Signal new epoch is needed for triggering non_runtime_plotting of VisualLoggers
    train_logger.signal_finished()
    val_logger.signal_finished()
    return

if __name__ == "__main__":
    main()

