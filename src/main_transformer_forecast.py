"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
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
from termcolor import colored, cprint
from datetime import datetime

import os
import sys

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

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
create_folder_if_not_exists(WORKING_DIR)

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

ALL_COLUMNS = [
    "year",
    "date_x",
    "date_y",
    "time_x",
    "time_y",
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
    "line 1 pump speed 0",
    "line 1 pump speed 1",
    "line 1 pump speed 2",
    "line 1 pump speed 3",
    "line 1 pump speed 4",
    "line 1 pump speed 5",
    "line 1 pump speed 6",
    "line 1 pump speed 7",
    "line 1 pump speed 8",
    "line 1 pump speed 9",
    "line 1 pump speed 10",
    "line 2 pump speed 0",
    "line 2 pump speed 1",
    "line 2 pump speed 2",
    "line 2 pump speed 3",
    "line 2 pump speed 4",
    "line 2 pump speed 5",
    "line 2 pump speed 6",
    "line 2 pump speed 7",
    "line 2 pump speed 8",
    "line 2 pump speed 9",
    "line 2 pump speed 10",
    "PAC pump 1 speed 0",
    "PAC pump 1 speed 1",
    "PAC pump 1 speed 2",
    "PAC pump 1 speed 3",
    "PAC pump 1 speed 4",
    "PAC pump 1 speed 5",
    "PAC pump 1 speed 6",
    "PAC pump 1 speed 7",
    "PAC pump 1 speed 8",
    "PAC pump 1 speed 9",
    "PAC pump 1 speed 10",
    "PAC pump 2 speed 0",
    "PAC pump 2 speed 1",
    "PAC pump 2 speed 2",
    "PAC pump 2 speed 3",
    "PAC pump 2 speed 4",
    "PAC pump 2 speed 5",
    "PAC pump 2 speed 6",
    "PAC pump 2 speed 7",
    "PAC pump 2 speed 8",
    "PAC pump 2 speed 9",
    "PAC pump 2 speed 10",
]

Y_COLUMNS = [
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]

TGT_COLUMNS = "line 1 pump speed"

# Read scaling factors
def load_scaling_factors() -> dict:
    """Load scaling factors from data/processed

    Returns:
        dict: scaling factors in dict format 
            result:
                column: (scaling factors, stddev)\n
                column: (scaling factors, stddev)\n
                column: (scaling factors, stddev)
    """
    data = pd.read_csv(
        os.path.join(
        DATA_DIR, "processed", "scaling_factors.csv"
        ),
        index_col=0,
    )
    result = {}
    for index, row in data.iterrows():
        result[row[0]] = (
            row[1],
            row[2]
            )
    return result

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

def generate_skip_columns():
    """
    Skip the one-hot label columns
    """
    skip_columns = []
    for column in Y_COLUMNS:
        for i in range(11):
            skip_columns.append(f"{column} {i}")
    return skip_columns


def generate_src_columns() -> None:
    skip_columns = generate_skip_columns()
    src_columns = ALL_COLUMNS.copy()
    for column in skip_columns:
        src_columns.remove(column)
    for column in Y_COLUMNS:
        src_columns.remove(column)
    #src_columns.append(TGT_COLUMNS)
    return src_columns

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":     24,     # 4 hours
    "forecast_length":      1,      # 1 hour
    "embedding_dimension":  512,
    "batch_size":           256,    # 32 is pretty small
    "train_val_split_ratio":0.667,
    "scaling_factors":      load_scaling_factors(),
    "national_standards":   load_national_standards(),
    "src_columns":          generate_src_columns(),
    "tgt_columns":          TGT_COLUMNS,
    "tgt_y_columns":        TGT_COLUMNS,
}
INPUT_FEATURE_SIZE = len(HYPERPARAMETER["src_columns"])
FORECAST_FEATURE_SIZE = len(TGT_COLUMNS)
cprint(f"Source columns: {HYPERPARAMETER['src_columns']}", "green")
cprint(f"Target columns: {HYPERPARAMETER['tgt_columns']}", "green")

# Subprocess

def csv_to_loader(
        csv_dir: str,
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

def load(path: str, train_val_split: float=0.8) -> list:
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
    train = []
    for csv_file in csv_files[:int(len(csv_files)*train_val_split)]:
        current_csv = os.path.join(path, csv_file)
        try:
            train.append(csv_to_loader(current_csv))
        except Exception:
            continue

    # Compose validation loader
    val = []
    for csv_file in csv_files[int(len(csv_files)*train_val_split):]:
        current_csv = os.path.join(path, csv_file)
        try:
            val.append(csv_to_loader(current_csv))
        except Exception:
            continue
    
    return train, val

def main() -> None:
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")

    # path = "/".join(INPUT_DATA.split('/')[:-1])
    # name = INPUT_DATA.split('/')[-1].split(".")[0]

    train_loaders, val_loaders = load(INPUT_DATA, train_val_split=HYPERPARAMETER["train_val_split_ratio"])

    # Model
    model: TimeSeriesTransformer = TimeSeriesTransformer(
        INPUT_FEATURE_SIZE,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"],
        device,
        HYPERPARAMETER,
        model_name = MODEL_NAME,
        embedding_dimension = HYPERPARAMETER["embedding_dimension"],
    ).to(device)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Dump hyper parameters
    model.dump_hyper_parameters(WORKING_DIR)

    # Training
    loss_fn = nn.MSELoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    metrics.append(mae)
    ## Optimizer
    lr = 0.001  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer = optimizer,
        T_0 = 5,
    )
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        patience = 2,
    )
    t_epoch = TrackerEpoch(100)
    t_loss = TrackerLoss(10, model)
    t_train_loss = TrackerLoss(-1, model)
    # Validation logger
    train_logger = TransformerForecasterVisualLogger(
        "train",
        WORKING_DIR,
        runtime_plotting = True,
        which_to_plot = [0],
        plot_interval = 10,
    )
    val_logger = TransformerForecasterVisualLogger(
        "val",
        WORKING_DIR,
        runtime_plotting = True,
        which_to_plot = [0],
        in_one_figure = True,
        plot_interval = 2,
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
                train_logger.signal_new_epoch()
                bar.refresh()

                val_loss = model.val(
                    val_loaders, 
                    loss_fn, 
                    metrics,
                    vis_logger = val_logger,
                    )
                val_logger.signal_new_epoch()
                scheduler_0.step()
                scheduler_1.step()
                # scheduler_2.step(val_loss)

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

    save_model(model, WORKING_DIR)
    save_model(model_best_train, WORKING_DIR)
    visualize_loss(t_loss, WORKING_DIR, f"{MODEL_NAME}_val")
    visualize_loss(t_train_loss, WORKING_DIR, f"{MODEL_NAME}_train")

    # Signal new epoch is needed for triggering non_runtime_plotting of VisualLoggers
    train_logger.signal_new_epoch()
    val_logger.signal_new_epoch()
    return

if __name__ == "__main__":
    main()

