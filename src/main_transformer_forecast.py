"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config

from helper import *
from transformer import WaterFormer, TransformerForecasterVisualLogger, transformer_collate_fn, WaterFormerDataset

import torch
from torch import nn

import numpy as np
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
RAW_DIR = settings.RAW_DIR

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

X_COLUMNS = RAW_COLUMNS[:-4]
Y_COLUMNS = RAW_COLUMNS[-4:]

TGT_COLUMNS = RAW_COLUMNS[-4]

# Read scaling factors
def load_scaled_national_standards() -> dict:
    """Load scaled national standards from data

    Returns:
        dict: scaling factors in dict format 
            result:
                column: value\n
                column: value\n
                column: value
    """
    scaled_national_standards_path = os.path.join(
        DATA_DIR, "scaled_national_standards.json"
        )
    with open(scaled_national_standards_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":             32,     # 4 hours
    "forecast_length":              2,      # 1 hour
    "input_sequence_size":          None,   # Generated on the fly
    "output_sequence_size":         None,   # Generated on the fly
    "spatiotemporal_encoding_size": None,   # Generated on the fly
    "batch_size":                   32,    # 32 is pretty small
    "train_val_split_ratio":        0.8,
    "scaled_national_standards":    load_scaled_national_standards(),
    "src_columns":                  X_COLUMNS,
    "tgt_columns":                  TGT_COLUMNS,
    "tgt_y_columns":                TGT_COLUMNS,
    "random_seed":                  42,
}

INPUT_FEATURE_SIZE = len(HYPERPARAMETER["src_columns"])
FORECAST_FEATURE_SIZE = len(TGT_COLUMNS)

cprint(f"Source columns: {HYPERPARAMETER['src_columns']}", "green")
cprint(f"Target columns: {HYPERPARAMETER['tgt_columns']}", "green")

# Helper

# Subprocess

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

    # Read data and do split
    data = pd.read_csv(
        INPUT_DATA,
        low_memory=False,
        index_col=0,
        parse_dates=["timestamp"],
    )
    train_size = int(data.shape[0] * HYPERPARAMETER["train_val_split_ratio"])
    val_size = data.shape[0] - train_size
    
    def dataframe_to_loader(
            data: pd.DataFrame,
            ) -> torch.utils.data.DataLoader:
        # Downcast data
        data = to_numeric_and_downcast_data(data.copy())
        
        # Make sure data is in ascending order by timestamp
        data.sort_values(by=["timestamp"], inplace=True)
        
        # Split data
        src = np.array(data[HYPERPARAMETER["src_columns"]].values)
        tgt = np.expand_dims(np.array(data[HYPERPARAMETER["tgt_columns"]].values), axis=1)
        timestamp = data.reset_index(names="timestamp")["timestamp"].to_numpy(dtype=np.datetime64)
        
        dataset = WaterFormerDataset(
            src,
            tgt,
            timestamp,
            HYPERPARAMETER["knowledge_length"] * src.shape[1],
            HYPERPARAMETER["forecast_length"] * tgt.shape[1],
            device=DEVICE,
        )
        HYPERPARAMETER["input_sequence_size"] = dataset.input_sequence_size
        HYPERPARAMETER["output_sequence_size"] = dataset.output_sequence_size
        HYPERPARAMETER["spatiotemporal_encoding_size"] = dataset.spatiotemporal_encoding_size

        loader = torch.utils.data.DataLoader(
            dataset,
            HYPERPARAMETER["batch_size"],
            drop_last=False,
            collate_fn=transformer_collate_fn,
        )
        return loader
    
    train_loader = dataframe_to_loader(data.head(train_size))
    val_loader = dataframe_to_loader(data.tail(val_size))
    
    # Model
    model: WaterFormer = WaterFormer(
        HYPERPARAMETER["input_sequence_size"],
        HYPERPARAMETER["output_sequence_size"],
        HYPERPARAMETER["spatiotemporal_encoding_size"],
        device=DEVICE
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
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_0, scheduler_1])
    t_epoch = TrackerEpoch(50)
    t_val_loss = TrackerLoss(10, model)
    t_train_loss = TrackerLoss(-1, model)
    # Validation logger
    train_logger = TransformerForecasterVisualLogger(
        "train",
        WORKING_DIR,
        runtime_plotting = True,
        plot_interval = 10,
        format="png",
    )
    val_logger = TransformerForecasterVisualLogger(
        "val",
        WORKING_DIR,
        runtime_plotting = True,
        in_one_figure = False,
        plot_interval = 3,
        format="png",
    )
    print(colored("Training:", "black", "on_green"), "\n")
    
    # Performance profiler
    """
    profiler =  torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{WORKING_DIR}/profiling'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()
    """
    with tqdm(total=t_epoch.max_epoch, unit="epoch", position=0) as bar:
        while True:
            try:
                lr = scheduler.get_last_lr()[0]
                tqdm.write(colored("--------------------------------------------", "cyan", attrs=["bold"]))
                tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
                tqdm.write(colored(f"Learning rate {lr:.5f}", "green"))
                tqdm.write(colored(f"Recent training loss trend: {t_train_loss.get_trend(3):.5f}", "green"))
                tqdm.write(colored(f"Recent validation loss trend: {t_val_loss.get_trend(3):.5f}", "green"))
                tqdm.write(colored(f"Best training loss: {t_train_loss.lowest_loss:.5f}", "green"))
                tqdm.write(colored(f"Best validation loss: {t_val_loss.lowest_loss:.5f}", "green"))
                tqdm.write(colored("--------------------------------------------", "cyan", attrs=["bold"]))

                train_loss = model.learn(
                    train_loader, 
                    loss_fn, 
                    optimizer, 
                    visual_logger = train_logger,
                    #profiler=profiler,
                )
                note = f"{str(type(loss_fn))[7:-2].split('.')[-1]}: {train_loss}"
                train_logger.signal_new_epoch(note=note)
                bar.refresh()

                val_loss = model.val(
                    val_loader, 
                    loss_fn, 
                    metrics,
                    visual_logger = val_logger,
                    )
                note = f"{str(type(loss_fn))[7:-2].split('.')[-1]}: {val_loss}"
                val_logger.signal_new_epoch(note=note)

                scheduler.step()

                if not t_val_loss.check(val_loss, model):
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
    #profiler.stop()

    print(colored("Done!", "black", "on_green"), "\n")
    
    # Bring back the best known model
    model = t_val_loss.get_best_model()
    print(colored(f"The best model has the validation loss of {t_val_loss.lowest_loss}", "cyan"))
    model_best_train = t_train_loss.get_best_model()
    model_best_train.model_name += "_best_trained"
    cprint(f"Best trained model has an train loss of {t_train_loss.lowest_loss}", "cyan")

    # Dump hyper parameters
    model.dump_hyper_parameters(WORKING_DIR)

    # Save model
    save_model(model, WORKING_DIR)
    save_model(model_best_train, WORKING_DIR)
    visualize_loss(t_val_loss, WORKING_DIR, f"{MODEL_NAME}_val")
    visualize_loss(t_train_loss, WORKING_DIR, f"{MODEL_NAME}_train")

    # Save data
    save_data(INPUT_DATA, WORKING_DIR)

    # Signal new epoch is needed for triggering non_runtime_plotting of VisualLoggers
    train_logger.signal_finished()
    val_logger.signal_finished()
    return

if __name__ == "__main__":
    main()

