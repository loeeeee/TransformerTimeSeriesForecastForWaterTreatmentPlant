"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config
import utils # TODO: recode this

from helper import *
from transformer import ClassifierTransformer, TransformerDataset, TransformerClassifierVisualLogger, transformer_collate_fn

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
    MODEL_NAME = f"trans_cls_{formatted_time}"
VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
create_folder_if_not_exists(WORKING_DIR)

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":     24,     # 4 hours
    "forecast_length":      6,      # 1 hour
    "batch_size":           128,    # 32 is pretty small
    "train_val_split_ratio":0.667,
}
SKIP_COLUMNS = [
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]
def generate_target_columns() -> list:
    tgt_columns = []
    for i in range(11):
        tgt_columns.append(f"line 1 pump speed {i}")
    return tgt_columns
TGT_COLUMNS = generate_target_columns()
def generate_y_columns() -> list:
    y_columns = []
    for column in SKIP_COLUMNS:
        for i in range(11):
            y_columns.append(f"{column} {i}")
    for column in TGT_COLUMNS:
        y_columns.remove(column)
    return y_columns
SKIP_COLUMNS.extend(generate_y_columns())
INPUT_FEATURE_SIZE = 16 + 11 # Features and historical results
FORECAST_FEATURE_SIZE = 11

# Subprocess

def train_test(dataloader: DataLoader,
               model: ClassifierTransformer,
               loss_fn: any,
               optimizer: torch.optim,
               device: str,
               forecast_length: int,
               knowledge_length: int) -> None:
    """
    Only for the purpose of debugging
    """
    tqdm.write(f"Length of the dataloader: {len(dataloader)}")
    bar = tqdm(total=len(dataloader), position=1)
    for i, (src, tgt, tgt_y) in enumerate(dataloader):
        # """
        tqdm.write(colored(i, "green", "on_red"))
        tqdm.write(f"src shape: {src.shape}\ntgt shape: {tgt.shape}\ntgt_y shape: {tgt_y.shape}")
        # """
        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=forecast_length
            )
        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=knowledge_length
            )
        tqdm.write(f"tgt_mask shape: {tgt_mask.shape}\nsrc_mask: {src_mask.shape}\n")
        bar.update()
    bar.close()
    return

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
    
    # Remove skip columns
    data = data.drop(
        columns=SKIP_COLUMNS,
    )

    # Split data
    tgt = data[TGT_COLUMNS]
    src = data.copy()

    # Drop data that is too short for the prediction
    if len(tgt.values) < HYPERPARAMETER["forecast_length"] + HYPERPARAMETER["knowledge_length"]:
        print(f"Drop {colored(csv_dir, 'red' )}")
        raise Exception
    
    src = torch.tensor(src.values)
    tgt = torch.tensor(tgt.values, dtype=torch.float32)
    
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

"""
year                          int8
date_x                     float32
date_y                     float32
time_x                     float32
time_y                     float32
inlet flow                 float32
inlet COD                  float32
inlet ammonia nitrogen     float32
inlet total nitrogen       float32
inlet phosphorus           float32
outlet COD                 float32
outlet ammonia nitrogen    float32
outlet total nitrogen      float32
outlet phosphorus          float32
line 1 nitrate nitrogen    float32
line 2 nitrate nitrogen    float32
"""

def main() -> None:
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")

    # path = "/".join(INPUT_DATA.split('/')[:-1])
    # name = INPUT_DATA.split('/')[-1].split(".")[0]

    train_loaders, val_loaders = load(INPUT_DATA, train_val_split=0.667)

    # Model
    model = ClassifierTransformer(
        INPUT_FEATURE_SIZE,
        FORECAST_FEATURE_SIZE,
        HYPERPARAMETER,
        model_name = MODEL_NAME,
        embedding_dimension = 512
    ).to(device)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Dump hyper parameters
    model.dump_hyper_parameters(WORKING_DIR)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    metrics.append(mae)
    metrics.append(mse)
    ## Optimizer
    lr = 0.1  # learning rate
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
    train_logger = TransformerClassifierVisualLogger(
        "train",
        WORKING_DIR,
        meta_data = HYPERPARAMETER,
        runtime_plotting = True,
        which_to_plot = [0,int(HYPERPARAMETER["forecast_length"]/2), HYPERPARAMETER["forecast_length"]-1],
        plot_interval = 10,
    )
    val_logger = TransformerClassifierVisualLogger(
        "val",
        WORKING_DIR,
        meta_data = HYPERPARAMETER,
        runtime_plotting = True,
        which_to_plot = [0,int(HYPERPARAMETER["forecast_length"]/2), HYPERPARAMETER["forecast_length"]-1],
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
                """
                train_test(
                    train_loader, 
                    model, 
                    loss_fn, 
                    optimizer, 
                    device,
                    HYPERPARAMETER["forecast_length"],
                    HYPERPARAMETER["knowledge_length"]
                    )
                break
                """
                train_loss = model.learn(
                    train_loaders, 
                    loss_fn, 
                    optimizer, 
                    device,
                    HYPERPARAMETER["forecast_length"],
                    HYPERPARAMETER["knowledge_length"],
                    vis_logger = train_logger,
                )
                train_logger.signal_new_epoch()
                bar.refresh()

                val_loss = model.val(
                    val_loaders, 
                    loss_fn, 
                    device,
                    HYPERPARAMETER["forecast_length"],
                    HYPERPARAMETER["knowledge_length"], 
                    metrics,
                    WORKING_DIR,
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

