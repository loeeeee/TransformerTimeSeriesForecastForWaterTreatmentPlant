"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config
import utils # TODO: recode this

from helper import *
from transformer import TimeSeriesTransformer, TransformerDataset, TransformerVisualLogger, transformer_collate_fn

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd

from tqdm import tqdm
from termcolor import colored
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
    MODEL_NAME = f"trans_{formatted_time}"
VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
create_folder_if_not_exists(WORKING_DIR)

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

# Subprocess
def train_test(dataloader: DataLoader,
               model: TimeSeriesTransformer,
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
line 1 pump speed          float32
line 2 pump speed          float32
PAC pump 1 speed           float32
PAC pump 2 speed           float32
"""

def main() -> None:
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")

    # HYPERPARAMETER
    HYPERPARAMETER = {
        "knowledge_length":     24,     # 4 hours
        "forecast_length":      6,      # 1 hour
        "batch_size":           128,    # 32 is pretty small
    }

    # path = "/".join(INPUT_DATA.split('/')[:-1])
    # name = INPUT_DATA.split('/')[-1].split(".")[0]

    train = load_data(INPUT_DATA, "train")
    val = load_data(INPUT_DATA, "val")
    console_general_data_info(train)
    # Add scaling factors, it for explanation of the model
    scaling_factors = pd.read_csv(
        os.path.join(INPUT_DATA, "scaling_factors.csv"),
        index_col = 0,
        usecols = [1, 2, 3]
        )

    # train = train.head(1000) # HACK Out of memory
    # val = val.head(1000)

    # Split data
    tgt_column = "line 1 pump speed"
    train_src = train.drop(columns=[
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ])
    train_tgt = train[tgt_column]
    val_src = val.drop(columns=[
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ])
    val_tgt = val[tgt_column]
    scaling_factors = scaling_factors.loc[[tgt_column]]

    # Convert data to Tensor object
    train_src = torch.tensor(train_src.values)
    train_tgt = torch.tensor(train_tgt.values).unsqueeze(1)
    val_src = torch.tensor(val_src.values)
    val_tgt = torch.tensor(val_tgt.values).unsqueeze(1)
    ## Check the tensor data type
    print(colored("Check tensor data type", "green"))
    print(f"train_src data type: {train_src.dtype}")
    print(f"train_tgt data type: {train_tgt.dtype}")
    print(f"val_src data type: {val_src.dtype}")
    print(f"val_tgt data type: {val_tgt.dtype}\n")


    # Context based variable
    input_feature_size = train_src.shape[1]
    forecast_feature_size = train_tgt.shape[1]

    train_dataset = TransformerDataset(
        train_src,
        train_tgt,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"]
        )
    val_dataset = TransformerDataset(
        val_src,
        val_tgt,
        HYPERPARAMETER["knowledge_length"],
        HYPERPARAMETER["forecast_length"]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        HYPERPARAMETER["batch_size"],
        drop_last=True,
        collate_fn=transformer_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        HYPERPARAMETER["batch_size"],
        drop_last=True,
        collate_fn=transformer_collate_fn
    )

    # Model
    model = TimeSeriesTransformer(
        input_feature_size,
        forecast_feature_size,
        model_name = MODEL_NAME,
        embedding_dimension = 1024
    ).to(device)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Training
    loss_fn = nn.MSELoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    metrics.append(mae)
    ## Optimizer
    lr = 0.0001  # learning rate
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
    t_loss = TrackerLoss(-1, model)
    # Validation logger
    train_logger = TransformerVisualLogger(
        "train",
        WORKING_DIR,
        meta_data = HYPERPARAMETER,
        runtime_plotting = True,
        which_to_plot = [0,int(HYPERPARAMETER["forecast_length"]/2), HYPERPARAMETER["forecast_length"]-1]
    )
    val_logger = TransformerVisualLogger(
        "val",
        WORKING_DIR,
        meta_data = HYPERPARAMETER,
        runtime_plotting = True,
        which_to_plot = [0,int(HYPERPARAMETER["forecast_length"]/2), HYPERPARAMETER["forecast_length"]-1]
    )
    print(colored("Training:", "black", "on_green"), "\n")
    with tqdm(total=t_epoch.max_epoch, unit="epoch", position=1) as bar:
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
                model.learn(
                    train_loader, 
                    loss_fn, 
                    optimizer, 
                    device,
                    HYPERPARAMETER["forecast_length"],
                    HYPERPARAMETER["knowledge_length"],
                    vis_logger = train_logger,
                )
                train_logger.plot()
                bar.refresh()

                loss = model.val(
                    val_loader, 
                    loss_fn, 
                    device,
                    HYPERPARAMETER["forecast_length"],
                    HYPERPARAMETER["knowledge_length"], 
                    metrics,
                    WORKING_DIR,
                    vis_logger = val_logger,
                    )
                val_logger.plot()
                scheduler_0.step()
                scheduler_1.step()
                scheduler_2.step(loss)

                if not t_loss.check(loss, model):
                    tqdm.write(colored("Loss no longer decrease, finish training", "green", "on_red"))
                    break
                if not t_epoch.check():
                    tqdm.write(colored("Maximum epoch reached. Finish training", "green", "on_red"))
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

    save_model(model, WORKING_DIR)
    visualize_val_loss(t_loss, WORKING_DIR, MODEL_NAME)
    train_logger.save_data()
    train_logger.plot()
    val_logger.save_data()
    val_logger.plot()
    return

if __name__ == "__main__":
    main()

