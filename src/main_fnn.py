"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config

from helper import *
from fnn import *

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

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
    MODEL_NAME = f"fnn_{formatted_time}"
VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
create_folder_if_not_exists(WORKING_DIR)

print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

def main() -> None:
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")

    # Load csv
    train = load_data(INPUT_DATA, "train")
    val = load_data(INPUT_DATA, "val")
    console_general_data_info(train)
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
    
    # Dataset
    train_dataset = GenericDataFrameDataset(train_src, train_tgt)
    val_dataset = GenericDataFrameDataset(val_src, val_tgt)

    # Dataloader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

    # Model
    model = FeedForwardNeuralNetwork(train_src.shape[1], MODEL_NAME).to(device)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Training
    loss_fn = nn.MSELoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    metrics.append(mae)
    ## Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer = optimizer,
        T_0 = 5,
    )
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        patience = 2,
    )
    t_epoch = TrackerEpoch(500)
    t_loss = TrackerLoss(7, model)
    print(colored("Training:", "black", "on_green"), "\n")
    with tqdm(total=t_epoch.max_epoch, unit="epoch") as bar:
        while True:
            try:
                lr = scheduler_0.get_last_lr()[0]
                tqdm.write("----------------------------------")
                tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
                tqdm.write(colored(f"Learning rate {lr}", "green"))
                tqdm.write("----------------------------------")
                
                model.learn(
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                )
                
                loss = model.val(
                    val_loader,
                    loss_fn,
                    device,
                    metrics,
                )

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

            if not t_loss.check(loss, model):
                tqdm.write(colored("Loss no longer decrease, finish training", "green", "on_red"))
                break
            if not t_epoch.check():
                tqdm.write(colored("Maximam epoch reached. Finish training", "green", "on_red"))
                break
            bar.update()
        bar.close()

    print(colored("Done!", "black", "on_green"), "\n")
    
    # Bring back the best known model
    model = t_loss.get_best_model()
    print(colored(f"The best model has the validation loss of {t_loss.lowest_loss}", "cyan"))

    # Save the models
    save_model(model, WORKING_DIR)

    # Visualize training process
    visualize_val_loss(t_loss, WORKING_DIR, MODEL_NAME)
    return

if __name__ == "__main__":
    main()