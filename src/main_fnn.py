"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config
import utils # TODO: recode this

from helper import console_general_data_info, create_folder_if_not_exists
from helper import get_best_device, TrackerLoss, TrackerEpoch

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

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.linear_relu_stack(x)
    

def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim, device: str):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            tqdm.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val(dataloader, model, loss_fn, device, metrics: list):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    additional_loss = {}
    for additional_monitor in metrics:
        additional_loss[str(type(additional_monitor))] = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for additional_monitor in metrics:
                additional_loss[str(type(additional_monitor))] += additional_monitor(pred, y).item()
    test_loss /= num_batches
    correct /= size
    tqdm.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")
    for additional_monitor in metrics:
        name = str(type(additional_monitor))[8:-2].split(".")[-1]
        loss = additional_loss[str(type(additional_monitor))] / num_batches
        tqdm.write(f" {name}: {loss:>8f}")
    tqdm.write("\n")
    return test_loss

def main() -> None:
    # Load csv
    data = pd.read_csv(INPUT_DATA,
                       header=[0,1],
                       index_col=0)

    console_general_data_info(data)

    # Split data
    ## Split the DataFrame into features and target variable
    X = data.drop(columns=['f16', 'f17', 'f18', 'f31', 'f32'])
    y = data[('f16', 'pac泵1频率')]
    ## Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    ## Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)
    # X_train.to_csv(os.path.join(PROCESSED_DIR, "temp_X_train.csv"))
    # X_val.to_csv(os.path.join(PROCESSED_DIR, "temp_X_val.csv"))
    # X_test.to_csv(os.path.join(PROCESSED_DIR, "temp_X_test.csv"))

    # Dataset
    train_dataset = GenericDataFrameDataset(X_train, y_train)
    val_dataset = GenericDataFrameDataset(X_val, y_val)
    test_dataset = GenericDataFrameDataset(X_test, y_test)

    # Dataloader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    # Model
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")
    model = FeedForwardNeuralNetwork(X_train.shape[1]).to(device)
    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)

    # Training
    loss_fn = nn.MSELoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    metrics.append(mae)
    ## Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_epoch = TrackerEpoch(500)
    t_loss = TrackerLoss(5, model)
    print(colored("Training:", "black", "on_green"), "\n")
    with tqdm(total=t_epoch.max_epoch, unit="epoch") as bar:
        while True:
            tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
            tqdm.write("-------------------------------")
            train(train_loader, model, loss_fn, optimizer, device)
            loss = val(val_loader, model, loss_fn, device, metrics)
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
    ## Create a sub folder
    root_saving_dir = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(root_saving_dir):
        os.mkdir(root_saving_dir)

    model_saving_dir = os.path.join(root_saving_dir, MODEL_NAME)
    print(f"Save data to {model_saving_dir}")
    torch.save(model.state_dict(), model_saving_dir)
    
    # Save unused test data as csv
    csv_saving_dir = os.path.join(root_saving_dir, f"{MODEL_NAME}_X_test.csv") 
    X_test.to_csv(path_or_buf = csv_saving_dir)
    csv_saving_dir = os.path.join(root_saving_dir, f"{MODEL_NAME}_y_test.csv") 
    y_test.to_csv(path_or_buf = csv_saving_dir)

    # Visualize training process
    loss_history = t_loss.get_loss_history()
    fig_name = f"{MODEL_NAME}_loss_history"
    plt.plot(range(len(loss_history)), loss_history)
    plt.title(fig_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.figtext(0, 0, f"Minimum loss: {t_loss.lowest_loss}", color="#a41095")
    plt.savefig(os.path.join(root_saving_dir, f"{fig_name}.png"), dpi=300)
    return

if __name__ == "__main__":
    main()