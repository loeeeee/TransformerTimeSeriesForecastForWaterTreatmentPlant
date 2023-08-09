import os
import torch
import shutil
import settings

import pandas as pd
import numpy as np

from torch import nn
from helper import *
from tqdm import tqdm
from datetime import datetime
from termcolor import cprint, colored
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformer import TransformerLossConsolePlotter
from fnn import FeedForwardNeuralNetwork, FeedForwardNeuralNetworkDataset

# Get the current timestamp
current_time = datetime.now()
# Format the timestamp as YY-MM-DD-HH-MM
formatted_time = current_time.strftime("%y-%m-%d-%H-%M")
MODEL_NAME = f"fnn_{formatted_time}"
VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
DEVICE = settings.DEVICE
# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
isResumed = create_folder_if_not_exists(WORKING_DIR)
# Create check point dir
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pt")
if isResumed:
    INPUT_DATA = os.path.join(WORKING_DIR, "data.csv")
GREEN = "#00af34"
BLACK = "#ffffff"


DATA_PATH = os.path.join(DATA_DIR, "processed.csv")
DATA_SPLIT_RATIO = 0.7
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

TGT_COLUMNS = "line 1 pump speed discrete"
BATCH_SIZE = 1024
isResumed = False # HACK

HYPERPARAMETER = {
    "batch_size": BATCH_SIZE,
    "x_columns": X_COLUMNS,
    "tgt_columns": TGT_COLUMNS,
    "data_split_ratio": DATA_SPLIT_RATIO,
    "dict_size": 100,
}


print(colored(f"Save all files to {WORKING_DIR}", "black", "on_green"))
print()

def save_checkpoint(
        model: nn.Module,
        optimizer,
        hyperparameter: dict,
        ) -> None:
    # Save current training state to checkpoint
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameter': hyperparameter,
            }, CHECKPOINT_PATH)
    return None

def load_data(data_path) -> pd.DataFrame:
    data = pd.read_csv(
        data_path,
        low_memory=False,
        index_col=0,
        parse_dates=["timestamp"],
    )
    data = data.dropna()
    data = to_numeric_and_downcast_data(data)
    train_size = int(data.shape[0] * DATA_SPLIT_RATIO)
    val_size = data.shape[0] - train_size
    train_data = pd.concat([data[:int(train_size/2)], data[int(train_size/2)+val_size:]])
    val_data = data[int(train_size/2):int(train_size/2)+val_size]
    return train_data, val_data

def remove_nan(X: np.array, y: np.array) -> np.array:
    """
    nan_filter = np.isnan(X)
    X = X[nan_filter]
    y = y[nan_filter]
    nan_filter = np.isnan(y)
    X = X[nan_filter]
    y = y[nan_filter]
    """
    return X, y

def dataframe_to_loader(dataframe_X: pd.DataFrame, dataframe_y: pd.DataFrame, batch_size: int = 256) -> DataLoader:
    # Dataset
    np_X, np_y = remove_nan(dataframe_X.to_numpy(), dataframe_y.to_numpy())
    tensor_X = torch.tensor(np_X, device=DEVICE)
    tensor_y = torch.tensor(np_y, device=DEVICE)
    dataset = FeedForwardNeuralNetworkDataset(tensor_X, tensor_y, device=DEVICE)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    return loader

def main():
    train_data, val_data = load_data(DATA_PATH)
    train_loader = dataframe_to_loader(train_data[X_COLUMNS], train_data[TGT_COLUMNS])
    val_loader = dataframe_to_loader(val_data[X_COLUMNS], val_data[TGT_COLUMNS])
    train_data, val_data = load_data(DATA_PATH)
    model = FeedForwardNeuralNetwork(
        d_in=len(X_COLUMNS),
        d_out=HYPERPARAMETER["dict_size"],
        n_depth=20,
        device=DEVICE,
        hyperparameter_dict=HYPERPARAMETER
    )
    print(model)
    print("Init TensorBoard!")
    writer = SummaryWriter(WORKING_DIR)
    if isResumed:
        cprint("--------------------------------", "red", "on_green", attrs=["blink"])
        cprint("Resuming existing model training", "red", "on_green", attrs=["blink"])
        cprint("--------------------------------", "red", "on_green", attrs=["blink"])
        print()
        # Read input data from the existing checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    metrics.append(mae)
    metrics.append(mse)
    ## Optimizer
    lr = 1e-5  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,0.001, epochs=100, steps_per_epoch=len(train_loader))
    t_epoch = TrackerEpoch(100)
    t_val_loss = TrackerLoss(-1, model)
    t_train_loss = TrackerLoss(-1, model)
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
    train_console_plot = TransformerLossConsolePlotter("Train")
    eval_console_plot = TransformerLossConsolePlotter("Eval")
    # Load checkpoint
    if isResumed:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    with tqdm(total=t_epoch.max_epoch, unit="epoch", position=0) as epoch_bar:
        while True:
            try:
                lr = scheduler.get_last_lr()[0]
                tqdm.write(colored("--------------------------------------------", "cyan", attrs=["bold"]))
                tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
                tqdm.write(colored(f"Learning rate {lr:.8f}", "green"))
                tqdm.write(colored(f"Recent training loss trend: {t_train_loss.get_trend(3):.5f}", "green"))
                tqdm.write(colored(f"Recent validation loss trend: {t_val_loss.get_trend(3):.5f}", "green"))
                tqdm.write(colored(f"Best training loss: {t_train_loss.lowest_loss:.5f}", "green"))
                tqdm.write(colored(f"Best validation loss: {t_val_loss.lowest_loss:.5f}", "green"))
                tqdm.write(colored("--------------------------------------------", "cyan", attrs=["bold"]))

                def learn(
                        model: FeedForwardNeuralNetwork, 
                        optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler, 
                        loss_fn: torch.nn.CrossEntropyLoss, 
                        loader, 
                        metrics: list,
                        epoch_cnt: int,
                        ) -> float:
                    #---------#
                    #--Train--#
                    #---------#
                    # Model enters training mode
                    model.train()

                    # Metadata
                    total_batch = len(loader)
                    writer_loop_size = int(total_batch/20)
                    bar = tqdm(
                        total       = total_batch, 
                        position    = 1,
                        colour      = GREEN,
                        )
                    correct = 0
                    epoch_correct = 0
                    epoch_train_loss = 0

                    additional_loss = {}
                    for additional_monitor in metrics:
                        additional_loss[str(type(additional_monitor))] = 0

                    # Iterate through dataloaders
                    for batch_cnt, (src, tgt) in enumerate(loader, start=1):
                        # zero the parameter gradients
                        model.zero_grad(set_to_none=True)
                        # Forward
                        #tgt = tgt.unsqueeze(1)
                        with torch.autocast(device_type=DEVICE):
                            # Make forecasts
                            prediction = model(src)
                            # Compute and backprop loss
                            #print(prediction.size(), tgt.size())
                            loss = loss_fn(prediction, tgt)
                            prediction = torch.argmax(prediction, dim=1)
                            _correct = (prediction == tgt).sum().item() / BATCH_SIZE
                            for additional_monitor in metrics:
                                    additional_loss[str(type(additional_monitor))] += \
                                        (additional_monitor(prediction, tgt.to(torch.float)).item()) / BATCH_SIZE
                        # Backward
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                        # Take optimizer step
                        optimizer.step()

                        # CPU
                        batch_loss = loss.item()
                        epoch_train_loss += batch_loss
                        correct += _correct
                        epoch_correct += _correct

                        # Plotter
                        if batch_cnt % writer_loop_size == 0:
                            normalized_correct = correct / writer_loop_size * 100
                            correct = 0
                            train_console_plot.append(normalized_correct)
                            train_console_plot.do_a_plot()

                        scheduler.step()
                        bar.set_description(desc=f"Instant loss: {batch_loss:.3f} LR: {scheduler.get_last_lr()[0]:.8f}", refresh=True)
                        bar.update()
                    bar.colour = BLACK
                    bar.close()
                    
                    # Report 1
                    epoch_correct /= total_batch
                    epoch_train_loss /= total_batch
                    tqdm.write(f"Train Error: \n Accuracy: {(100*epoch_correct):>0.1f}%, Avg loss: {epoch_train_loss:>8f} ")
                    writer.add_scalar("Train-loss", epoch_train_loss, global_step=epoch_cnt)
                    writer.add_scalar("Train-correct", epoch_correct * 100, global_step=epoch_cnt)
                    for additional_monitor in metrics:
                        name = str(type(additional_monitor))[8:-2].split(".")[-1]
                        loss = additional_loss[str(type(additional_monitor))] / total_batch
                        tqdm.write(f" {name}: {loss:>8f}")
                        writer.add_scalar(f"Train-{name}", loss, global_step=epoch_cnt)
                    return epoch_train_loss
                
                train_loss = learn(model, optimizer, scheduler, loss_fn, train_loader, metrics, t_epoch.epoch())

                def eval(
                        model: FeedForwardNeuralNetwork, 
                        loss_fn: torch.nn.CrossEntropyLoss, 
                        loader, 
                        metrics: list,
                        epoch_cnt: int,
                        ) -> float:
                    #--------------#
                    #--Evaluation--#
                    #--------------#
                    # Metadata 
                    total_batch = len(loader)
                    writer_loop_size = int(total_batch/5)

                    # Start evaluation
                    model.eval()

                    additional_loss = {}
                    for additional_monitor in metrics:
                        additional_loss[str(type(additional_monitor))] = 0

                    # Validation
                    correct = 0
                    epoch_correct = 0
                    epoch_val_loss = 0
                    bar = tqdm(
                        total       = total_batch, 
                        position    = 1,
                        colour      = GREEN,
                        )
                    with torch.no_grad():
                        for batch_cnt, (src, tgt) in enumerate(loader, start=1):
                            with torch.autocast(device_type=DEVICE):
                                prediction = model(src)
                                #print(prediction.size(), tgt.size())
                                loss = loss_fn(prediction, tgt)
                                prediction = torch.argmax(prediction, dim=1)
                                for additional_monitor in metrics:
                                    additional_loss[str(type(additional_monitor))] += \
                                        additional_monitor(prediction, tgt.to(torch.float)).item() / BATCH_SIZE
                                _correct = (prediction == tgt).sum().item() / BATCH_SIZE
                                    
                            # CPU part
                            batch_loss = loss.item()
                            epoch_val_loss += batch_loss
                            correct += _correct 
                            epoch_correct += _correct 

                            # Tensorboard
                            if batch_cnt % writer_loop_size == 0:
                                normalized_correct = correct / writer_loop_size * 100
                                eval_console_plot.append(normalized_correct)
                                eval_console_plot.do_a_plot()
                                correct = 0

                            bar.update()
                            bar.set_description(desc=f"Loss: {batch_loss:.3f}", refresh=True)
                    bar.colour = BLACK
                    bar.close()
                    epoch_correct /= total_batch
                    epoch_val_loss /= total_batch

                    # Report 2
                    tqdm.write(f"Test Error: \n Accuracy: {(100*epoch_correct):>0.1f}%, Avg loss: {epoch_val_loss:>8f} ")
                    writer.add_scalar("Eval-loss", epoch_val_loss, global_step=epoch_cnt)
                    writer.add_scalar("Eval-correct", epoch_correct * 100, global_step=epoch_cnt)
                    for additional_monitor in metrics:
                        name = str(type(additional_monitor))[8:-2].split(".")[-1]
                        loss = additional_loss[str(type(additional_monitor))] / total_batch
                        tqdm.write(f" {name}: {loss:>8f}")
                        writer.add_scalar(f"Eval-{name}", loss, global_step=epoch_cnt)

                    return epoch_val_loss
                
                val_loss = eval(model, loss_fn, val_loader, metrics, t_epoch.epoch())

                # Checkpoint
                if t_epoch.epoch() % 10 == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        hyperparameter=HYPERPARAMETER
                    )

                #-------------#
                #--Finish Up--#
                #-------------#
                train_console_plot.signal_new_epoch()
                eval_console_plot.signal_new_epoch()
                if not t_val_loss.check(val_loss, model):
                    tqdm.write(colored("Validation loss no longer decrease, finish training", "green", "on_red"))
                    break
                if not t_epoch.check():
                    tqdm.write(colored("Maximum epoch reached. Finish training", "green", "on_red"))
                    break
                if not t_train_loss.check(train_loss, model):
                    tqdm.write(colored("Training loss no longer decrease, finish training", "green", "on_red"))
                    break
                epoch_bar.update()
            except KeyboardInterrupt:
                tqdm.write(colored("Early stop triggered by Keyboard Input", "green", "on_red"))
                break
        epoch_bar.close()
    #profiler.stop()

    writer.close()
    # Bring back the best known model
    model = t_val_loss.get_best_model()
    print(colored(f"The best model has the validation loss of {t_val_loss.lowest_loss}", "cyan"))
    model_best_train = t_train_loss.get_best_model()
    model_best_train.name += "_best_trained"
    cprint(f"Best trained model has an train loss of {t_train_loss.lowest_loss}", "cyan")

    if not isResumed:
        # If it is the first time the model is running

        # Dump hyper parameters
        model.dump_hyperparameter(WORKING_DIR)

        # Save data
        shutil.copyfile(
            INPUT_DATA, 
            os.path.join(WORKING_DIR, "data.csv")
            )
        
        # Save train data
        train_data.to_csv(
            os.path.join(WORKING_DIR, "train.csv"),
        )

        # Save evaluation data
        val_data.to_csv(
            os.path.join(WORKING_DIR, "val.csv"),
        )


    # Save model
    save_model(model, WORKING_DIR, train_loader, DEVICE)
    save_model(model_best_train, WORKING_DIR, train_loader, DEVICE)
    
    # Visualize loss
    visualize_loss(t_val_loss, WORKING_DIR, f"{MODEL_NAME}_val")
    visualize_loss(t_train_loss, WORKING_DIR, f"{MODEL_NAME}_train")

main()