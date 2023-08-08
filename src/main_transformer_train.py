"""
This transformer code works on the fully normalized input, and multiply the loss by the scaling factors
"""
import settings # Get config

from helper import *
from transformer import WaterFormer, TransformerLossConsolePlotter, WaterFormerDataset, transformer_collate_fn,\
    generate_square_subsequent_mask, GREEN, BLACK, TransformerForecastPlotter

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

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
    formatted_time = current_time.strftime("%y-%m-%d-%H-%M-%s")
    MODEL_NAME = f"trans_for_{formatted_time}"

VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
DEVICE = settings.DEVICE
RAW_DIR = settings.RAW_DIR

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
isResumed = create_folder_if_not_exists(WORKING_DIR)
# Create check point dir
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pt")
if isResumed:
    INPUT_DATA = os.path.join(WORKING_DIR, "data.csv")

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

TGT_COLUMNS = "line 1 pump speed discrete"

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

# Read pump dictionary
def load_pump_dictionary() -> dict:
    dictionary_path = os.path.join(
        DATA_DIR, "pump_dictionary.json"
        )
    with open(dictionary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# HYPERPARAMETER
HYPERPARAMETER = {
    "knowledge_length":             64,    
    "spatiotemporal_encoding_size": None,  # Generated on the fly
    "batch_size":                   64,    # 32 is pretty small
    "train_val_split_ratio":        0.7,
    "scaled_national_standards":    load_scaled_national_standards(),
    "pump_dictionary":              load_pump_dictionary(),
    "src_columns":                  X_COLUMNS,
    "tgt_columns":                  TGT_COLUMNS,
    "tgt_y_columns":                TGT_COLUMNS,
    "random_seed":                  42,
    "encoder_layer_cnt":            4,
    "decoder_layer_cnt":            4,
    "average_last_n_decoder_output":2,
    "word_embedding_size":          256,
    "decoder_layer_head_cnt":       8,
    "encoder_layer_head_cnt":       8,
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

# Main

def main() -> None:
    print(colored(f"Using {DEVICE} for training", "black", "on_green"), "\n")
    
    if isResumed:
        cprint("--------------------------------", "red", "on_green", attrs=["blink"])
        cprint("Resuming existing model training", "red", "on_green", attrs=["blink"])
        cprint("--------------------------------", "red", "on_green", attrs=["blink"])
        print()
        # Read input data from the existing checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH)
        global HYPERPARAMETER
        HYPERPARAMETER = checkpoint["hyperparameter"]
    print(f"Hyperparameter:\n{json.dumps(HYPERPARAMETER, indent=2)}")

    print("Init TensorBoard!")
    writer = SummaryWriter(WORKING_DIR)

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
        tgt = np.array(data[HYPERPARAMETER["tgt_columns"]].values).reshape((-1, 1))

        timestamp = data.reset_index(names="timestamp")["timestamp"].to_numpy(dtype=np.datetime64)
        
        dataset = WaterFormerDataset(
            src,
            tgt,
            timestamp,
            HYPERPARAMETER["knowledge_length"],
            HYPERPARAMETER["pump_dictionary"]["dict_size"],
            device=DEVICE,
        )
        HYPERPARAMETER["spatiotemporal_encoding_size"] = dataset.spatiotemporal_encoding_size

        loader = torch.utils.data.DataLoader(
            dataset,
            HYPERPARAMETER["batch_size"],
            drop_last=False,
            collate_fn=transformer_collate_fn,
        )
        return loader
    
    train_data = pd.concat([data[:int(train_size/2)], data[int(train_size/2)+val_size:]])
    val_data = data[int(train_size/2):int(train_size/2)+val_size]
    train_loader = dataframe_to_loader(train_data)
    val_loader = dataframe_to_loader(val_data)
    
    # Model
    model: WaterFormer = WaterFormer(
        HYPERPARAMETER["pump_dictionary"]["dict_size"],
        HYPERPARAMETER["spatiotemporal_encoding_size"],
        device=DEVICE,
        encoder_layer_cnt               = HYPERPARAMETER["encoder_layer_cnt"],
        decoder_layer_cnt               = HYPERPARAMETER["decoder_layer_cnt"],
        average_last_n_decoder_output   = HYPERPARAMETER["average_last_n_decoder_output"],
        word_embedding_size             = HYPERPARAMETER["word_embedding_size"],
        decoder_layer_head_cnt          = HYPERPARAMETER["decoder_layer_head_cnt"],
        encoder_layer_head_cnt          = HYPERPARAMETER["encoder_layer_head_cnt"],
        hyperparameter_dict = HYPERPARAMETER,
    ).to(DEVICE)

    print(colored("Model structure:", "black", "on_green"), "\n")
    print(model)
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    ## Additional monitoring
    metrics = []
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    metrics.append(mae)
    metrics.append(mse)
    ## Optimizer
    lr = 1e-7  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,0.001, epochs=300, steps_per_epoch=len(train_loader))
    t_epoch = TrackerEpoch(300)
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
    ## Visualizer
    train_console_plot = TransformerLossConsolePlotter("Train")
    eval_console_plot = TransformerLossConsolePlotter("Eval")
    train_plotter = TransformerForecastPlotter("train", WORKING_DIR, plot_interval=5)
    eval_plotter = TransformerForecastPlotter("eval", WORKING_DIR, plot_interval=2)

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
                        model: WaterFormer, 
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
                    for batch_cnt, (src, tgt, raw_tgt_y) in enumerate(loader, start=1):


                        # zero the parameter gradients
                        model.zero_grad(set_to_none=True)
                        # Forward
                        with torch.autocast(device_type=DEVICE):
                            # Make forecasts
                            prediction = model(src, tgt)
                            # Compute and backprop loss
                            loss = loss_fn(prediction, raw_tgt_y)
                            prediction = torch.argmax(prediction, dim=1)
                            _correct = (prediction == raw_tgt_y).sum().item() /\
                                 (HYPERPARAMETER["batch_size"] * HYPERPARAMETER["knowledge_length"])
                            for additional_monitor in metrics:
                                    additional_loss[str(type(additional_monitor))] += \
                                        (additional_monitor(prediction, raw_tgt_y.to(torch.float)).item()) /\
                                             (HYPERPARAMETER["batch_size"] * HYPERPARAMETER["knowledge_length"])

                        # Backward
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                        # Take optimizer step
                        optimizer.step()

                        # CPU
                        batch_loss = loss.item() / HYPERPARAMETER["knowledge_length"]
                        epoch_train_loss += batch_loss
                        correct += _correct
                        epoch_correct += _correct

                        # Plotter
                        train_plotter.append(raw_tgt_y, prediction)
                        if batch_cnt % writer_loop_size == 0:
                            train_plotter.signal_new_dataloader()
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
                        model: WaterFormer, 
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
                        for batch_cnt, (src, tgt, raw_tgt_y) in enumerate(loader, start=1):
                            with torch.autocast(device_type=DEVICE):
                                prediction = model(src, tgt)
                                loss = loss_fn(prediction, raw_tgt_y)
                                prediction = torch.argmax(prediction, dim=1)
                                for additional_monitor in metrics:
                                    additional_loss[str(type(additional_monitor))] += \
                                        additional_monitor(prediction, raw_tgt_y.to(torch.float)).item() /\
                                    (HYPERPARAMETER["batch_size"] * HYPERPARAMETER["knowledge_length"])
                                _correct = (prediction == raw_tgt_y).sum().item() /\
                                    (HYPERPARAMETER["batch_size"] * HYPERPARAMETER["knowledge_length"])

                            # CPU part
                            batch_loss = loss.item() / HYPERPARAMETER["knowledge_length"]
                            epoch_val_loss += batch_loss
                            correct += _correct 
                            epoch_correct += _correct 

                            # Tensorboard
                            eval_plotter.append(raw_tgt_y, prediction)
                            if batch_cnt % writer_loop_size == 0:
                                eval_plotter.signal_new_dataloader()
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
                        hyperparameter=HYPERPARAMETER,
                    )

                #-------------#
                #--Finish Up--#
                #-------------#
                train_console_plot.signal_new_epoch()
                eval_console_plot.signal_new_epoch()
                train_plotter.signal_new_epoch()
                eval_plotter.signal_new_epoch()
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
    train_plotter.signal_finished()
    eval_plotter.signal_finished()

    writer.close()

    # Save current training state to checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        hyperparameter=HYPERPARAMETER,
    )
    
    print(colored("Done!", "black", "on_green"), "\n")
    
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

    return

if __name__ == "__main__":
    main()

