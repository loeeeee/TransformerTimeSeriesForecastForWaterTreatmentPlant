import settings # Get config
import utils
import inference

import dataset as ds
from helper import console_general_data_info, split_data
from torch_helper import GenericDataFrameDataset, get_best_device, TrackerLoss, TrackerEpoch

import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Tuple
from termcolor import colored
from datetime import datetime

import os
import re
import sys
import math

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
VISUAL_DIR = os.environ["VISUAL_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
MODEL_DIR = os.environ["MODEL_DIR"]

print(f"Read from {INPUT_DATA}")


class PositionalEncoding(nn.Module):
    """
    Copied from pytorch tutorial
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TimeSeriesTransformer(nn.Module):
    """
    Loe learned a lot doing this
    """
    def __init__(self, 
                 input_size: int,
                 forecast_feature_size: int,
                 *args, 
                 embedding_dimension: int               = 512,
                 multi_head_attention_head_size: int    = 8,
                 num_of_encoder_layers: int             = 4,
                 num_of_decoder_layers: int             = 4,
                 **kwargs
                 ) -> None:
        """
        input_size: number of features
        forecast_feature_size: number of features to forecast
        embedding_dimension: the internal dimension of the model
        multi_head_attention_head_size: number of head of the attention layer, 
            applied to all the attention layers
        num_of_encoder_layers: literally
        num_of_decoder_layers: literally
        """
        super().__init__()

        # Input embedding
        self.encoder_input_layer = nn.Linear(
            in_features     = input_size, 
            out_features    = embedding_dimension
        )
        
        # Output embedding
        self.decoder_input_layer = nn.Linear(
            in_features     = forecast_feature_size,
            out_features    = embedding_dimension
        )
        
        # Final forecast output of decoder
        self.final_output = nn.Linear(
            in_features     = embedding_dimension,
            out_features    = forecast_feature_size
        )

        # Positional encoding layer after the encoder input
        self.encoder_positional_encoder_layer = PositionalEncoding(
            d_model         = embedding_dimension
        )

        # Positional encoding layer after the decoder input
        self.decoder_positional_encoder_layer = PositionalEncoding(
            d_model         = embedding_dimension
        )

        # Example encoder layer to pass into nn.TransformerEncoder
        example_encoder = nn.TransformerEncoderLayer(
            d_model         = embedding_dimension,
            nhead           = multi_head_attention_head_size,
        )
        encoder_norm_layer = nn.BatchNorm1d(
            num_features    = embedding_dimension, # TODO
        )

        # Build encoder with the encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer   = example_encoder,
            num_layers      = num_of_encoder_layers,
            norm            = None
        )

        # Example decoder layer to pass into nn.TransformerDecoder
        example_decoder = nn.TransformerDecoderLayer(
            d_model         = embedding_dimension,
            nhead           = multi_head_attention_head_size
        )
        decoder_norm_layer = nn.BatchNorm1d(
            num_features    = embedding_dimension,
        )

        # Build decoder with the decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer   = example_decoder,
            num_layers      = num_of_decoder_layers,
            norm            = None
        )
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor
                ) -> torch.Tensor:
        # Before encoder
        src = self.encoder_input_layer(src)

        # Encoder positional encoding layer
        src = self.encoder_positional_encoder_layer(src)

        # Encoder
        src = self.encoder(src)

        # Before decoder
        tgt = self.decoder_input_layer(tgt)

        # Decoder positional encoding layer
        tgt = self.decoder_positional_encoder_layer(tgt)

        # Decoder
        combined = self.decoder(
            tgt         = tgt,
            memory      = src,
            tgt_mask    = tgt_mask,
            memory_mask = src_mask
        )

        # Final linear layer
        forecast = self.final_output(combined)

        return forecast
    

class TransformerDataset(torch.utils.data.Dataset):
    """
    Hand-crafted dataset for transformer
    knowledge_length: the amount of time series data the transformer model will know to make forecast
    forecast_length: the amount of the time series data the transformer model will forecast given the information
    """
    def __init__(self,
                 src: pd.DataFrame,
                 tgt: pd.DataFrame,
                 knowledge_length: int,
                 forecast_length: int
                 ) -> None:
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.knowledge_length = knowledge_length
        self.forecast_length = forecast_length
        return
    
    def __len__(self) -> int:
        return (self.src.shape[0] - self.forecast_length - self.knowledge_length)
    
    def __getitem__(self, knowledge_start: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Here is what the src would look like:

        src = [xt-4, xt-3, xt-2, xt-1, xt]

        where x denotes the series we're dealing with, e.g. electricity prices.

        The objective is to predict tgt_y which would be:

        tgt_y = [xt+1, xt+2, xt+3]

        So our tgt , which the model needs as input in order to make its forecast for tgt_y , should be:

        tgt = [xt, xt+1, xt+2]
        """
        result_src = self.src[knowledge_start:
                               knowledge_start + self.knowledge_length]
        result_tgt = self.tgt[knowledge_start + self.knowledge_length - 1: 
                              knowledge_start + self.knowledge_length - 1 + self.forecast_length]
        result_tgt_y = self.tgt[knowledge_start + self.knowledge_length: 
                                knowledge_start + self.knowledge_length + self.forecast_length]      
        return result_src, result_tgt, result_tgt_y

"""
class TransformerSampler(torch.utils.data.SequentialSampler):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
"""

# Subprocess

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

def train_test(dataloader: DataLoader,
               model: TimeSeriesTransformer,
               loss_fn: any,
               optimizer: torch.optim,
               device: str,
               forecast_length: int,
               knowledge_length: int) -> None:
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

def train_time_series_transformer(dataloader: DataLoader,
                                  model: TimeSeriesTransformer,
                                  loss_fn: any,
                                  optimizer: torch.optim,
                                  device: str,
                                  forecast_length: int,
                                  knowledge_length: int) -> None:
    # Start training
    model.train()
    
    bar = tqdm(total=len(dataloader), position=0)
    total_loss = 0
    for i, (src, tgt, tgt_y) in enumerate(dataloader):
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=forecast_length
            )
        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=knowledge_length
            )
        # tqdm.write(f"tgt_mask shape: {tgt_mask.shape}\nsrc_mask: {src_mask.shape}\n")

        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        # tqdm.write(f"tgt_y shape: {tgt_y.shape}\nprediction shape: {prediction.shape}\n")
        loss = loss_fn(tgt_y, prediction)
        total_loss += loss
        loss.backward()

        # Take optimizer step
        optimizer.step()

        bar.set_description(desc=f"Instant loss: {loss:.3f}, Continuous loss: {(total_loss/(i+1)):.3f}", refresh=True)
        bar.update()
    bar.close()
    return

def val_time_series_transformer(dataloader: DataLoader,
                                model: nn.Module,
                                loss_fn: any,
                                device: str,
                                forecast_length: int,
                                knowledge_length: int,
                                metrics: list) -> None:
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    # Start evaluation
    model.eval()

    additional_loss = {}
    for additional_monitor in metrics:
        additional_loss[str(type(additional_monitor))] = 0

    with torch.no_grad():
        test_loss = 0
        correct = 0
        bar = tqdm(total=len(dataloader), position=0)
        for i, (src, tgt, tgt_y) in enumerate(dataloader):
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            # tqdm.write(f"{src.shape}, {tgt.shape}, {tgt_y.shape}")
            pred = inference.run_encoder_decoder_inference(
               model, 
               src, 
               forecast_length,
               src.shape[1],
               device
               )
            
            # tqdm.write(f"tgt_y shape: {tgt_y.shape}\nprediction shape: {pred.shape}\n")
            test_loss += loss_fn(pred, tgt_y).item()
            correct += (pred == tgt_y).type(torch.float).sum().item()
            for additional_monitor in metrics:
                additional_loss[str(type(additional_monitor))] += additional_monitor(pred, tgt_y).item()
            bar.update()
            bar.set_description(desc=f"Loss: {(test_loss/(1+i)):.3f}", refresh=True)
        bar.close()
    test_loss /= num_batches
    correct /= size
    tqdm.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")
    for additional_monitor in metrics:
        name = str(type(additional_monitor))[8:-2].split(".")[-1]
        loss = additional_loss[str(type(additional_monitor))] / num_batches
        tqdm.write(f" {name}: {loss:>8f}")
    tqdm.write("\n")
    return test_loss

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

def load_data(path, name) -> pd.DataFrame:
    """
    Load data, return a DataFrame
    """
    regex = f'.*{name}.*\.csv$'  # Regular expression pattern to match the user input in the CSV file name
    pattern = re.compile(regex)

    file_dir = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.match(pattern, file):
                file_dir = os.path.join(root, file)
                file_dir = os.path.abspath(file_dir)
                break

    if file_dir == None:
        raise FileNotFoundError

    print(colored(f"Loading {file_dir}", "green"))
    data = pd.read_csv(file_dir,
                       index_col=0,
                       low_memory=False,
                       parse_dates=["timestamp"]
                    )
    print()
    
    # Make sure all "n/e" values have been removed from df. 
    if utils.is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = utils.to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=["timestamp"], inplace=True)

    return data

def transformer_collate_fn(data):
    """
    src, tgt, tgt_y
    """
    result = [[], [], []]
    for i in range(3):
        result[i] = [torch.Tensor(temp[i]) for temp in data]
        result[i] = torch.stack(result[i])
        result[i] = result[i].permute(1, 0, 2)
    return result[0], result[1], result[2]

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
    # HYPERPARAMETER
    knowledge_length    = 18    # 3 hours
    forecast_length     = 6     # 1 hour
    batch_size          = 128    # 32 is pretty small

    # path = "/".join(INPUT_DATA.split('/')[:-1])
    # name = INPUT_DATA.split('/')[-1].split(".")[0]

    train = load_data(INPUT_DATA, "train")
    val = load_data(INPUT_DATA, "val")
    console_general_data_info(train)

    train = train.head(10000) # HACK Out of memory
    val = val.head(500)

    # Split data
    train_src = train.drop(columns=[
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ])
    train_tgt = train["line 1 pump speed"]
    val_src = val.drop(columns=[
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ])
    val_tgt = val["line 1 pump speed"]

    # Convert data to Tensor object
    train_src = torch.tensor(train_src.values)
    train_tgt = torch.tensor(train_tgt.values).unsqueeze(1)
    val_src = torch.tensor(val_src.values)
    val_tgt = torch.tensor(val_tgt.values).unsqueeze(1)

    # Context based variable
    input_feature_size = train_src.shape[1]
    forecast_feature_size = train_tgt.shape[1]

    train_dataset = TransformerDataset(
        train_src,
        train_tgt,
        knowledge_length,
        forecast_length
        )
    val_dataset = TransformerDataset(
        val_src,
        val_tgt,
        knowledge_length,
        forecast_length
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        drop_last=True,
        collate_fn=transformer_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        drop_last=True,
        collate_fn=transformer_collate_fn
    )

    # Model
    device = get_best_device()
    print(colored(f"Using {device} for training", "black", "on_green"), "\n")
    model = TimeSeriesTransformer(
        input_feature_size,
        forecast_feature_size
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_epoch = TrackerEpoch(500)
    t_loss = TrackerLoss(5, model)
    print(colored("Training:", "black", "on_green"), "\n")
    with tqdm(total=t_epoch.max_epoch, unit="epoch", position=1) as bar:
        while True:
            tqdm.write(colored(f"Epoch {t_epoch.epoch()}", "green"))
            tqdm.write("-------------------------------")
            """
            train_test(
                train_loader, 
                model, 
                loss_fn, 
                optimizer, 
                device,
                forecast_length,
                knowledge_length
                )
            break
            """
            train_time_series_transformer(
                train_loader, 
                model, 
                loss_fn, 
                optimizer, 
                device,
                forecast_length,
                knowledge_length
                )
            loss = val_time_series_transformer(
                val_loader, 
                model, 
                loss_fn, 
                device,
                forecast_length,
                knowledge_length, 
                metrics)
            if not t_loss.check(loss, model):
                tqdm.write(colored("Loss no longer decrease, finish training", "green", "on_red"))
                break
            if not t_epoch.check():
                tqdm.write(colored("Maximum epoch reached. Finish training", "green", "on_red"))
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

