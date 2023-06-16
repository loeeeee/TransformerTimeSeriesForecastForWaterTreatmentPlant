import os
import re
import math
import utils
import inference

import torch
from torch import nn

from tqdm import tqdm
from typing import Tuple, Union
from helper import planned_obsolete, create_folder_if_not_exists

import pandas as pd
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex

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
                 model_name: str                        = "time_series_transformer",
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
        
        self.model_name = model_name

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
    
    def learn(self,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: any,
              optimizer: torch.optim,
              device: str,
              forecast_length: int,
              knowledge_length: int) -> None:
        # Start training
        self.train()
        # self, optimizer = ipex.optimize(self, optimizer=optimizer, dtype=torch.float32)

        bar = tqdm(total=len(dataloader), position=0)
        total_loss = 0
        # Generate masks
        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=knowledge_length
            )
        # for idx, row in enumerate(src_mask):
        #     tqdm.write(f"src_mask line {str(idx).zfill(2)}: {row}")
        
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=forecast_length
            )
        # for idx, row in enumerate(tgt_mask):
        #     tqdm.write(f"tgt_mask line {str(idx).zfill(2)}: {row}")
            
        # tqdm.write(f"tgt_mask shape: {tgt_mask.shape}\nsrc_mask: {src_mask.shape}\n")
        for i, (src, tgt, tgt_y) in enumerate(dataloader):
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Make forecasts
            prediction = self(src, tgt, src_mask, tgt_mask)

            # Compute and backprop loss
            # tqdm.write(f"tgt_y shape: {tgt_y.shape}\nprediction shape: {prediction.shape}\n")
            loss = loss_fn(tgt_y, prediction)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            # Take optimizer step
            optimizer.step()

            bar.set_description(desc=f"Instant loss: {loss:.3f}, Continuous loss: {(total_loss/(i+1)):.3f}", refresh=True)
            bar.update()
            # planned_obsolete(5)
        bar.close()
        return
    
    def val(self,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: any,
            device: str,
            forecast_length: int,
            knowledge_length: int,
            metrics: list,
            visualize_dir: str,
            which_to_plot: Union[None, list] = None
            ) -> None:
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        visualizer = TransformerVisualizer(forecast_length, visualize_dir, self.model_name)
        # Start evaluation
        self.eval()
        # self = ipex.optimize(self, dtype=torch.float32)

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
                   self, 
                   src, 
                   forecast_length,
                   src.shape[1],
                   device
                   )

                visualizer.add_data(
                    ground_truth_series = tgt_y,
                    forecast_series     = pred,
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
        visualizer.plot_forecast_vs_ground_truth(which_to_plot)
        for additional_monitor in metrics:
            name = str(type(additional_monitor))[8:-2].split(".")[-1]
            loss = additional_loss[str(type(additional_monitor))] / num_batches
            tqdm.write(f" {name}: {loss:>8f}")
        tqdm.write("\n")
        return test_loss
    

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

class TransformerVisualizer:
    def __init__(
            self, 
            forecast_length: int,
            save_dir: str,
            model_name: str
            ) -> None:
        """
        Only works for single feature prediction
        """
        self.forecast_length    = forecast_length
        self.save_dir           = save_dir
        self.model_name         = model_name
        # Runtime variable
        self.all_ground_truth   = []
        self.all_forecast_guess = [[] for i in range(self.forecast_length)]

    def add_data(
            self, 
            ground_truth_series: torch.Tensor, 
            forecast_series: torch.Tensor
            ) -> None:
        """
        For the simplicity of coding, last data will be dropped
        """
        batch_size = ground_truth_series.shape[1]
        ground_truth_series_np = ground_truth_series.numpy()
        forecast_series_np = forecast_series.numpy()
        for i in range(batch_size):
            self.all_ground_truth.append(ground_truth_series_np[0, i, 0])
            for j in range(self.forecast_length):
                self.all_forecast_guess[j].append(forecast_series_np[j, i, 0])

        # Dropping last data
        ## Ground truth is shorter than the forecast_guess
        ## Cut the forecast guess to the length of the ground truth
        ground_truth_length = len(self.all_ground_truth)
        self.all_forecast_guess = [i[:ground_truth_length] for i in self.all_forecast_guess]
        return
    
    def plot_forecast_vs_ground_truth(self, which_to_plot: Union[None, list] = None) -> None:
        """
        which_to_plot: accept a list that contains the forecast sequence user would like to plot,
            Default: all
            Usage example: [0, 3, 5] plot forecast sequence 0, 3, 5.
        """
        fig_name = f"{self.model_name}_prediction_trend"
        # Create subfolder
        working_dir = os.path.join(self.save_dir, fig_name)
        create_folder_if_not_exists(working_dir)
        # Get figure sequence
        fig_sequence = self._get_plot_sequence(working_dir, fig_name)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the data
        default_colors = ['#1f77b4', 
                          '#ff7f0e', 
                          '#2ca02c', 
                          '#d62728', 
                          '#9467bd', 
                          '#8c564b', 
                          '#e377c2',
                          '#7f7f7f', 
                          '#bcbd22', 
                          '#17becf'
                          ]
        x_axis = range(len(self.all_ground_truth))
        ax.plot(x_axis, self.all_ground_truth, linewidth=2)
        for i, c in zip(range(self.forecast_length), default_colors):
            if which_to_plot == None:
                pass
            elif i not in which_to_plot:
                continue
            data = self.all_forecast_guess[i]
            label = f"{i}-unit forecast line"
            ax.plot(data, linewidth=1, label=label, color=c, alpha=0.5)
            
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Data')
        ax.set_title('Prediction Trend Plot')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Customize the tick labels
        ax.tick_params(axis='x', rotation=45)

        # Add a background color
        ax.set_facecolor('#F2F2F2')

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        plt.title(" ".join(fig_name.split("_")))

        # Show the plot
        plt.savefig(
            os.path.join(
                working_dir, 
                f"{fig_name}_{fig_sequence}.png"
                ), 
            dpi = 400)
        plt.clf()
        return
    
    def _get_plot_sequence(self, working_dir: str, fig_name: str) -> int:
        """
        See if there is already a plot there
        """
        max_sequence = -1

        for filename in os.listdir(working_dir):
            if filename.endswith('.png'):
                # Extract the base name and sequence number from the file name
                base, ext = os.path.splitext(filename)
                parts = base.split('_')

                if (len(parts) > 1 and '_'.join(parts[:-1]) == fig_name) or len(parts) == 1:
                    # Extract the sequence number from the last part
                    sequence_number = int(parts[-1])

                    if sequence_number > max_sequence:
                        max_sequence = sequence_number

        new_sequence_number = max_sequence + 1

        return new_sequence_number