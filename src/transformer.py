from __future__ import annotations

import os
import math
import json
import random

import torch
from torch import nn

from tqdm import tqdm
from numpy.typing import NDArray
from termcolor import colored, cprint
from tqdm.utils import _term_move_up
from typing import Tuple, Union, Optional, Any, List
from helper import create_folder_if_not_exists

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GREEN = "#00af34"
BLACK = "#ffffff"

class NotEnoughData(Exception):
    def __init__(self):            
        # Call the base class constructor with the parameters it needs
        super().__init__("Not enough data")


def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor, 
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=False
    ) -> torch.Tensor:

    """
    NB! This function is currently only tested on models that work with 
    batch_first = False
    
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  
    
    This function returns a prediction of length forecast_window for each batch in src
    
    NB! If you want the inference to be done without gradient calculation, 
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()
        
    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be 
    backpropagated during training and gradient calculation hence is required.
    
    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that 
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.
    
    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before 
               passed to this function.
               
        src: The input to the model
        
        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.
                           
        batch_size: batch size
        
        batch_first: If true, the shape of the model input should be 
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]
    
    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of the target variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, 0] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            ).to(device)

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
            ).to(device)

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask) 

        # If statement simply makes sure that the predicted value is 
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step 
            # represented in tgt
            last_predicted_value = prediction[-1, :, :] 

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
    
    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        ).to(device)

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        ).to(device)

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction

def generate_square_subsequent_mask(dim1: int, dim2: int, device: str="cpu") -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    result = torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1).to(device=device, dtype=torch.float32)
    return result
"""

██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗     ██╗███████╗███████╗██████╗ 
██║   ██║██║██╔════╝██║   ██║██╔══██╗██║     ██║╚══███╔╝██╔════╝██╔══██╗
██║   ██║██║███████╗██║   ██║███████║██║     ██║  ███╔╝ █████╗  ██████╔╝
╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══╝  ██╔══██╗
 ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗██║███████╗███████╗██║  ██║
  ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
                                                                        
"""
"""
The following part is modified from uniplot. Thanks a lot! Uniplot
"""

from uniplot.multi_series import MultiSeries
from uniplot.options import Options
import uniplot.layer_assembly as layer_assembly
import uniplot.plot_elements as elements
from uniplot.getch import getch
from uniplot.param_initializer import validate_and_transform_options
from uniplot.axis_labels.extended_talbot_labels import extended_talbot_labels

def plot(ys: Any, xs: Optional[Any] = None, **kwargs) -> list:
    """
    2D scatter dot plot on the terminal.

    Parameters:

    - `ys` are the y coordinates of the points to plot. This parameter is mandatory and
      can either be a list or a list of lists, or the equivalent NumPy array.
    - `xs` are the x coordinates of the points to plot. This parameter is optional and
      can either be a `None` or of the same shape as `ys`.
    - Any additional keyword arguments are passed to the `uniplot.options.Options` class.
    """
    series: MultiSeries = MultiSeries(xs=xs, ys=ys)
    options: Options = validate_and_transform_options(series=series, kwargs=kwargs)
    # Things to plot
    things_to_plot = []

    # Print header
    for line in _generate_header(options):
        things_to_plot.append(line)

    # Main loop for interactive mode. Will only be executed once when not in interactive # mode.
    continue_looping: bool = True
    loop_iteration: int = 0
    while continue_looping:
        # Make sure we stop after first iteration when not in interactive mode
        if not options.interactive:
            continue_looping = False

        (
            x_axis_labels,
            y_axis_labels,
            pixel_character_matrix,
        ) = _generate_body_raw_elements(series, options)

        # Delete plot before we re-draw
        if loop_iteration > 0:
            nr_lines_to_erase = options.height + 4
            if options.legend_labels is not None:
                nr_lines_to_erase += len(options.legend_labels)
            elements.erase_previous_lines(nr_lines_to_erase)

        for line in _generate_body(
            x_axis_labels, y_axis_labels, pixel_character_matrix, options
        ):
            things_to_plot.append(line)
    return things_to_plot

def _generate_header(options: Options) -> List[str]:
    """
    Generates the header of the plot, so everything above the first line of plottable area.
    """
    if options.title is None:
        return []

    return [elements.plot_title(options.title, width=options.width)]


def _generate_body(
    x_axis_labels: str,
    y_axis_labels: List[str],
    pixel_character_matrix: NDArray,
    options: Options,
) -> List[str]:
    """
    Generates the body of the plot.
    """
    lines: List[str] = []

    # Print plot (double resolution)
    lines.append(f"┌{'─'*options.width}┐")
    for i in range(options.height):
        row = pixel_character_matrix[i]
        lines.append(f"│{''.join(row)}│ {y_axis_labels[i]}")
    lines.append(f"└{'─'*options.width}┘")
    lines.append(x_axis_labels)

    # Print legend if labels were specified
    if options.legend_labels is not None:
        lines.append(elements.legend(options.legend_labels, width=options.width))

    return lines


def _generate_body_raw_elements(
    series: MultiSeries, options: Options
) -> Tuple[str, List[str], NDArray]:
    """
    Generates the x-axis labels, y-axis labels, and the pixel character matrix.
    """
    # Prepare y axis labels
    y_axis_label_set = extended_talbot_labels(
        x_min=options.y_min,
        x_max=options.y_max,
        available_space=options.height,
        unit=options.y_unit,
        log=options.y_as_log,
        vertical_direction=True,
    )
    y_axis_labels = [""] * options.height
    if y_axis_label_set is not None:
        y_axis_labels = y_axis_label_set.render()

    # Observe line_length_hard_cap
    if options.line_length_hard_cap is not None:
        options.reset_width()
        # Determine maximum length of y axis label
        max_y_label_length = max([len(l) for l in y_axis_labels])
        # Make sure the total plot does not exceed `line_length_hard_cap`
        if 2 + options.width + 1 + max_y_label_length > options.line_length_hard_cap:
            # Overflow, so we need to reduce width of plot area
            options.width = options.line_length_hard_cap - (2 + 1 + max_y_label_length)
            if options.width < 1:
                raise

    # Prepare x axis labels
    x_axis_label_set = extended_talbot_labels(
        x_min=options.x_min,
        x_max=options.x_max,
        available_space=options.width,
        unit=options.x_unit,
        log=options.x_as_log,
        vertical_direction=False,
    )
    x_axis_labels = ""
    if x_axis_label_set is not None:
        x_axis_labels = x_axis_label_set.render()[0]

    # Prepare graph surface
    pixel_character_matrix = layer_assembly.assemble_scatter_plot(
        xs=series.xs, ys=series.ys, options=options
    )

    return (x_axis_labels, y_axis_labels, pixel_character_matrix)

"""
End of uniplot part
"""

class TransformerLossConsolePlotter:
    def __init__(self, name: str) -> None:
        """
        This class plots the loss trend in console
        This class plots training loss and validation loss at the same time.
        """
        self.name = name
        self._loss = []
        self._dataloader_cnt = 0
        self._x_axis = []
        self._epoch_cnt = 0
        self._x_axis_cnt = 0

        self._temp_loss = []

    def append(self, loss) -> None:
        """
        Append data to be plot in the training loss trend
        """
        self._temp_loss.append(loss)
        return
    
    def do_a_plot(self) -> None:
        """
        Move the plot left a bit.
        """
        # Spacing
        if self._dataloader_cnt == 0:
            tqdm.write("\n"*20)

        # Organize data
        try:
            total_loss = sum(self._temp_loss) / len(self._temp_loss)
        except ZeroDivisionError:
            # Remove previous plot
            for i in range(21):
                tqdm.write(_term_move_up() + "\r" + " "*70 + "\r", end="")
            # Plot something to for the debug purpose
            tqdm.write(colored("ERROR! Divided by zero!", "red", attrs=["blink"]))
            for i in range(20):
                tqdm.write("")
            # For whatever the reason, certain dataloader does not produce loss
        else:
            self._loss.append(total_loss)
            self._temp_loss = []
            self._x_axis.append(self._x_axis_cnt)
            self._x_axis_cnt += 1
            
            # Pop only if elements are added
            if self._epoch_cnt >= 2: # So that we could compare two epoch
                self._loss.pop(0)
                self._x_axis.pop(0)

            # Plot
            to_plot = plot(
                self._loss,
                xs = self._x_axis,
                title = (f"{self.name} trend"),
                lines = True
                )
            # Remove previous plot
            for i in range(21):
                tqdm.write(_term_move_up() + "\r" + " "*70 + "\r", end="")
            for i in to_plot:
                tqdm.write(i)
            # Count
            self._dataloader_cnt += 1
        return
    
    def signal_new_epoch(self) -> None:
        """
        Modify the x-axis, adding 1
        """
        self._epoch_cnt += 1
        self._dataloader_cnt = 0
        return
    
    def save_data(self, dir_overwrite: str="") -> None:
        """
        Save the loss data
        """
        # TODO
        return
    
    def load_data(self) -> None:
        """
        Load the loss data
        """
        # TODO
        return
    

class TransformerTruthAndGuess:
    def __init__(self) -> None:
        """
        This class is mainly for limiting the scope of the data, reducing the complexity of the Transformer Visual Logger.
        This should store one segment of data, or in other words, one dataloader amount of data.
        """
        self._ground_truth = []
        self._forecast_guess = []
        
    def append(self, truth: torch.tensor, guess: torch.tensor) -> None:
        """
        Add one batch worth of data to the object
        The input is expected to be batch first
        """
        # Metadata
        batch_size = truth.shape[0]
        forecast_length = truth.shape[1]
        
        # Not necessary if only use cpu
        truth = truth.detach().clone().cpu().tolist()
        guess = guess.detach().clone().cpu().tolist()

        # Init in batch forecast guess
        if self._forecast_guess == []:
            self._forecast_guess = [[] for i in range(forecast_length)]
        
        # Organize data
        for i in range(batch_size):
            self._ground_truth.append(truth[i][0][0])
            for j in range(forecast_length):
                self._forecast_guess[j].append(guess[i][j][0])

        # Dropping last data
        ## Ground truth is shorter than the forecast_guess
        ## Cut the forecast guess to the length of the ground truth
        ground_truth_length = len(self._ground_truth)
        self._forecast_guess = [i[:ground_truth_length] for i in self._forecast_guess]
        return
    
    def get(self) -> tuple:
        """
        Return the (ground_truth, forecast_guess) data pair when called.
        When no index is passed, default being the last pair
        """
        return self._ground_truth, self._forecast_guess
    
    def __len__(self) -> int:
        """Get the length of the stored data

        Returns:
            int: Length of the ground truth list
        """
        return len(self._ground_truth)


class TransformerForecastPlotter:
    """
    Logging evaluation process for visualization
    If runtime_plotting is set to false, save_data must be called before calling plot.
    """
    def __init__(self, 
                 name: str,
                 working_dir: str, 
                 runtime_plotting: bool = True,
                 which_to_plot: Union[None, list] = None,
                 in_one_figure: bool = False,
                 plot_interval: int = 1,
                 format: str = "png",
                 ) -> None:
        """
        name: Name of the logger, will be used for names for the folder, do not duplicate the name with other loggers
        runtime_plotting: True means plot when a new data is added
        in_one_figure: Plot all the data point in one figure regardless of the dataloader signal.
        plot_interval: define how often the plotting is called.
        TODO: Async plotting
        """
        self.name = name
        self.working_dir = working_dir
        self.runtime_plotting = runtime_plotting
        self.which_to_plot = which_to_plot
        self.isFinished = False
        self.epoch_cnt = -1 
            # Epoch counter is only used when the runtime plotting is set to True
            # Epoch_cnt starts from -1 because when evert a new epoch is called,
            # it will add 1 first, and starts its end-of-epoch work
        self.in_one_figure = in_one_figure
        self.plot_interval = plot_interval
        self.format = format

        # Truth Guess is stored in the transformer prediction and truth object
        self._truth_guess_per_dataloader = [TransformerTruthAndGuess()]
        # After one epoch, the above _truth_guess_per_dataloader will be append into _truth_guess_per_epoch, and then emptied
        self._truth_guess_per_epoch = []

        # Create subfolder to store all the plot
        subdir_name = f"{self.name}_prediction_trend"
        # Update the working directory
        self.working_dir = os.path.join(
            self.working_dir, 
            subdir_name
            )
        create_folder_if_not_exists(self.working_dir)

    def save_data(self, dir_overwrite: str = "") -> None:
        """
        Dump data to the file
        """
        # Create dir
        if dir_overwrite == "":
            dir = self.working_dir
        else:
            dir = dir_overwrite
        dir = os.path.join(dir, self.name)
        create_folder_if_not_exists(dir)

        # Unzipping data
        unzipped_data = [
            [
                dataloader_data.get()
                for dataloader_data in epoch_data
            ] 
            for epoch_data in self._truth_guess_per_epoch
        ]
        
        data_file_path = os.path.join(dir, "truth&guess.json")
        with open(data_file_path, "w", buffering=16384, encoding="utf-8") as f:
            json.dump(unzipped_data, f, indent=2)

        # Set finish flag
        self.isFinished = True
        return
    
    def load_data(self) -> None:
        """
        Load from file
        """
        # Find working dir
        dir = os.path.join(self.working_dir, self.name)

        # Read data
        data_file_path = os.path.join(dir, "truth&guess.json")
        with open(data_file_path, "w", buffering=16384, encoding="utf-8") as f:
            unzipped_data = json.load(f)
        
        # Zipping data into original format
        self._truth_guess_per_epoch = [
            [
                TransformerTruthAndGuess().append(
                    (dataloader_data[0], dataloader_data[1])
                )
                for dataloader_data in epoch_data
            ] 
            for epoch_data in unzipped_data
        ]
        return
    
    def append(self, 
               ground_truth, 
               forecast_guess
               ) -> None:
        """
        Add data pair to the runtime storage
        When running forecast model, the size of ground truth is [forecast length, batch size, 1];
        the size of forecast guess is [forecast length, batch size, 1]
        """
        # This append is a custom append of TransformerTruthAndGuess
        self._truth_guess_per_dataloader[-1].append(ground_truth, forecast_guess)
        return
    
    def signal_new_dataloader(self) -> None:
        """
        Signal a segment for the later plotting,
        It creates a new Transformer Prediction and Truth object
        """
        if not self.in_one_figure:
            self._truth_guess_per_dataloader.append(TransformerTruthAndGuess())
        return
    
    def signal_new_epoch(self, note: str = "") -> None:
        """
        signal_new_epoch should be called at the end of the epoch, no matter if the runtime plot is set to True or False.
        Because signal_new_epoch not only plots, but also reorganize data, and signal a new epoch.
        """
        # Count
        self.epoch_cnt += 1

        # Organize data
        # Select the data with length greater than 0
        self._truth_guess_per_dataloader = [truth_and_guess for truth_and_guess in self._truth_guess_per_dataloader if (len(truth_and_guess) > 0)]
        # Remove the last unused TransformerTruthAndGuess
            
        self._truth_guess_per_epoch.append(self._truth_guess_per_dataloader)
        self._truth_guess_per_dataloader = [TransformerTruthAndGuess()]

        # If not at plot interval, skip the plotting
        if self.epoch_cnt % self.plot_interval != 0 and not self.isFinished:
            # Skip the plotting all together
            return

        # Start plotting
        if not self.isFinished and self.runtime_plotting:
            self._plot_truth_vs_guess_init(
                idx = self.epoch_cnt,
                which_to_plot = self.which_to_plot,
                in_one_figure = self.in_one_figure,
                format = self.format,
                note = note,
            )
        elif self.isFinished and not self.runtime_plotting:
            # Find y_min_max in both ground truth and forecast guess
            global_min = []
            global_max = []
            for epoch_data in self._truth_guess_per_epoch:
                # Find the minimum and maximum of the value
                """
                Data structure of TransformerTruthAndGuess.get()
                (
                []: truth,
                [[]]: guess
                )
                """
                truth_min = min(
                    [(lambda x: min(x.get()[0]))(dataloader_data) 
                     for dataloader_data in epoch_data]
                    )
                guess_min = min(
                    [(lambda x: self._find_minimum_value(x.get()[1]))(dataloader_data) 
                     for dataloader_data in epoch_data]
                    )
                truth_max = max(
                    [(lambda x: max(x.get()[0]))(dataloader_data) 
                     for dataloader_data in epoch_data]
                    )
                guess_max = max(
                    [(lambda x: self._find_maximum_value(x.get()[1]))(dataloader_data) 
                     for dataloader_data in epoch_data]
                    )
                global_min.append(min(truth_min, guess_min))
                global_max.append(max(truth_max, guess_max))

            global_min = min(global_min)
            global_max = max(global_max)

            y_min_max = (global_min, global_max)

            for i in range(len(self._truth_guess_per_epoch)):
                self._plot_truth_vs_guess_init(
                    idx = i,
                    which_to_plot = self.which_to_plot,
                    y_min_max= y_min_max,
                    in_one_figure = self.in_one_figure,
                    format = self.format,
                    note = note,
                )
        return
    
    def signal_finished(self, note: str="") -> None:
        """signal finished should be called after signal epoch when the training finishes
        """
        self.isFinished = True
        self.signal_new_epoch(note=note)
        return
        
    def _plot_truth_vs_guess_init(self,
                                  idx: int = -1,
                                  which_to_plot: Union[None, list] = None,
                                  y_min_max: Union[None, tuple] = None,
                                  in_one_figure: bool = False,
                                  format: str = "png",
                                  note: str = "",
                                ) -> None:
        if not in_one_figure:
            # Create subfolder for each epoch
            # Working dir is now the subfolder
            subdir_name = f"epoch_{idx}"
            working_dir = os.path.join(
                self.working_dir, 
                subdir_name
                )
            create_folder_if_not_exists(working_dir)

            # Tracking progress
            bar = tqdm(
                total       = len(self._truth_guess_per_epoch[idx]), 
                desc        = colored("Plotting", "green", attrs=["blink"]),
                unit        = "frame",
                position    = 1,
                colour      = GREEN,
                )
            # Plotting
            basename = f"{self.name}_prediction_trend"
            for dataloader_truth_and_guess in self._truth_guess_per_epoch[idx]:
                # Get figure sequence
                fig_sequence = self._get_plot_sequence(
                    working_dir, 
                    basename
                    )
                fig_name = f"{basename}_{str(fig_sequence).zfill(3)}"

                # Call the plotting function
                self._plot_truth_vs_guess(
                    fig_name,
                    working_dir,
                    dataloader_truth_and_guess,
                    which_to_plot=which_to_plot,
                    y_min_max=y_min_max,
                    format=format,
                    note=note,
                )  
                bar.update()
            bar.set_description("Finish plotting")
            bar.colour = BLACK
            bar.close()
        else:
            fig_name = f"epoch_{idx}_{self.name}_prediction_trend"
            dataloader_truth_and_guess = self._truth_guess_per_epoch[idx][0]
            # Call the plotting function
            self._plot_truth_vs_guess(
                fig_name,
                self.working_dir,
                dataloader_truth_and_guess,
                which_to_plot=which_to_plot,
                y_min_max=y_min_max,
                format=format,
                note=note,
            )
        return
    
    def _plot_truth_vs_guess(self,
                                 figure_name: str,
                                 working_dir: str,
                                 truth_and_guess: TransformerTruthAndGuess,
                                 which_to_plot: Union[None, list] = None,
                                 y_min_max: Union[None, tuple] = None,
                                 format: str = "png",
                                 note: str = ""
                                 ) -> None:
        """
        which_to_plot: accept a list that contains the forecast sequence user would like to plot,
            Default: all
            Usage example: [0, 3, 5] plot forecast sequence 0, 3, 5.
        """
        # Get data
        ground_truth, forecast_guess = truth_and_guess.get()
        # Calculate MSE for 1-unit forecast
        ground_truth = np.asarray(ground_truth)
        forecast_guess = np.asarray(forecast_guess)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the data
        default_colors = ['#ff7f0e', 
                          '#2ca02c', 
                          '#d62728', 
                          '#9467bd', 
                          '#8c564b', 
                          '#e377c2',
                          '#7f7f7f', 
                          '#bcbd22', 
                          '#17becf',
                          '#1f77b4',
                          ]
        ax.plot(ground_truth, linewidth=1)
        ## Draw the line
        if which_to_plot == None:
            which_to_plot = []
            for i in range(len(forecast_guess)):
                which_to_plot.append(i)
                
        for i, c in zip(which_to_plot, default_colors):
            x_axis = [j+i for j in range(len(forecast_guess[i]))]
            data = forecast_guess[i]
            label = f"{i}-unit forecast line"
            ax.plot(x_axis, data, linewidth=0.7, label=label, color=c, alpha=0.5)
            
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Data')
        ax.set_title('Prediction Trend Plot')
        plt.legend(loc="upper left")
        plt.figtext(0, 0, note, color="#a41095")

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Customize the tick labels
        ax.tick_params(axis='x', rotation=45)

        # Add a background color
        ax.set_facecolor('#F2F2F2')

        # Control the size the fig
        if y_min_max != None:
            ax.set_ylim(y_min_max)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        plt.title(" ".join(figure_name.split("_")))

        # Show the plot
        plt.savefig(
            os.path.join(
                working_dir, 
                f"{figure_name}.{format}"
                ), 
            dpi = 400,
            format = format)
        plt.clf()
        plt.close()
        return
    
    def _get_plot_sequence(self, working_dir: str, fig_name: str) -> int:
        """
        See if there is already a plot there
        """
        max_sequence = -1

        for filename in os.listdir(working_dir):
            base, ext = os.path.splitext(filename)

            # Extract the base name and sequence number from the file name
            parts = base.split('_')
            
            # Extract the sequence number from the last part
            sequence_number = int(parts[-1])
            
            if sequence_number > max_sequence:
                max_sequence = sequence_number

        new_sequence_number = max_sequence + 1

        return new_sequence_number
    
    def _find_minimum_value(self, matrix) -> float:
        """
        Find the minimum value in a 2-D matrix
        """
        # Initialize the minimum value with the first element in the matrix
        min_value = 0x3f3f3f3f

        # Iterate over each row in the matrix
        for row in matrix:
            if len(row) == 1:
                local_minimum = row[0]
            else:
                local_minimum = min(*row)
            if local_minimum < min_value:
                min_value = local_minimum

        return min_value
    
    def _find_maximum_value(self, matrix) -> float:
        """
        Find the maximum value in a 2-D matrix
        """
        # Initialize the minimum value with the first element in the matrix
        max_value = -0x3f3f3f3f

        # Iterate over each row in the matrix
        for row in matrix:
            if len(row) == 1:
                local_maximum = row[0]
            else:
                local_maximum = max(*row)
            if local_maximum > max_value:
                max_value = local_maximum

        return max_value
    
class TransformerClassifierPlotter(TransformerForecastPlotter):
    def __init__(self, 
                 name: str, 
                 working_dir: str, 
                 runtime_plotting: bool = True, 
                 which_to_plot: list | None = None,
                 in_one_figure: bool = False,
                plot_interval: int = 1,
                 ) -> None:
        """
        Because the data structure is different, a new class is needed.
        """
        super().__init__(name, 
                         working_dir, 
                         runtime_plotting, 
                         which_to_plot,
                         in_one_figure,
                         plot_interval,
                         )
    
    def append(self, ground_truth, forecast_guess) -> None:
        """
        When running classification model, the size of ground truth is [forecast length, batch size, one-hot label count];
        the size of forecast guess is [forecast length, batch size, one-hot label count]
        ground truth: [
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0...],
                [0, 0, 0, 1, 0, 0, 0, 0, 0...],
                ...
            ],
            ...
        ]
        """
        ground_truth = torch.argmax(ground_truth, dim=2).unsqueeze(2)
        forecast_guess = torch.argmax(forecast_guess, dim=2).unsqueeze(2)
        return super().append(ground_truth, forecast_guess)
        
    
class TransformerForecasterVisualLogger:
    def __init__(self, 
                name: str,
                working_dir: str, 
                runtime_plotting: bool = True,
                which_to_plot: Union[None, list] = None,
                in_one_figure: bool = False,
                plot_interval: int = 1,
                format: str = "png",
                ) -> None:
        
        self.tfp = TransformerForecastPlotter(
            name,
            working_dir,
            runtime_plotting=runtime_plotting,
            which_to_plot=which_to_plot,
            in_one_figure=in_one_figure,
            plot_interval=plot_interval,
            format=format,
        )
        self.tlcp = TransformerLossConsolePlotter(
            name,
        )

    def signal_new_epoch(self, note: str="") -> None:
        self.tfp.signal_new_epoch(note=note)
        self.tlcp.signal_new_epoch()
        return

    def segment(self) -> None:
        self.tfp.signal_new_dataloader()
        self.tlcp.do_a_plot()
        return
    
    def signal_finished(self, note: str="") -> None:
        self.tfp.signal_finished(note=note)
        return

    def save_data(self, dir_overwrite: str="") -> None:
        self.tfp.save_data(dir_overwrite=dir_overwrite)
        return
    
    def load_data(self) -> None:
        self.tfp.load_data()
        return
    
    def append_to_forecast_plotter(self, ground_truth, forecast_guess) -> None:
        self.tfp.append(ground_truth, forecast_guess)
        return
    
    def append_to_loss_console_plotter(self, loss) -> None:
        self.tlcp.append(loss)
        return

"""

███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
                                            
"""
        
class NotEnoughLayerToAverage(Exception):
    def __init__(self):            
        # Call the base class constructor with the parameters it needs
        super().__init__("Not enough layer to average")


class SequenceLengthDoesNotMatch(Exception):
    def __init__(self):            
        # Call the base class constructor with the parameters it needs
        super().__init__("Sequence length does not match")


class PositionalEncoding(nn.Module):
    """
    Copied from pytorch tutorial
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False, device: str = "cpu"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        # TODO: Implement batch_first
        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        if batch_first:
            pe = torch.zeros(1, max_len, d_model, device=device)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.forward = self._forward_batch_first
        else:
            pe = torch.zeros(max_len, 1, d_model, device=device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.forward = self._forward_not_batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` if batch_first is False
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` if batch_first is True
        """
        raise Exception
    
    def _forward_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, knowledge_length, embedding_dim]`` if batch_first is True
        """
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)
    
    def _forward_not_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[knowledge_length, batch_size, embedding_dim]`` if batch_first is False
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class WaterFormer(nn.Module):
    def __init__(
            self,
            dict_size: int,
            word_spatiotemporal_embedding_size: int,
            *args,
            word_embedding_size: int = 512,
            encoder_layer_cnt: int = 8,
            decoder_layer_cnt: int = 8,
            encoder_layer_head_cnt: int = 8,
            decoder_layer_head_cnt: int = 8,
            average_last_n_decoder_output: int = 4,
            device: str = "cpu",
            model_name: str = "WaterFormer",
            hyperparameter_dict: dict = None,
            **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # Always batch first for easy representation
        isBatchFirst = True
        # Spatiotemporal encoding is not always the same
        """
        [<batch>
            [<input_sequence>
                [<spatiotemporal_encoding>
                    sensor reading: float,
                    positional variable: int,
                    isValid: bool,
                    temporal encoding: float, (This can be more than one)
                ],
                [<spatiotemporal_encoding>
                    ...
                ],
                ...
            ],
            [<input_sequence>
                ...
            ],
            ...
        ]
        """
        # Check input integrity
        if average_last_n_decoder_output > decoder_layer_cnt:
            raise NotEnoughLayerToAverage
        # Take notes of device 
        self.device = device
        # Take notes of model name
        self.name = model_name
        # Take notes of hyperparameters
        self.hyperparameter = hyperparameter_dict
        
        # Input embedding convert layer
        ## Semantic extraction
        self.input_embedding = nn.Linear(
            in_features=word_spatiotemporal_embedding_size,
            out_features=word_embedding_size,
            device=device,
        )

        # Encoder positional encoding
        self.encoder_positional_encoding = PositionalEncoding(
            d_model=word_embedding_size,
            batch_first=True,
            device=device,
        )

        # Decoder positional encoding
        self.decoder_positional_encoding = PositionalEncoding(
            d_model=word_embedding_size,
            batch_first=True,
            device=device,
        )

        # Encoder layers
        encoder_layer_example = nn.TransformerEncoderLayer(
            d_model=word_embedding_size,
            nhead=encoder_layer_head_cnt,
            batch_first=isBatchFirst,
            device=device,
        )
        # Actual encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer_example,
            num_layers=encoder_layer_cnt,
            enable_nested_tensor=False,
        )

        # Decoder layers
        decoder_layer_example = nn.TransformerDecoderLayer(
            d_model=word_embedding_size,
            nhead=decoder_layer_head_cnt,
            batch_first=isBatchFirst,
            device=device,
        )
        self.decoder_1 = nn.TransformerDecoder(
            decoder_layer=decoder_layer_example,
            num_layers= decoder_layer_cnt - average_last_n_decoder_output,
        )

        self.decoder_2 = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=word_embedding_size,
                nhead=decoder_layer_head_cnt,
                batch_first=isBatchFirst,
                device=device,
            ) for i in range(average_last_n_decoder_output)
        ])

        # Output dense layer
        # layer_size_after_flatten = word_embedding_size * dict_size
        self.output_dense_layer = nn.Linear(
            in_features=word_embedding_size,
            out_features=dict_size,
        )

    
    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor, 
            src_mask: Union[torch.Tensor, None] = None, 
            tgt_mask: Union[torch.Tensor, None] = None
            ) -> torch.Tensor:
        """Informer style forward
        
        This model will always use batch first tensor format.

        Args:
            src (torch.tensor): Input of encoder
            tgt (torch.tensor): Input of decoder

        Returns:
            torch.tensor: Output of decoder
        """
        # Embedding converting layer
        ## Passing though the embedding converting layer, tensor shape would change
        ## [batch_size, input_sequence_size, spatiotemporal_encoding_size]
        ## To 
        ## [batch_size, input_sequence_size, word_embedding_size]
        ## Semantic extraction
        context = self.input_embedding(src)
        target = self.input_embedding(tgt)

        # Encoder side
        ## Passing though the positional encoding, tensor shape would not change
        ## [batch_size, input_sequence_size, word_embedding_size]
        context = self.encoder_positional_encoding(context)
        ## Passing though the encoder, tensor shape would not change
        ## [batch_size, input_sequence_size, word_embedding_size]
        context = self.encoder(
            context,
            )

        # Decoder side
        ## Passing though the positional encoding, tensor shape would not change
        ## [batch_size, input_sequence_size, word_embedding_size]
        target = self.decoder_positional_encoding(target)

        ## Passing though the decoder, tensor shape would not change
        ## [batch_size, input_sequence_size, word_embedding_size]
        target = self.decoder_1(
            target,
            context,
            memory_mask = src_mask,
            tgt_mask = tgt_mask,
            )
        
        average_bucket = []
        for i, layer in enumerate(self.decoder_2):
            target = layer(
                target, 
                context,
                memory_mask=src_mask,
                tgt_mask=tgt_mask,
                )
            average_bucket.append(target)
        
        ## Passing though the average layer
        ## stacking result: [average_last_n_decoder_output, batch_size, input_sequence_size, word_embedding_size]
        ## average result: [batch_size, input_sequence_size, word_embedding_size]
        average = torch.stack(average_bucket, dim=0).mean(dim=0)
        
        # Output
        ## Passing though dense layer
        ## Shape is changed to
        ## [batch_size, input_sequence_size, dict_size]
        result = self.output_dense_layer(average)
        return result

    def dump_hyperparameter(self, working_dir: str) -> None:
        cprint("Dumping hyperparameter to json file", "green")
        with open(os.path.join(working_dir, "hyperparameter.json"), mode="w", encoding="utf-8") as f:
            json.dump(self.hyperparameter, f, indent=2)
        return

"""

██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║   
██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║   
██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║   
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   
                                                          
"""  

def transformer_collate_fn(data):
    """
    src, tgt, tgt_y
    structure of data:
        [
            [
                [src
                
                ],
                [tgt
                
                ],
                [tgt_y
                
                ]
            ],
            [
                [src
                
                ],
                [...
                
                ],
                ...
            ],
            ...
        ]
    """
    result = [[], [], []]
    for i in range(3):
        result[i] = [temp[i] for temp in data]
        result[i] = torch.stack(result[i])
    return result[0], result[1], result[2]


class WaterFormerDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            src: np.ndarray,
            tgt: np.ndarray,
            timestamp: Union[list, np.array],
            maximum_knowledge_length: int,
            dict_size: int,
            *args, 
            device: str = "cpu",
            **kwargs
            ) -> None:
        """
        The dataset is inspired by (Long-Range Transformers for Dynamic Spatiotemporal Forecasting)[https://arxiv.org/abs/2109.12218]

        Some blog post also helps the understanding of the dataset. [https://pub.towardsai.net/time-series-regression-using-transformer-models-a-plain-english-introduction-3215892e1cc]

        The dataset formats the data in a way that avoid the limitation of attention layer in a vanilla transformer.
        The attention layer could only focus on one word minimum in the original Attention Is All You Need paper, which 
        is fine for NLP, as it a word, typically is the smallest semantic unit in a language. However, in the context of 
        forecasting continuous values, like pump speed, if we still arrange the values from the same time stamp the 
        same as how we arrange the embeddings of a single word, the attention layer, by nature, cannot find the 
        relationship in between the values of the same timestamp, but from different sensors.

        To solve the problem, the dataset will treat every single measured datum as a semantic unit, instead of treating 
        all the data points in a single measure as one semantic unit. In additional to that, the dataset will also add 
        helpful metadata like encoded timestamp, positional variables (different from positional encoding in the 
        transformer), and whether the data is NaN so that it could also help with the robustness of the model.

        The dataset is expected to receive formatted list of input.

        [
            [sensor_1, sensor_2],

            [sensor_1, sensor_2],
        ]

        The output is in the format of following.

        [
            src, 
            tgt, 
            tgt_y
        ]
        """
        super().__init__(*args, **kwargs)
        self.src_feature_length = src.shape[1] # Step size when doing __getitem__
        cprint(f"Source number of features: {self.src_feature_length}", color="green")
        self.tgt_feature_length = tgt.shape[1]
        cprint(f"Target number of features: {self.tgt_feature_length}", color="green")
        self.knowledge_length = maximum_knowledge_length

        # Calculate sequence length
        self.max_input_sequence_size = maximum_knowledge_length * self.src_feature_length
        cprint(f"input sequence size: {maximum_knowledge_length}", "green")
        self.dict_size = dict_size
        self.device = device
        
        self._length = src.shape[0] - self.knowledge_length - 1
        if self._length <= 0:
            raise NotEnoughData
        else:
            cprint(f"The length of the dataset is {self._length}", "green")
        
        # Starting formatting data
        cprint("Convert timestamp", "cyan")
        self.timestamp = self._convert_timestamp(timestamp)
        cprint("Combine timestamp with source", "cyan")
        self.src = self._forge_timestamp_with_data(src, self.timestamp)
        cprint(f"Source shape: {self.src.shape}", "green")
        cprint("Combine timestamp with target", "cyan")
        self.tgt = self._forge_timestamp_with_data(tgt, self.timestamp)
        cprint(f"Target shape: {self.tgt.shape}", "green")
        self.tgt_y = self.tgt[:, 0].copy().astype(np.int32)
        cprint(f"Target Y shape: {self.tgt_y.shape}", "green")
        cprint("Finish initialize\n", "green")

        # Note the spatiotemporal_encoding_size
        self.spatiotemporal_encoding_size = self.src.shape[1]

    def __len__(self) -> int:
        """Get the length of current dataset

        Returns:
            int: Amount of elements in the dataset
        """
        return int(self._length)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Get the *index* item in the dataset, the dataset is using sliding windows

        Args:
            index (int): The nth datum in the dataset

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: src, tgt, tgt_y in sequence
        """
        index += 1
        src_offset_index = index * self.src_feature_length + self.max_input_sequence_size
        tgt_offset_index = index * self.tgt_feature_length + self.knowledge_length
        src = torch.tensor(self.src[src_offset_index - self.max_input_sequence_size: src_offset_index], device=self.device)
        tgt = torch.tensor(self.tgt[tgt_offset_index - self.knowledge_length: tgt_offset_index], device=self.device)
        tgt_y = torch.tensor(self.tgt_y[tgt_offset_index - self.knowledge_length + 1: tgt_offset_index + 1]).type(torch.LongTensor)
        tgt_y = torch.nn.functional.one_hot(tgt_y, num_classes=self.dict_size).to(dtype=torch.float32, device=self.device)
        return src, tgt, tgt_y
    
    @staticmethod
    def _convert_timestamp(timestamp: Union[list, np.array]):
        """Convert absolute timestamp to sinusoid timestamp

        The default behaviors are convert the input timestamp to sinusoid, cos representation with 
        hour, day, week, month,and year as period. In total it should generate 10 new dimension

        Args:
            timestamp (Union[list, np.array]): The list of timestamp will be convert to np.array

        Returns:
            np.array: The sinusoid representation of the timestamp. The default shape would be 
                        (timestamp.shape[0], 10)
        """
        # Convert to desired input
        if isinstance(timestamp, list):
            timestamp: np.array = np.array(timestamp, type=np.timedelta64)
        
        int_timestamp = timestamp.copy().astype('datetime64[m]').astype(np.int64)
        
        # Hour
        hour_length = 60
        sin_hours = np.sin(2 * np.pi * int_timestamp / hour_length)
        cos_hours = np.cos(2 * np.pi * int_timestamp / hour_length)
        # Day
        day_length = 60 * 24
        sin_days = np.sin(2 * np.pi * int_timestamp / day_length)
        cos_days = np.cos(2 * np.pi * int_timestamp / day_length)

        # Week
        week_length = 60 * 24 * 7
        sin_weeks = np.sin(2 * np.pi * int_timestamp / week_length)
        cos_weeks = np.cos(2 * np.pi * int_timestamp / week_length)

        # Year
        year_length = 60 * 24 * 365
        sin_years = np.sin(2 * np.pi * int_timestamp / year_length)
        cos_years = np.cos(2 * np.pi * int_timestamp / year_length)

        # Month
        sin_months = np.zeros(shape=int_timestamp.shape, dtype=np.float32)
        cos_months = np.zeros(shape=int_timestamp.shape, dtype=np.float32)
        # Extract the month information using NumPy functions
        months = timestamp.astype('datetime64[M]').astype(int) % 12 + 1
        # Get an array of boolean values where True represents months with 31 days
        big_month_length = 60 * 24 * 31
        is_31_days_month = np.isin(months, [1, 3, 5, 7, 8, 10, 12])
        sin_months[is_31_days_month] = np.sin(2 * np.pi * int_timestamp[is_31_days_month] / big_month_length)
        cos_months[is_31_days_month] = np.cos(2 * np.pi * int_timestamp[is_31_days_month] / big_month_length)
        # Get an array of boolean values where True represents months with 30 days
        small_month_length = 60 * 24 * 30
        is_30_days_month = np.isin(months, [4, 6, 9, 11])
        sin_months[is_30_days_month] = np.sin(2 * np.pi * int_timestamp[is_30_days_month] / small_month_length)
        cos_months[is_30_days_month] = np.cos(2 * np.pi * int_timestamp[is_30_days_month] / small_month_length)
        # Get an array of boolean values where True represents months being Feb
        tiny_month_length = 60 * 24 * 28
        is_feb = np.isin(months, [2])
        sin_months[is_feb] = np.sin(2 * np.pi * int_timestamp[is_feb] / tiny_month_length)
        cos_months[is_feb] = np.cos(2 * np.pi * int_timestamp[is_feb] / tiny_month_length)

        cprint("Example timestamp encoding:", "green")
        print(f"Hour:\n{sin_hours[:10]}\n{cos_hours[:10]}")
        print(f"Day:\n{sin_days[:10]}\n{cos_days[:10]}")
        print(f"Week:\n{sin_weeks[:10]}\n{cos_weeks[:10]}")
        print(f"Month:\n{sin_months[:10]}\n{cos_months[:10]}")
        print(f"Year:\n{sin_years[:10]}\n{cos_years[:10]}")

        sinusoid_timestamp = np.stack(
            [sin_hours, cos_hours, sin_days, cos_days, sin_weeks, cos_weeks, sin_months, cos_months, sin_years, cos_years], 
            axis=0,
        )

        # Transpose the new array for easy usage
        ## Change from
        """
        [
            [sin_hour, sin_hour, ...],
            [cos_hour, cos_hour, ...],
            [sin_day, sin_day, ...],
            ...
        ]
        """
        ## Change to
        """
        [
            [sin_hour, cos_hour, sin_day, cos_day, ...],
            [sin_hour, cos_hour, sin_day, cos_day, ...],
            [sin_hour, cos_hour, sin_day, cos_day, ...],
            ...
        ]
        """
    
        sinusoid_timestamp = sinusoid_timestamp.T

        return sinusoid_timestamp
    
    @staticmethod
    def _forge_timestamp_with_data(data: np.array, timestamp: np.array) -> np.array:
        record_cnt: int = data.shape[0]
        feature_cnt: int = data.shape[1]
        # Reshape data
        """
        [
            var_0, var_1, var_2, ..., var_n, var_0, ...
        ]
        """
        data: np.array = data.reshape((-1))

        # Create isValid array, adding 1 dimension
        ## The rational for this is that the model will know the value should not be used.
        nan_filter = np.isnan(data)
        isValid = np.ones(shape=(data.shape[0]))
        isValid[nan_filter] = 0
        data[nan_filter] = -100

        # Create positional variables, adding 2 dimensions
        """
        [
            sin_0, sin_1, sin_2, ..., sin_n, sin_0, ...
        ]
        """
        sin_positional_variable = np.sin(2 * np.pi * np.arange(feature_cnt) / feature_cnt)
        cos_positional_variable = np.cos(2 * np.pi * np.arange(feature_cnt) / feature_cnt)
        sin_positional_variable = np.tile(sin_positional_variable, record_cnt)
        cos_positional_variable = np.tile(cos_positional_variable, record_cnt)

        # Duplicate timestamp, adding no dimension
        """
        [
            [0],
            [0],
            ...,
            [0],
            [1],
            ...,
            ...
        ]
        """
        timestamp = np.repeat(timestamp, feature_cnt, axis=0)

        # Join
        """
        [
            [<data>],
            [<sin>],
            [<cos>]
        ]
        """
        data = np.stack([data, isValid, sin_positional_variable, cos_positional_variable], axis=0)
        """
        [
            [data, sin_positional_variable, cos_positional_variable],
            [data, sin_positional_variable, cos_positional_variable],
            [data, sin_positional_variable, cos_positional_variable],
            ...
        ]
        """
        data = data.T
        """
        [
            [data, sin_positional_variable, cos_positional_variable, sin_hour, cos_hour, sin_day, cos_day, ...],
            [data, sin_positional_variable, cos_positional_variable, sin_hour, cos_hour, sin_day, cos_day, ...],
            [data, sin_positional_variable, cos_positional_variable, sin_hour, cos_hour, sin_day, cos_day, ...],
            ...
        ]
        """
        data = np.concatenate([data, timestamp], axis=1, dtype=np.float32)        
        return data