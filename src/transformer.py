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

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
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
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
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
    
    def signal_new_dataloader(self) -> None:
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
            if self._epoch_cnt >= 1:
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
        """
        # Metadata
        batch_size = truth.shape[1]
        forecast_length = truth.shape[0]
        
        # Not necessary if only use cpu
        truth = truth.detach().clone().cpu().tolist()
        guess = guess.detach().clone().cpu().tolist()

        # Init in batch forecast guess
        if self._forecast_guess == []:
            self._forecast_guess = [[] for i in range(forecast_length)]
        
        # Organize data
        for i in range(batch_size):
            self._ground_truth.append(truth[0][i][0])
            for j in range(forecast_length):
                self._forecast_guess[j].append(guess[j][i][0])

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
            json.dump(unzipped_data, f)

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
    
    def signal_new_epoch(self) -> None:
        """
        signal_new_epoch should be called at the end of the epoch, no matter if the runtime plot is set to True or False.
        Because signal_new_epoch not only plots, but also reorganize data, and signal a new epoch.
        """
        # Count
        self.epoch_cnt += 1

        # Organize data
        self._truth_guess_per_dataloader = [truth_and_guess for truth_and_guess in self._truth_guess_per_dataloader if (len(truth_and_guess.get()[0]) > 0)]
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
                )
        return
        
    def _plot_truth_vs_guess_init(self,
                                  idx: int = -1,
                                  which_to_plot: Union[None, list] = None,
                                  y_min_max: Union[None, tuple] = None,
                                  in_one_figure: bool = False,
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
            )
        return
    
    def _plot_truth_vs_guess(self,
                                 figure_name: str,
                                 working_dir: str,
                                 truth_and_guess: TransformerTruthAndGuess,
                                 which_to_plot: Union[None, list] = None,
                                 y_min_max: Union[None, tuple] = None,
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
        mse = (np.square(ground_truth - forecast_guess[0])).mean(axis=0)

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
        ax.plot(ground_truth, linewidth=1)
        ## Draw the line
        for i, c in zip(which_to_plot, default_colors):
            x_axis = [j+i+1 for j in range(len(forecast_guess[i]))]
            data = forecast_guess[i]
            label = f"{i}-unit forecast line"
            ax.plot(x_axis, data, linewidth=0.7, label=label, color=c, alpha=0.5)
            
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Data')
        ax.set_title('Prediction Trend Plot')
        plt.legend(loc="upper left")
        plt.figtext(0, 0, f"MSE for 1 unit forecast: {mse}", color="#a41095")

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
                f"{figure_name}.png"
                ), 
            dpi = 400,
            format = "png")
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
                ) -> None:
        
        self.tfp = TransformerForecastPlotter(
            name,
            working_dir,
            runtime_plotting=runtime_plotting,
            which_to_plot=which_to_plot,
            in_one_figure=in_one_figure,
            plot_interval=plot_interval,
        )
        self.tlcp = TransformerLossConsolePlotter(
            name,
        )

    def signal_new_epoch(self) -> None:
        self.tfp.signal_new_epoch()
        self.tlcp.signal_new_epoch()
        return

    def signal_new_dataloader(self) -> None:
        self.tfp.signal_new_dataloader()
        self.tlcp.signal_new_dataloader()
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


class TransformerClassifierVisualLogger(TransformerForecasterVisualLogger):
    def __init__(self, 
                 name: str, 
                 working_dir: str, 
                 meta_data: dict = {}, 
                 runtime_plotting: bool = True, 
                 which_to_plot: list | None = None,
                 in_one_figure: bool = False,
                 plot_interval: int = 1,
                 ) -> None:
        super().__init__(name, 
                         working_dir, 
                         meta_data, 
                         runtime_plotting, 
                         which_to_plot,
                         in_one_figure,
                         plot_interval,
                         )
        self.tfp = TransformerClassifierPlotter(
            name,
            working_dir,
            meta_data=meta_data,
            runtime_plotting=runtime_plotting,
            which_to_plot=which_to_plot,
            in_one_figure=in_one_figure,
            plot_interval=plot_interval,
        )


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
    
"""

███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
                                            
"""
        
class TimeSeriesTransformer(nn.Module):
    """
    Loe learned a lot doing this
    """
    def __init__(self, 
                 input_size: int,
                 knowledge_length: int,
                 forecast_length: int,
                 device: str,
                 metadata: dict,
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
        metadata: store something not used in the model, but essential to evaluation
        forecast_feature_size: number of features to forecast
        embedding_dimension: the internal dimension of the model
        multi_head_attention_head_size: number of head of the attention layer, 
            applied to all the attention layers
        num_of_encoder_layers: literally
        num_of_decoder_layers: literally
        """
        super().__init__()
        # Store all the args and kwargs for restoring
        self.args = [
            input_size,
            knowledge_length,
            forecast_length,
            device,
            metadata,
        ]
        self.kwargs = {
            "embedding_dimension":              embedding_dimension,
            "multi_head_attention_head_size":   multi_head_attention_head_size,
            "num_of_encoder_layers":            num_of_encoder_layers,
            "num_of_decoder_layers":            num_of_decoder_layers,
            "model_name":                       model_name,
        }
        
        self.model_name = model_name
        self.device = device
        # Generate masks
        self.src_mask = generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=knowledge_length
            ).to(device)
        
        self.tgt_mask = generate_square_subsequent_mask(
            dim1=forecast_length,
            dim2=forecast_length
            ).to(device)

        # Force the model only have one output feature
        _forecast_feature_size = 1

        # Input embedding
        self.encoder_input_layer = nn.Linear(
            in_features     = input_size, 
            out_features    = embedding_dimension
        )
        
        # Output embedding
        self.decoder_input_layer = nn.Linear(
            in_features     = _forecast_feature_size,
            out_features    = embedding_dimension
        )
        
        # Final forecast output of decoder
        self.final_output = nn.Linear(
            in_features     = embedding_dimension,
            out_features    = _forecast_feature_size
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
            tgt_mask    = self.tgt_mask,
            memory_mask = self.src_mask
        )

        # Final linear layer
        forecast = self.final_output(combined)

        return forecast
        
    def dump_hyper_parameters(self, dir: str) -> None:
        """Storing hyper parameters in json files for later inference

        Args:
            dir (str): directory where the json is stored
        """
        args_dir = os.path.join(dir, "args.json")
        kwargs_dir = os.path.join(dir, "kwargs.json")
        with open(args_dir, "w", encoding="utf-8") as f:
            json.dump(self.args, f)

        with open(kwargs_dir, "w", encoding="utf-8") as f:
            json.dump(self.kwargs, f)
        return
    
    def get_metadata(self) -> dict:
        return self.args[1]
    
    def learn(self,
              dataloaders: list[torch.utils.data.DataLoader],
              loss_fn: any,
              optimizer: torch.optim,
              vis_logger: Union[None, TransformerForecasterVisualLogger] = None,
              ) -> float:
        """
        Return the loss of the training
        """
        # Start training
        self.train()

        # Metadata
        total_length = sum([len(dataloader) for dataloader in dataloaders])
        bar = tqdm(
            total       = total_length, 
            position    = 1,
            colour      = GREEN,
            )
        total_loss = 0
        
        # Iterate through dataloaders
        batch_cnt = 0
        for dataloader in dataloaders:
            for (src, tgt, tgt_y) in dataloader:
                src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Make forecasts
                prediction = self(src, tgt)

                # Compute and backprop loss
                loss = loss_fn(prediction, tgt_y)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                # Take optimizer step
                optimizer.step()

                # Add data to val_logger
                if vis_logger != None:
                    vis_logger.append_to_forecast_plotter(tgt_y, prediction)
                    vis_logger.append_to_loss_console_plotter(loss.item())

                bar.set_description(desc=f"Instant loss: {loss:.3f}, Continuous loss: {(total_loss/(batch_cnt+1)):.3f}", refresh=True)
                batch_cnt += 1
                bar.update()
            if vis_logger != None:
                vis_logger.signal_new_dataloader()
        bar.colour = BLACK
        bar.close()

        return total_loss/total_length
    
    def val(self,
            dataloaders: list[torch.utils.data.DataLoader],
            loss_fn: any,
            metrics: list,
            vis_logger: Union[None, TransformerForecasterVisualLogger] = None,
            ) -> float:
        """
        Which to plot: receive a list that contains the forecast sequence needed to be plotted
        scaler: receive a tuple, (average, stddev). The scaler is used to reproduce original values, instead of the normalized ones.
        Return the loss of validation
        """
        # Metadata
        total_batches = sum([len(dataloader) for dataloader in dataloaders])

        # Start evaluation
        self.eval()

        additional_loss = {}
        for additional_monitor in metrics:
            additional_loss[str(type(additional_monitor))] = 0
        
        # Validation
        with torch.no_grad():
            test_loss = 0
            correct = 0
            bar = tqdm(
                total       = total_batches, 
                position    = 1,
                colour      = GREEN,
                )
            batch_cnt = 0
            for dataloader in dataloaders:
                for (src, tgt, tgt_y) in dataloader:
                    src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)

                    pred = self(src, tgt)
                    loss = loss_fn(pred, tgt_y).item()
                    test_loss += loss
                    # TODO: Check accuracy calculation
                    correct += (pred == tgt_y).type(torch.float).sum().item()

                    if vis_logger != None:
                        vis_logger.append_to_forecast_plotter(tgt_y, pred)
                        vis_logger.append_to_loss_console_plotter(loss)

                    for additional_monitor in metrics:
                        additional_loss[str(type(additional_monitor))] += additional_monitor(pred, tgt_y).item()
                    bar.update()
                    bar.set_description(desc=f"Loss: {(test_loss/(1+batch_cnt)):.3f}", refresh=True)
                    batch_cnt += 1
                if vis_logger != None:
                    vis_logger.signal_new_dataloader()
            bar.colour = BLACK
            bar.close()
        test_loss /= total_batches
        correct /= total_batches
        
        # Report
        tqdm.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")
        for additional_monitor in metrics:
            name = str(type(additional_monitor))[8:-2].split(".")[-1]
            loss = additional_loss[str(type(additional_monitor))] / total_batches
            tqdm.write(f" {name}: {loss:>8f}")

        return test_loss


class ClassifierTransformer(TimeSeriesTransformer):
    def __init__(self, 
                 input_size: int,
                 forecast_feature_size: int,
                 metadata: dict,
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
        super().__init__(
            input_size,
            metadata,
            *args, 
            embedding_dimension = embedding_dimension,
            multi_head_attention_head_size = multi_head_attention_head_size,
            num_of_encoder_layers = num_of_encoder_layers,
            num_of_decoder_layers= num_of_decoder_layers,
            model_name = model_name,
            **kwargs
        )
        # Store all the args and kwargs for restoring
        self.args = [
            input_size,
            forecast_feature_size,
            metadata,
        ]
        
        self.model_name = model_name

        # Output embedding
        self.decoder_input_layer = nn.Linear(
            in_features     = forecast_feature_size,
            out_features    = embedding_dimension,
        )
        
        # Final forecast output of decoder
        self.final_output = nn.Linear(
            in_features     = embedding_dimension,
            out_features    = forecast_feature_size
        )

    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor,
                ) -> torch.Tensor:
        result = super().forward(src, tgt)
        return result
    
    def learn(self,
              dataloaders: list[torch.utils.data.DataLoader],
              loss_fn: any,
              optimizer: torch.optim,
              vis_logger: Union[None, TransformerForecasterVisualLogger] = None,
              ) -> float:
        """
        Return the loss of the training
        """
        # Start training
        self.train()

        # Metadata
        total_length = sum([len(dataloader) for dataloader in dataloaders])
        bar = tqdm(
            total       = total_length, 
            position    = 1,
            colour      = GREEN,
            )
        total_loss = 0
        
        # Iterate through dataloaders
        batch_cnt = 0
        for dataloader in dataloaders:
            for (src, tgt, tgt_y) in dataloader:
                src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Make forecasts
                prediction = self(src, tgt)

                # Compute and backprop loss
                transformed_prediction = torch.permute(prediction, (1, 2, 0))
                transformed_tgt_y = torch.permute(tgt_y, (1, 2, 0))

                loss = loss_fn(transformed_prediction, transformed_tgt_y)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                # Take optimizer step
                optimizer.step()

                # Add data to val_logger
                if vis_logger != None:
                    vis_logger.append_to_forecast_plotter(tgt_y, prediction)
                    vis_logger.append_to_loss_console_plotter(loss.item())

                bar.set_description(desc=f"Instant loss: {loss:.3f}, Continuous loss: {(total_loss/(batch_cnt+1)):.3f}", refresh=True)
                batch_cnt += 1
                bar.update()
            if vis_logger != None:
                vis_logger.signal_new_dataloader()
        bar.colour = BLACK
        bar.close()

        return total_loss/total_length
    
    def val(self,
            dataloaders: list[torch.utils.data.DataLoader],
            loss_fn: any,
            metrics: list,
            vis_logger: Union[None, TransformerForecasterVisualLogger] = None,
            ) -> float:
        """
        Which to plot: receive a list that contains the forecast sequence needed to be plotted
        scaler: receive a tuple, (average, stddev). The scaler is used to reproduce original values, instead of the normalized ones.
        Return the loss of validation
        """
        # Metadata
        total_batches = sum([len(dataloader) for dataloader in dataloaders])

        # Start evaluation
        self.eval()

        additional_loss = {}
        for additional_monitor in metrics:
            additional_loss[str(type(additional_monitor))] = 0

        # Validation
        with torch.no_grad():
            test_loss = 0
            correct = 0
            bar = tqdm(
                total       = total_batches, 
                position    = 1,
                colour      = GREEN,
                )
            batch_cnt = 0
            for dataloader in dataloaders:
                for (src, tgt, tgt_y) in dataloader:
                    src, tgt, tgt_y = src.to(self.device), tgt.to(self.device), tgt_y.to(self.device)

                    pred = self(src, tgt)

                    # The transformation is for the input format requirement of the CrossEntropy
                    transformed_prediction = torch.permute(pred, (1, 2, 0))
                    transformed_tgt_y = torch.permute(tgt_y, (1, 2, 0))

                    loss = loss_fn(transformed_prediction, transformed_tgt_y).item()
                    test_loss += loss
                    correct += (pred == tgt_y).type(torch.float).sum().item()

                    if vis_logger != None:
                        vis_logger.append_to_forecast_plotter(tgt_y, pred)
                        vis_logger.append_to_loss_console_plotter(loss)

                    for additional_monitor in metrics:
                        additional_loss[str(type(additional_monitor))] += additional_monitor(pred, tgt_y).item()
                    bar.update()
                    bar.set_description(desc=f"Loss: {(test_loss/(1+batch_cnt)):.3f}", refresh=True)
                    batch_cnt += 1
                if vis_logger != None:
                    vis_logger.signal_new_dataloader()
            bar.colour = BLACK
            bar.close()
        test_loss /= total_batches
        correct /= total_batches
        
        # Report
        tqdm.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} ")
        for additional_monitor in metrics:
            name = str(type(additional_monitor))[8:-2].split(".")[-1]
            loss = additional_loss[str(type(additional_monitor))] / total_batches
            tqdm.write(f" {name}: {loss:>8f}")

        return test_loss
        
    def get_metadata(self) -> dict:
        return self.args[2]

"""

██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║   
██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║   
██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║   
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   
                                                          
"""

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
        result[i] = [torch.Tensor(temp[i]) for temp in data]
        result[i] = torch.stack(result[i])
        result[i] = result[i].permute(1, 0, 2)
    return result[0], result[1], result[2]
