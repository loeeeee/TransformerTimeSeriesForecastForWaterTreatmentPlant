import torch
from torch import nn
from torch.utils.data import Dataset

from typing import Union
from termcolor import colored
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import copy


class GenericDataFrameDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None, target_transform=None):
        """
        X is a DataFrame
        y is a DataFrame
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TrackerEpoch:
    def __init__(self, epoch: int):
        """
        Track epoch count
        """
        self.max_epoch = epoch
        self.current_epoch = 0

    def check(self) -> bool:
        self.current_epoch += 1
        if self.current_epoch > self.max_epoch:
            return False
        else:
            return True

    def epoch(self) -> int:
        return self.current_epoch


class TrackerLoss:
    def __init__(self, patience: int, model):
        """
        If the Loss have not improve in continues three epochs,
        it signals a stop
        """
        self.loss = []
        self.loss_delta = []
        self.lowest_loss = 0x3f3f3f3f
        self.known_best_model = copy.deepcopy(model)
        if patience <= 0:
            print("TrackLoss: NOT GOOD")
        self.patience = patience
        self.patience_left = patience

    def check(self, loss, model) -> bool:
        """
        If loss improve in all epoch,
        If loss improve in previous epoch, but worse in recent ones,
        If loss worse in all epoch,
        If loss worse in previous epoch, but improve in recent ones
        """
        self.loss.append(loss)
        
        # When it is in early epoch
        if len(self.loss) <= 1:
            self.loss_delta.append(-self.loss[-1])
            return True
        
        self.loss_delta.append(self.loss[-1] - self.loss[-2])

        # When a good model is meet
        if loss < self.lowest_loss:
            self.known_best_model = copy.deepcopy(model)
            self.lowest_loss = loss

        if self.patience == -1:
            # Force ignore the patience
            return True
        
        if sum(self.loss_delta) >= 0 or self.loss_delta[-1] >= 0:
            # When the loss is not decreasing
            self.patience_left -= 1
        else:
            # When the loss decreases
            self.patience_left = self.patience
        
        return bool(self.patience_left)

    def get_best_model(self):
        return self.known_best_model

    def get_loss_history(self) -> list:
        return self.loss
    
    def get_trend(self, scope: int) -> float:
        """Get the trend of loss

        Args:
            scope (int): from how long ago it starts measuring the loss trend

        Returns:
            float: the loss trend represented in a positive or negative float
        """
        if len(self.loss_delta) < scope:
            return sum(self.loss_delta)
        else:
            return sum(self.loss_delta[-scope:])


def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

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
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=["timestamp"], inplace=True)

    return data

def save_model(model: nn.Module, root_saving_dir: str, dataloader, device: str="cpu") -> None:
    print(f"Save data to {root_saving_dir}")
    save_dir = os.path.join(root_saving_dir, f"{model.name}.onnx")
    example_input = next(iter(dataloader))
    args = (example_input[0].to(device=device), example_input[1].to(device=device))
    model.to(device=device)
    model.eval()
    with torch.autocast(device_type=device):
        scripted = torch.jit.trace(model, example_inputs=args, check_trace=False) # HACK
    torch.onnx.export(
        model = scripted,
        args = args,
        f = save_dir,
        export_params = True,        # store the trained parameter weights inside the model file
        do_constant_folding = True,  # whether to execute constant folding for optimization
        input_names = ['encoder', 'decoder'],   # the model's input names
        output_names = ['forecast'], # the model's output names
        dynamic_axes = {'encoder' : {0: 'batch_size', 1: 'flatten_encoder_sequence'},    # variable length axes
                        'decoder' : {0: 'batch_size', 1: 'flatten_decoder_sequence'},
                        'forecast': {0: 'batch_size', 1: 'output_sequence'}
                        },
        opset_version = 18,
        )
    return

def console_general_data_info(data: pd.DataFrame) -> None:
    print(colored("\nGeneral data information:", "green"))
    print(colored("Example data:", "blue"))
    print(data.head())
    print(colored("Data types:", "blue"))
    print(data.dtypes)
    print(colored("Additional information:", "blue"))
    print(data.info, "\n")
    return

def create_folder_if_not_exists(dir) -> bool:
    """
    Return bool, indicating if the folder is *newly* created, True means exists
    """
    if not (isExist := os.path.exists(dir)):
        os.mkdir(dir)
    return isExist

def new_remove(input_list: list, sth_to_remove: Union[str, int, float, bool]) -> list:
    """
    Remove the first item from the list with the given name
    """
    input_list.remove(sth_to_remove)
    return input_list

def visualize_loss(t_loss: TrackerLoss, root_saving_dir: str, model_name: str) -> None:
    # Visualize training process
    loss_history = t_loss.get_loss_history()
    fig_name = f"{model_name}_loss_history"
    plt.plot(range(len(loss_history)), loss_history)
    plt.title(fig_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.figtext(0, 0, f"Minimum loss: {t_loss.lowest_loss}", color="#a41095")
    plt.savefig(os.path.join(root_saving_dir, f"{fig_name}.png"), dpi=300)
    plt.clf()
    return

def split_data(data: pd.DataFrame):
    # Split test, validation and training data
    ## Split the DataFrame into features and target variable
    X = data.drop(columns=['f16', 'f17', 'f18', 'f31', 'f32'])
    y = data[('f16', 'pac泵1频率')]
    ## Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    ## Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    ## Print the shapes of each set
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test
