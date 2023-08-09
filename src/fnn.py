import os
import json
import torch

from torch import nn
from termcolor import cprint
from torch.utils.data import Dataset


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_ff: int = 2048, device: str = "cpu"):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_in, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_out, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            d_ff: int = 2048,
            n_depth: int = 10,
            device: str = "cpu",
            hyperparameter_dict: dict = None,
            ):
        super().__init__()

        self.input_layer = nn.Linear(
            in_features=d_in,
            out_features=int(d_ff/2),
            device=device,
        )

        self.feed_forward_layers = nn.ModuleList(
            [
                PositionWiseFeedForward(int(d_ff/2), int(d_ff/2), d_ff=d_ff, device=device)
                for i in range(n_depth)
            ]
        )

        self.output_layer = nn.Linear(
            in_features=int(d_ff/2),
            out_features=d_out,
            device=device,
        )

        self.hyperparameter = hyperparameter_dict

    def forward(self, X) -> torch.Tensor:
        X = self.input_layer(X)
        for layer in self.feed_forward_layers:
            X = layer(X)
        result = self.output_layer(X)
        return result
    
    def dump_hyperparameter(self, working_dir: str) -> None:
        cprint("Dumping hyperparameter to json file", "green")
        with open(os.path.join(working_dir, "hyperparameter.json"), mode="w", encoding="utf-8") as f:
            json.dump(self.hyperparameter, f, indent=2)
        return

class FeedForwardNeuralNetworkDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None, target_transform=None, device: str = "cpu"):
        """
        X is a DataFrame
        y is a DataFrame
        """
        self.X = X
        self.y = y
        self.device = device

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], device=self.device)
        y = torch.tensor(self.y[idx]).type(torch.LongTensor).to(self.device)
        return X, y