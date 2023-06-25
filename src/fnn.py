import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

from tqdm import tqdm


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, name: str):
        super().__init__()
        self.model_name = name
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

    def learn(self, 
              dataloader: DataLoader, 
              loss_fn, 
              optimizer: torch.optim, 
              device: str
              ) -> None:
        size = len(dataloader.dataset)
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                tqdm.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def val(self, 
            dataloader: DataLoader, 
            loss_fn, 
            device: str, 
            metrics: list
            ) -> torch.Tensor:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0
        additional_loss = {}
        for additional_monitor in metrics:
            additional_loss[str(type(additional_monitor))] = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
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