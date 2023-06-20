import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

from tqdm import tqdm
from termcolor import colored

import os
import sys
import copy


class GenericDataFrameDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, transform=None, target_transform=None):
        """
        X is a DataFrame
        y is a DataFrame
        """
        self.X = torch.tensor(X.values, dtype = torch.float32)
        self.y = torch.tensor(y.values, dtype = torch.float32).unsqueeze(1)
        self.transform = transform
        self.target_transform = target_transform

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
        If the Loss have not improve in continues three epoches,
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
        if self.patience == -1:
            return True

        self.loss.append(loss)
        
        # When it is in early epoch
        if len(self.loss) <= 1:
            return True
        self.loss_delta.append(self.loss[-1] - self.loss[-2])

        # When a good model is meet
        if loss < self.lowest_loss:
            self.known_best_model = copy.deepcopy(model)
            self.lowest_loss = loss
        
        if sum(self.loss_delta[-self.patience:]) >= 0 or self.loss_delta[-1] >= 0:
            self.patience_left -= 1
        else:
            self.patience_left = self.patience

        return bool(self.patience_left)

    def get_best_model(self):
        return self.known_best_model

    def get_loss_history(self) -> list:
        return self.loss

       
def get_best_device() -> str:
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    return device
