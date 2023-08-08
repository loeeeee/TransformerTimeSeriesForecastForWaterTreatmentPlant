import torch
from torch import nn

class FeedForwardNeuralNetwork(nn.modules):
    def __init__(self):
        super.__init__()

    def forward(self, X) -> torch.Tensor:

        return