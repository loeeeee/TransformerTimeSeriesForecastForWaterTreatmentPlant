import torch
from torch import nn

"""
TODO: Hahahaha XD
"""

class DualStageAttentionBasedRecurrentNeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return