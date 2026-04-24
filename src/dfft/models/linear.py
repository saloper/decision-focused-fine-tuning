import torch
import torch.nn as nn
from .base import BaseForecaster


class LinearForecaster(BaseForecaster):
    def __init__(self, input_len: int, output_len: int, in_channels: int = 1):
        super().__init__(input_len, output_len)
        self.projection = nn.Linear(input_len * in_channels, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, input_len)
        flattened = x.reshape(x.size(0), -1)
        return self.projection(flattened)
