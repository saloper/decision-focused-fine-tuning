import torch
import torch.nn as nn
from .base import BaseForecaster


class MLPForecaster(BaseForecaster):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        in_channels: int = 1,
        hidden_dims=None,
        dropout: float = 0.1,
    ):
        super().__init__(input_len, output_len)
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_len * in_channels
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_len))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, input_len)
        flattened = x.reshape(x.size(0), -1)
        return self.network(flattened)
