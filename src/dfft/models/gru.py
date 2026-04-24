import torch
import torch.nn as nn
from .base import BaseForecaster


class GRUForecaster(BaseForecaster):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_dim: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__(input_len, output_len)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * num_directions),
            nn.Linear(hidden_size * num_directions, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, input_len) -> (batch, input_len, channels)
        x_seq = x.transpose(1, 2)
        encoded, _ = self.encoder(x_seq)
        last_hidden = encoded[:, -1, :]
        return self.head(last_hidden)
