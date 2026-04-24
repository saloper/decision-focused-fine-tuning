import torch
import torch.nn as nn
from .base import BaseForecaster


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def _causal_trim(self, x: torch.Tensor, trim: int) -> torch.Tensor:
        if trim <= 0:
            return x
        return x[:, :, :-trim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trim1 = self.conv1.padding[0]
        out = self.conv1(x)
        out = self._causal_trim(out, trim1)
        out = self.relu(out)
        out = self.dropout(out)

        trim2 = self.conv2.padding[0]
        out = self.conv2(out)
        out = self._causal_trim(out, trim2)
        out = self.relu(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNForecaster(BaseForecaster):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(input_len, output_len)

        blocks = []
        current_in = in_channels
        for level in range(num_layers):
            dilation = 2 ** level
            blocks.append(
                _TemporalBlock(
                    in_channels=current_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_in = hidden_channels

        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, input_len)
        features = self.network(x)
        last_timestep = features[:, :, -1]
        return self.head(last_timestep)
