import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseForecaster(nn.Module, ABC):
    """
    Abstract base class for all time-series forecasting models in this project.
    Ensures that all models, from baselines to foundation models, share the same API.
    """
    def __init__(self, input_len: int, output_len: int):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, channels, input_len)
        Returns:
            Tensor of shape (batch_size, output_len)
        """
        pass
    