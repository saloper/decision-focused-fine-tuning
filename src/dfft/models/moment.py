"""
Module Name: moment.py
Description: code for MOMENT foundational timeseries model
Usage: likely an important to a training script
"""

import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from .base import BaseForecaster

class MOMENTForecaster(BaseForecaster):
    def __init__(self, input_len: int, output_len: int, model_name: str = "AutonLab/MOMENT-1-small", freeze_linear: bool = True, random_seed: int = 13):
        super().__init__(input_len, output_len)
        control_randomness(seed=random_seed)

        # Load MOMENT exactly as specified in the official guidelines
        self.moment_pipeline = MOMENTPipeline.from_pretrained(
            model_name, 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': output_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': freeze_linear,  # Freeze the patch embedding layer
                'freeze_embedder': True, # Freeze the transformer encoder
                'freeze_head': freeze_linear,              # Unfreeze the linear forecasting head if asked
            }
        )
        self.moment_pipeline.init()
        
        # Quick print to verify trainable parameters
        print("Unfrozen parameters:")
        for name, param in self.moment_pipeline.named_parameters():    
            if param.requires_grad:
                print('    ', name)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, 1, input_len)
        if input_mask is not None:
            output = self.moment_pipeline(x_enc=x, input_mask=input_mask)
        else:
            output = self.moment_pipeline(x_enc=x)
            
        # The pipeline returns a TimeseriesOutputs object; we just want the forecast
        return output.forecast.squeeze(1) # Output is (batch_size, output_len) after squeezing the channel dimension