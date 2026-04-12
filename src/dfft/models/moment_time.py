"""
Module Name: moment_time.py
Description: MOMENT model augmented with temporal features (hour-of-day, day-of-week)
             to leverage the strong predictive power of time-of-day in energy markets
             (solar duck curve, evening load peaks, etc.).
Usage: Drop-in replacement for MOMENTForecaster in the training pipeline
"""

import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from .base import BaseForecaster

class MOMENTTimeForecaster(BaseForecaster):
    def __init__(self, input_len: int, output_len: int, 
                 model_name: str = "AutonLab/MOMENT-1-small", 
                 freeze_encoder: bool = True, 
                 freeze_linear: bool = True, 
                 random_seed: int = 13):
        super().__init__(input_len, output_len)
        control_randomness(seed=random_seed)

        # Load MOMENT with specified freezing configuration
        self.moment_pipeline = MOMENTPipeline.from_pretrained(
            model_name, 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': output_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': freeze_encoder,
                'freeze_embedder': True,
                'freeze_head': freeze_linear,
            }
        )
        self.moment_pipeline.init()
        
        # Get the dimension of MOMENT's output embeddings
        # MOMENT outputs are typically of shape (batch, output_len)
        # We need to extract embedding dimension from the model
        self.embedding_dim = 1  # Output dimension per timestep
        
        # Temporal feature dimensions:
        # - hour_of_day: 1 (normalized 0-23)
        # - day_of_week: 1 (normalized 0-6)
        temporal_dim = 2
        
        # Create a custom head that takes concatenated [MOMENT_output, hour, day_of_week]
        # We'll use a simple MLP to process this
        total_input_dim = self.embedding_dim + temporal_dim
        
        self.temporal_head = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        print("MOMENTTime initialized with temporal feature injection.")
        print("Unfrozen parameters in MOMENT:")
        for name, param in self.moment_pipeline.named_parameters():    
            if param.requires_grad:
                print('    ', name)

    def forward(self, x: torch.Tensor, 
                hours: torch.Tensor = None, 
                days_of_week: torch.Tensor = None,
                input_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Time series values of shape (batch_size, 1, input_len)
            hours: Hour of day for output timesteps (batch_size, output_len) with values 0-23
            days_of_week: Day of week for output timesteps (batch_size, output_len) with values 0-6
            input_mask: Optional mask for input
            
        Returns:
            Predictions of shape (batch_size, output_len)
        """
        # Get MOMENT's base forecast
        if input_mask is not None:
            output = self.moment_pipeline(x_enc=x, input_mask=input_mask)
        else:
            output = self.moment_pipeline(x_enc=x)
        
        # MOMENT outputs shape: (batch_size, output_len) after squeeze
        moment_forecast = output.forecast.squeeze(1)  # (batch_size, output_len)
        
        # If temporal features are provided, enhance the forecast with them
        if hours is not None and days_of_week is not None:
            batch_size = moment_forecast.shape[0]
            output_len = moment_forecast.shape[1]
            
            # Normalize temporal features to 0-1 range
            hours_normalized = hours / 24.0  # (batch_size, output_len)
            days_normalized = days_of_week / 7.0  # (batch_size, output_len)
            
            # Reshape MOMENT forecast for concatenation
            moment_flat = moment_forecast.view(batch_size * output_len, 1)
            hours_flat = hours_normalized.view(batch_size * output_len, 1)
            days_flat = days_normalized.view(batch_size * output_len, 1)
            
            # Concatenate temporal features with MOMENT output
            temporal_enhanced = torch.cat([moment_flat, hours_flat, days_flat], dim=1)
            
            # Pass through temporal head
            refined_forecast = self.temporal_head(temporal_enhanced)
            refined_forecast = refined_forecast.view(batch_size, output_len)
            
            return refined_forecast
        else:
            # If no temporal features provided, just return MOMENT's forecast
            return moment_forecast
