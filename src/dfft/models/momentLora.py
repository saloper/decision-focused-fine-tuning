"""
Module Name: momentLora.py
Description: Code for MOMENT foundational timeseries model augmented with LoRA 
             (Low-Rank Adaptation) for Parameter-Efficient Fine-Tuning.
Usage: Drop-in replacement for MOMENTForecaster in the training pipeline.
"""

import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from peft import LoraConfig, get_peft_model
from .base import BaseForecaster

class MOMENTLoRAForecaster(BaseForecaster):
    def __init__(self, 
                 input_len: int, 
                 output_len: int, 
                 model_name: str = "AutonLab/MOMENT-1-small", 
                 r: int = 8, 
                 lora_alpha: int = 16, 
                 lora_dropout: float = 0.05,
                 random_seed: int = 13):
        
        super().__init__(input_len, output_len)
        control_randomness(seed=random_seed)

        # 1. Load MOMENT with base weights frozen
        self.moment_pipeline = MOMENTPipeline.from_pretrained(
            model_name, 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': output_len,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,  # Keep the base transformer frozen
                'freeze_embedder': True, # Keep the patch embedding frozen
                'freeze_head': False,    # Keep the linear forecasting head unfrozen
            }
        )
        self.moment_pipeline.init()
        
        # 2. Configure LoRA
        # MOMENT uses a T5 architecture. The attention layers are named 'q', 'k', 'v', 'o'.
        # We inject adapters into Query ('q') and Value ('v') projections.
        lora_config = LoraConfig(
            r=r,                            # Rank of the update matrices
            lora_alpha=lora_alpha,          # LoRA scaling factor
            target_modules=["q", "v"],      # Target the T5 attention matrices
            lora_dropout=lora_dropout,
            bias="none",
            # CRITICAL: We must tell PEFT not to freeze our randomly initialized forecasting head!
            modules_to_save=["head"]        
        )
        
        # 3. Wrap the pipeline with PEFT
        self.moment_pipeline = get_peft_model(self.moment_pipeline, lora_config)
        
        # Print trainable parameters to verify LoRA and Head are active
        print("LoRA Adapters Injected Successfully.")
        self.moment_pipeline.print_trainable_parameters()

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, 1, input_len)
        if input_mask is not None:
            output = self.moment_pipeline(x_enc=x, input_mask=input_mask)
        else:
            output = self.moment_pipeline(x_enc=x)
            
        # The pipeline returns a TimeseriesOutputs object; we just want the forecast
        return output.forecast.squeeze(1) # Output is (batch_size, output_len)