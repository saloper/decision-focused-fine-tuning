"""
Module Name: timesfm.py
Description: TimesFM models (Zero-shot, Linear Head Finetuning, LoRA Finetuning)
"""

import torch
import torch.nn as nn
from transformers import TimesFm2_5ModelForPrediction
from peft import LoraConfig, get_peft_model
from .base import BaseForecaster
# ============================================================
# Base Wrapper
# ============================================================
class TimesFMBase(BaseForecaster):
    def __init__(self, input_len: int, output_len: int, model_name: str):
        super().__init__(input_len, output_len)
        self.output_len = output_len
        
        # Load the model INCLUDING the pre-trained forecasting head
        self.model = TimesFm2_5ModelForPrediction.from_pretrained(model_name)

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

# ============================================================
# 1. ZERO-SHOT
# ============================================================
class TimesFMZeroShot(TimesFMBase):
    def __init__(self, input_len, output_len, model_name, random_seed=42):
        super().__init__(input_len, output_len, model_name)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        print("TimesFM Zero-Shot initialized. All parameters frozen.")
        self.print_trainable_parameters()

    def forward(self, x):
        """
        x: (B, 1, L)
        """
        x = x.squeeze(1)
        
        # TimesFM native forward pass (handles internal denormalization)
        outputs = self.model(past_values=x)
        
        # Extract the point forecasts (mean predictions)
        return outputs.mean_predictions[:, -self.output_len:]


# ============================================================
# 2. NATIVE HEAD FINETUNING (Replaces Linear Head)
# ============================================================
class TimesFMLinearHead(TimesFMBase):
    def __init__(self, input_len, output_len, model_name, random_seed=42):
        super().__init__(input_len, output_len, model_name)

        # Freeze the entire backbone first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze ONLY the native forecasting head. 
        # This is strictly better than an nn.Linear layer because it 
        # preserves TimesFM's structural patching and de-normalization.
        for name, p in self.model.named_parameters():
            if any(keyword in name for keyword in ["head", "scoring", "output_layer"]):
                p.requires_grad = True
        print("Trainable paramereters in TimesFM Linear Head:")
        self.print_trainable_parameters()

    def forward(self, x):
        x = x.squeeze(1)
        outputs = self.model(past_values=x)
        return outputs.mean_predictions[:, -self.output_len:]


# ============================================================
# 3. LORA FINETUNING
# ============================================================
class TimesFMLoRA(TimesFMBase):
    def __init__(self,
                 input_len,
                 output_len,
                 model_name,
                 r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 random_seed=42):

        super().__init__(input_len, output_len, model_name)

        # Freeze base first
        for p in self.model.parameters():
            p.requires_grad = False

        # LoRA config targeting typical Hugging Face attention projections
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=lora_dropout,
            bias="none"
        )

        # Inject LoRA adapters into the backbone
        self.model = get_peft_model(self.model, lora_config)

        # Ensure the native forecasting head is also unfrozen so it can 
        # learn alongside the new LoRA adapters
        for name, p in self.model.named_parameters():
            if any(keyword in name for keyword in ["head", "scoring", "output_layer"]):
                p.requires_grad = True

        print("LoRA adapters added to TimesFM.")
        self.print_trainable_parameters()

    def forward(self, x):
        x = x.squeeze(1)
        outputs = self.model(past_values=x)
        return outputs.mean_predictions[:, -self.output_len:]