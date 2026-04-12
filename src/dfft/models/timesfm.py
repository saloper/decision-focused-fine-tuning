"""
Module Name: timesfm.py
Description: TimesFM models (Zero-shot, Linear Head Finetuning, LoRA Finetuning)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from .base import BaseForecaster


# ============================================================
# Base Wrapper
# ============================================================
class TimesFMBase(BaseForecaster):
    def __init__(self, input_len: int, output_len: int, model_name: str):
        super().__init__(input_len, output_len)

        self.model = AutoModel.from_pretrained(model_name)
        self.output_len = output_len

    def forward_backbone(self, x):
        """
        x: (B, 1, L)
        """
        # TimesFM expects (B, L)
        x = x.squeeze(1)

        outputs = self.model(x)
        hidden = outputs.last_hidden_state  # (B, L, D)

        # Take last token representation (common practice)
        return hidden[:, -1, :]  # (B, D)


# ============================================================
# 1. ZERO-SHOT
# ============================================================
class TimesFMZeroShot(TimesFMBase):
    def __init__(self, input_len, output_len, model_name, random_seed=42):
        super().__init__(input_len, output_len, model_name)

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # No training head → direct projection using a frozen linear layer
        hidden_dim = self.model.config.hidden_size
        self.head = nn.Linear(hidden_dim, output_len)

        # Freeze head too (true zero-shot)
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.forward_backbone(x)
        return self.head(features)


# ============================================================
# 2. LINEAR HEAD FINETUNING
# ============================================================
class TimesFMLinearHead(TimesFMBase):
    def __init__(self, input_len, output_len, model_name, random_seed=42):
        super().__init__(input_len, output_len, model_name)

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False

        hidden_dim = self.model.config.hidden_size

        # Trainable head
        self.head = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        features = self.forward_backbone(x)
        return self.head(features)


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

        # LoRA config
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # typical HF attention names
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["head"]
        )

        # Inject LoRA
        self.model = get_peft_model(self.model, lora_config)

        hidden_dim = self.model.config.hidden_size

        # Trainable head
        self.head = nn.Linear(hidden_dim, output_len)

        print("LoRA adapters added to TimesFM.")
        self.model.print_trainable_parameters()

    def forward(self, x):
        features = self.forward_backbone(x)
        return self.head(features)