import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, device: str,
                 scheduler=None, max_norm: float = 5.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        
        # We compute both MSE and MAE internally
        self.mse_criterion = nn.MSELoss().to(device)
        self.mae_criterion = nn.L1Loss().to(device)
        
        self.device = device
        self.scheduler = scheduler
        self.max_norm = max_norm
        
        # Enable mixed precision scaling
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))
        self.logger = logging.getLogger(__name__)

    def _compute_metrics(self, predictions, targets):
        mse = self.mse_criterion(predictions, targets)
        mae = self.mae_criterion(predictions, targets)
        return mse, mae

    def train_epoch(self):
        self.model.train()
        mse_losses = []
        mae_losses = []
        
        for batch_data in tqdm(self.train_loader, desc="Training", leave=False):
            # Handle both regular and temporal feature batches
            if len(batch_data) == 2:
                x, y = batch_data
                hours, days_of_week = None, None
            elif len(batch_data) == 4:
                x, y, hours, days_of_week = batch_data
                hours = hours.float().to(self.device)
                days_of_week = days_of_week.float().to(self.device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            
            with torch.amp.autocast(enabled=(self.device == 'cuda'), device_type=self.device):
                # Pass temporal features if available
                if hours is not None:
                    predictions = self.model(x, hours=hours, days_of_week=days_of_week)
                else:
                    predictions = self.model(x)
                mse, mae = self._compute_metrics(predictions, y)
                loss = mse # Use MSE as primary loss for optimization
            
            # Scales the loss for mixed precision training
            self.scaler.scale(loss).backward()
            
            # Clip gradients to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            mse_losses.append(mse.item())
            mae_losses.append(mae.item())
            
        return {"mse": np.average(mse_losses), "mae": np.average(mae_losses)}

    def evaluate(self, phase="Validating", loader=None):
        self.model.eval()
        mse_losses = []
        mae_losses = []
        
        all_preds = []
        all_trues = []

        if loader is None:
            loader = self.val_loader
        
        with torch.no_grad():
            for batch_data in tqdm(loader, desc=phase, leave=False):
                # Handle both regular and temporal feature batches
                if len(batch_data) == 2:
                    x, y = batch_data
                    hours, days_of_week = None, None
                elif len(batch_data) == 4:
                    x, y, hours, days_of_week = batch_data
                    hours = hours.float().to(self.device)
                    days_of_week = days_of_week.float().to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                
                with torch.amp.autocast(enabled=(self.device == 'cuda'), device_type=self.device):
                    # Pass temporal features if available
                    if hours is not None:
                        predictions = self.model(x, hours=hours, days_of_week=days_of_week)
                    else:
                        predictions = self.model(x)
                    mse, mae = self._compute_metrics(predictions, y)
                    
                mse_losses.append(mse.item())
                mae_losses.append(mae.item())
                
                all_preds.append(predictions.cpu().numpy())
                all_trues.append(y.cpu().numpy())
                
        metrics = {"mse": np.average(mse_losses), "mae": np.average(mae_losses)}
        preds_array = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        trues_array = np.concatenate(all_trues, axis=0) if all_trues else np.array([])
        
        return metrics, preds_array, trues_array

    def test(self):
        return self.evaluate(phase="Testing", loader=self.test_loader)
