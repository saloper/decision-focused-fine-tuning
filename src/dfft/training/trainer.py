import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
from dfft.models.aribtrage import BatteryAbritrageLayer

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, device: str,
                 scheduler=None, max_norm: float = 5.0, horizon: int = 24, decision_loss: bool = False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        
        # We compute both MSE and MAE internally
        self.mse_criterion = nn.MSELoss().to(device)
        self.mae_criterion = nn.L1Loss().to(device)
        #Create optimization layer to compute solution
        self.decision_loss = decision_loss
        self.bess = BatteryAbritrageLayer(T=horizon).to(device)

        self.device = device
        self.scheduler = scheduler
        self.max_norm = max_norm
        
        # Enable mixed precision scaling
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))
        self.logger = logging.getLogger(__name__)

    def _compute_metrics(self, predictions, targets):
        mse = self.mse_criterion(predictions, targets)
        mae = self.mae_criterion(predictions, targets)
        #Scale to a reasonable scale
        scaler = torch.max(torch.abs(predictions), dim=1, keepdim=True)[0].detach()
        scaled_predictions = predictions / (scaler + 1e-8)
        c_sol, d_sol, _ = self.bess(scaled_predictions) #Note the solution is scale invariant, so this is still valid
        profit = torch.mean(torch.sum(targets * (d_sol - c_sol), dim=1))
        return mse, mae, profit

    def train_epoch(self):
        self.model.train()
        mse_losses = []
        mae_losses = []
        profit_losses = []
        
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
                mse, mae, profit = self._compute_metrics(predictions, y)
    
                #Set loss function
                if self.decision_loss:
                    loss = -profit
                else:
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
            profit_losses.append(profit.item())
            
        return {"mse": np.average(mse_losses), "mae": np.average(mae_losses), "profit": np.average(profit_losses)}

    def evaluate(self, phase="Validating", loader=None):
        self.model.eval()
        mse_losses = []
        mae_losses = []
        profit_losses = []
        
        all_preds = []
        all_trues = []
        all_pred_c_sol = []
        all_pred_d_sol = []
        all_pred_s_sol = []
        all_pred_profit = []
        all_true_c_sol = []
        all_true_d_sol = []
        all_true_s_sol = []
        all_true_profit = []
        all_real_profit = []
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
                    mse, mae, profit = self._compute_metrics(predictions, y)
                    pred_c_sol, pred_d_sol, pred_s_sol = self.bess(predictions)
                    true_c_sol, true_d_sol, true_s_sol = self.bess(y)
                    
                mse_losses.append(mse.item())
                mae_losses.append(mae.item())
                profit_losses.append(profit.item())
                
                all_preds.append(predictions.cpu().numpy())
                all_trues.append(y.cpu().numpy())
                all_pred_c_sol.append(pred_c_sol.cpu().numpy())
                all_pred_d_sol.append(pred_d_sol.cpu().numpy())
                all_pred_s_sol.append(pred_s_sol.cpu().numpy())
                all_pred_profit.append(torch.sum(predictions * (pred_d_sol - pred_c_sol),dim=1).cpu().numpy())
                all_true_c_sol.append(true_c_sol.cpu().numpy())
                all_true_d_sol.append(true_d_sol.cpu().numpy())
                all_true_s_sol.append(true_s_sol.cpu().numpy())
                all_true_profit.append(torch.sum(y * (true_d_sol - true_c_sol),dim=1).cpu().numpy())
                all_real_profit.append(torch.sum(y * (pred_d_sol - pred_c_sol),dim=1).cpu().numpy())



        metrics = {"mse": np.average(mse_losses), "mae": np.average(mae_losses), "profit": np.average(profit_losses)}
        results = {
        'pred' : np.concatenate(all_preds, axis=0) if all_preds else np.array([]),
        'true' : np.concatenate(all_trues, axis=0) if all_trues else np.array([]),
        'pred_c_sol' : np.concatenate(all_pred_c_sol, axis=0) if all_pred_c_sol else np.array([]),
        'pred_d_sol' : np.concatenate(all_pred_d_sol, axis=0) if all_pred_d_sol else np.array([]),
        'pred_s_sol' : np.concatenate(all_pred_s_sol, axis=0) if all_pred_s_sol else np.array([]),
        'pred_profit' : np.concatenate(all_pred_profit, axis=0) if all_pred_profit else np.array([]),
        'true_c_sol' : np.concatenate(all_true_c_sol, axis=0) if all_true_c_sol else np.array([]),
        'true_d_sol' : np.concatenate(all_true_d_sol, axis=0) if all_true_d_sol else np.array([]),
        'true_s_sol' : np.concatenate(all_true_s_sol, axis=0) if all_true_s_sol else np.array([]),
        'true_proft' : np.concatenate(all_true_profit, axis=0) if all_true_profit else np.array([]),
        'real_profit' : np.concatenate(all_real_profit, axis=0) if all_real_profit else np.array([])
        }
        
        return metrics, results

    def test(self):
        return self.evaluate(phase="Testing", loader=self.test_loader)
