import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from src.dfft.training.trainer import Trainer
from src.dfft.training.model_registry import get_model

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def create_df(trues, preds, split_name):
    n_samples = trues.shape[0]
    df = pd.DataFrame({
        "split": [split_name] * n_samples,
        "sample_idx": np.arange(n_samples),
        "true": np.squeeze(trues).tolist(),
        "pred": np.squeeze(preds).tolist()
    })
    return df

def run_training_experiment(run_dir, model_name, data_cfg, train_cfg, train_dataset, val_dataset, test_dataset, device):
    logger = logging.getLogger()
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg["batch_size"], shuffle=False)

    logger.info(f"Initializing {model_name}...")
    
    # Extract model-specific parameters from config if any, primarily input/output lengths.
    config_model_kwargs = {
        "input_len": data_cfg["input_len"],
        "output_len": data_cfg["output_len"]
    }
    if "model_params" in data_cfg:
        config_model_kwargs.update(data_cfg["model_params"])
        
    model, trainable = get_model(model_name, config_model_kwargs)
    model.to(device)

    if trainable:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
        total_steps = len(train_loader) * train_cfg["epochs"]
        scheduler = OneCycleLR(optimizer, max_lr=train_cfg["learning_rate"], total_steps=total_steps, pct_start=0.3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            max_norm=5.0
        )

        history = {"train": {"mse": [], "mae": []}, "val": {"mse": [], "mae": []}}
        patience = train_cfg.get("patience", 10)
        best_val_metric = float('inf')
        epochs_no_improve = 0
        best_model_path = os.path.join(run_dir, "best_model.pt")

        logger.info(f"Starting training for {train_cfg['epochs']} epochs...")
        
        for epoch in range(1, train_cfg["epochs"] + 1):
            train_metrics = trainer.train_epoch()
            val_metrics, val_preds, val_trues = trainer.evaluate(phase="Validating")
            
            if scheduler is not None:
                scheduler.step()

            history["train"]["mse"].append(float(train_metrics["mse"]))
            history["train"]["mae"].append(float(train_metrics["mae"]))
            history["val"]["mse"].append(float(val_metrics["mse"]))
            history["val"]["mae"].append(float(val_metrics["mae"]))

            logger.info(f"Epoch {epoch:03d}/{train_cfg['epochs']} | "
                        f"Train MSE: {train_metrics['mse']:.4f} MAE: {train_metrics['mae']:.4f} | "
                        f"Val MSE: {val_metrics['mse']:.4f} MAE: {val_metrics['mae']:.4f}")

            if val_metrics["mse"] < best_val_metric:
                best_val_metric = val_metrics["mse"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"--> Validation MSE improved to {best_val_metric:.4f}. Saved best model.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                    break

        logger.info("Loading best model for testing...")
        model.load_state_dict(torch.load(best_model_path))
        
        train_metrics_final, train_preds, train_trues = trainer.evaluate(phase="Evaluating Train", loader=train_loader)
        val_metrics_final, val_preds, val_trues = trainer.evaluate(phase="Evaluating Val", loader=val_loader)
        test_metrics, test_preds, test_trues = trainer.test()

        logger.info(f"Final Test Metrics | MSE: {test_metrics['mse']:.4f} | MAE: {test_metrics['mae']:.4f}")

        results = {
            "training_history": history,
            "best_val_mse": float(best_val_metric),
            "test_results": {
                 "mse": float(test_metrics["mse"]),
                 "mae": float(test_metrics["mae"])
            }
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saving predictions...")
        train_df = create_df(train_trues, train_preds, "train")
        val_df = create_df(val_trues, val_preds, "val")
        test_df = create_df(test_trues, test_preds, "test")

        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        
        logger.info(f"All outputs saved to: {run_dir}. Experiment complete.")
    else:
        # Not trainable: Only run test set, skip training/val
        logger.info("Model is not trainable. Skipping training/validation. Running test only.")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=None,
            device=device,
            scheduler=None,
            max_norm=5.0
        )
        # Dummy history for compatibility
        history = {"train": {"mse": [], "mae": []}, "val": {"mse": [], "mae": []}}
        best_val_metric = None
        train_metrics_final, train_preds, train_trues = trainer.evaluate(phase="Evaluating Train", loader=train_loader)
        val_metrics_final, val_preds, val_trues = trainer.evaluate(phase="Evaluating Val", loader=val_loader)
        test_metrics, test_preds, test_trues = trainer.test()
        results = {
            "training_history": history,
            "best_val_mse": best_val_metric,
            "test_results": {
                 "mse": float(test_metrics["mse"]),
                 "mae": float(test_metrics["mae"])
            }
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Saving predictions...")
        train_df = create_df(train_trues, train_preds, "train")
        val_df = create_df(val_trues, val_preds, "val")
        test_df = create_df(test_trues, test_preds, "test")
        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        predictions_df.to_parquet(os.path.join(run_dir, "results.parquet"), engine='pyarrow', index=False)
        logger.info(f"All outputs saved to: {run_dir}. Experiment complete.")
