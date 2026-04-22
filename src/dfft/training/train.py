import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from dfft.training.trainer import Trainer
from dfft.training.model_registry import get_model

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

def create_df(results_dict, split_name):
    n_samples = results_dict["true"].shape[0]
    
    #Base columns
    df_data = {
        "split": [split_name] * n_samples,
        "sample_idx": np.arange(n_samples),
    }
    
    #Add all arrays from the dictionary
    for key, arr in results_dict.items():
        if arr.size > 0:
            df_data[key] = arr.tolist()
            
    return pd.DataFrame(df_data)

def run_training_experiment(run_dir, model_name, data_cfg, train_cfg, train_dataset, val_dataset, test_dataset, device, version="default"):
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

    # TensorBoard setup
    # Determine a good root for TB logs based on run_dir path structure
    paths_split = run_dir.split(os.sep)
    if "experiments" in paths_split:
        idx = paths_split.index("experiments")
        tb_root = os.path.join(*paths_split[:idx], "tensorboard", version)
        tb_log_dir = os.path.join(tb_root, *paths_split[idx+1:])
    else:
        tb_log_dir = os.path.join(run_dir, "tensorboard", version)
    
    writer = SummaryWriter(log_dir=tb_log_dir)

    hparams_to_log = {
        "model": model_name,
        "version": str(version),
        "batch_size": data_cfg.get("batch_size", 32)
    }
    for k, v in train_cfg.items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparams_to_log[k] = v
        else:
            hparams_to_log[k] = str(v)

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
            max_norm=5.0,
            horizon=data_cfg["output_len"],
            decision_loss=train_cfg["decision_loss"]
        )

        history = {"train": {"mse": [], "mae": [], "profit": []}, "val": {"mse": [], "mae": [], "profit": []}}
        patience = train_cfg.get("patience", 10)
        best_val_metric = float('inf')
        epochs_no_improve = 0
        best_model_path = os.path.join(run_dir, "best_model.pt")

        logger.info(f"Starting training for {train_cfg['epochs']} epochs...")
        
        for epoch in range(1, train_cfg["epochs"] + 1):
            train_metrics = trainer.train_epoch()
            val_metrics, _ = trainer.evaluate(phase="Validating")
            
            if scheduler is not None:
                scheduler.step()

            history["train"]["mse"].append(float(train_metrics["mse"]))
            history["train"]["mae"].append(float(train_metrics["mae"]))
            history["train"]["profit"].append(float(train_metrics["profit"]))
            history["val"]["mse"].append(float(val_metrics["mse"]))
            history["val"]["mae"].append(float(val_metrics["mae"]))
            history["val"]["profit"].append(float(val_metrics["profit"]))

            writer.add_scalar("MSE/Train", train_metrics['mse'], epoch)
            writer.add_scalar("MAE/Train", train_metrics['mae'], epoch)
            writer.add_scalar("Profit/Train", train_metrics['profit'], epoch)
            writer.add_scalar("MSE/Val", val_metrics['mse'], epoch)
            writer.add_scalar("MAE/Val", val_metrics['mae'], epoch)
            writer.add_scalar("Profit/Val", val_metrics['profit'], epoch)

            logger.info(f"Epoch {epoch:03d}/{train_cfg['epochs']} | "
                        f"Train MSE: {train_metrics['mse']:.4f} MAE: {train_metrics['mae']:.4f} | Profit: ${train_metrics['profit']:.2f} "
                        f"Val MSE: {val_metrics['mse']:.4f} MAE: {val_metrics['mae']:.4f} | Profit: ${val_metrics['profit']:.2f}" )

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
        
        train_metrics_final, train_results = trainer.evaluate(phase="Evaluating Train", loader=train_loader)
        val_metrics_final, val_results= trainer.evaluate(phase="Evaluating Val", loader=val_loader)
        test_metrics, test_results = trainer.test()

        logger.info(f"Final Train Metrics | MSE: {train_metrics_final['mse']:.4f} | MAE: {train_metrics_final['mae']:.4f} | Profit: ${train_metrics_final['profit']:.2f}")
        logger.info(f"Final Val Metrics | MSE: {val_metrics_final['mse']:.4f} | MAE: {val_metrics_final['mae']:.4f} | Profit: ${val_metrics_final['profit']:.2f}")
        logger.info(f"Final Test Metrics | MSE: {test_metrics['mse']:.4f} | MAE: {test_metrics['mae']:.4f} | Profit: ${test_metrics['profit']:.2f}")

        writer.add_scalar("MSE/Test", test_metrics["mse"], 0)
        writer.add_scalar("MAE/Test", test_metrics["mae"], 0)
        writer.add_scalar("Profit/Test", test_metrics["profit"], 0)
        
        # Adding to hparams dict correctly
        writer.add_hparams(
            hparam_dict=hparams_to_log,
            metric_dict={
                "hparam/Best_Val_MSE": best_val_metric,
                "hparam/Train_MSE": train_metrics_final["mse"],
                "hparam/Train_MAE": train_metrics_final["mae"],
                "hparam/Train_Profit": train_metrics_final["profit"],
                "hparam/Val_MSE": val_metrics_final["mse"],
                "hparam/Val_MAE": val_metrics_final["mae"],
                "hparam/Val_Profit": val_metrics_final["profit"],
                "hparam/Test_MSE": test_metrics["mse"],
                "hparam/Test_MAE": test_metrics["mae"],
                "hparam/Test_Profit": test_metrics["profit"]
            },
            run_name="."
        )

        results = {
            "training_history": history,
            "best_val_mse": float(best_val_metric),
            "final_train_results": {
                 "mse": float(train_metrics_final["mse"]),
                 "mae": float(train_metrics_final["mae"]),
                 "profit": float(train_metrics_final["profit"])
            },
            "final_val_results": {
                 "mse": float(val_metrics_final["mse"]),
                 "mae": float(val_metrics_final["mae"]),
                 "profit": float(val_metrics_final["profit"])
            },
            "test_results": {
                 "mse": float(test_metrics["mse"]),
                 "mae": float(test_metrics["mae"]),
                 "profit": float(test_metrics["profit"])
            }
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Saving predictions...")
        train_df = create_df(train_results, "train")
        val_df = create_df(val_results, "val")
        test_df = create_df(test_results, "test")
        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        predictions_df.to_parquet(os.path.join(run_dir, "results.parquet"), engine='pyarrow', index=False)
        
        logger.info(f"All outputs saved to: {run_dir}. Experiment complete.")
        writer.close()

        if train_cfg.get("save_weights", False):
            try:
                os.remove(best_model_path)
            except FileNotFoundError:
                logger.warning(f"Best model file not found for cleanup: {best_model_path}")
                pass

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
        history = {"train": {"mse": [], "mae": [], "profit": []}, "val": {"mse": [], "mae": [], "profit": []}}
        best_val_metric = None
        train_metrics_final, train_results = trainer.evaluate(phase="Evaluating Train", loader=train_loader)
        val_metrics_final, val_results= trainer.evaluate(phase="Evaluating Val", loader=val_loader)
        test_metrics, test_results = trainer.test()

        logger.info(f"Final Train Metrics | MSE: {train_metrics_final['mse']:.4f} | MAE: {train_metrics_final['mae']:.4f} | Profit: ${train_metrics_final['profit']:.2f}")
        logger.info(f"Final Val Metrics | MSE: {val_metrics_final['mse']:.4f} | MAE: {val_metrics_final['mae']:.4f} | Profit: ${val_metrics_final['profit']:.2f}")
        logger.info(f"Final Test Metrics | MSE: {test_metrics['mse']:.4f} | MAE: {test_metrics['mae']:.4f} | Profit: ${test_metrics['profit']:.2f}")

        writer.add_scalar("MSE/Test", test_metrics["mse"], train_cfg["epochs"])
        writer.add_scalar("MAE/Test", test_metrics["mae"], train_cfg["epochs"])
        writer.add_scalar("Profit/Test", test_metrics["profit"], train_cfg["epochs"])
        writer.add_scalar("MSE/Train", train_metrics_final["mse"], train_cfg["epochs"])
        writer.add_scalar("MAE/Train", train_metrics_final["mae"], train_cfg["epochs"])
        writer.add_scalar("Profit/Train", train_metrics_final["profit"], train_cfg["epochs"])
        writer.add_scalar("MSE/Val", val_metrics_final["mse"], train_cfg["epochs"])
        writer.add_scalar("MAE/Val", val_metrics_final["mae"], train_cfg["epochs"])
        writer.add_scalar("Profit/Val", val_metrics_final["profit"], train_cfg["epochs"])

        writer.add_hparams(
            hparam_dict=hparams_to_log,
            metric_dict={
                "hparam/Train_MSE": train_metrics_final["mse"],
                "hparam/Train_MAE": train_metrics_final["mae"],
                "hparam/Train_Profit": train_metrics_final["profit"],
                "hparam/Val_MSE": val_metrics_final["mse"],
                "hparam/Val_MAE": val_metrics_final["mae"],
                "hparam/Val_Profit": val_metrics_final["profit"],
                "hparam/Test_MSE": test_metrics["mse"],
                "hparam/Test_MAE": test_metrics["mae"],
                "hparam/Test_Profit": test_metrics["profit"]
            },
            run_name="."
        )

        results = {
            "training_history": history,
            "best_val_mse": best_val_metric,
            "final_train_results": {
                 "mse": float(train_metrics_final["mse"]),
                 "mae": float(train_metrics_final["mae"]),
                 "profit": float(train_metrics_final["profit"])
            },
            "final_val_results": {
                 "mse": float(val_metrics_final["mse"]),
                 "mae": float(val_metrics_final["mae"]),
                 "profit": float(val_metrics_final["profit"])
            },
            "test_results": {
                 "mse": float(test_metrics["mse"]),
                 "mae": float(test_metrics["mae"]),
                 "profit": float(test_metrics["profit"])
            }
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Saving predictions...")
        train_df = create_df(train_results, "train")
        val_df = create_df(val_results, "val")
        test_df = create_df(test_results, "test")
        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
        predictions_df.to_parquet(os.path.join(run_dir, "results.parquet"), engine='pyarrow', index=False)
        logger.info(f"All outputs saved to: {run_dir}. Experiment complete.")
        writer.close()
