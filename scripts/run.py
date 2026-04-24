import argparse
import sys
import os
from xml.parsers.expat import model
import yaml
import datetime
import shutil
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.dfft.data.dataset import NEMADataset
from src.dfft.training.train import setup_logger, run_training_experiment

# 1. Remove default arguments and argparse from the main function definition
def main(config_path, model_name, dataset_name, version):
    
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    # Create Run Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if train_cfg['decision_loss']:
        run_dir = os.path.join(paths_cfg["output_path"], model_name + "-decision", dataset_name, f"{timestamp}_v{version}")
    else:
        run_dir = os.path.join(paths_cfg["output_path"], model_name, dataset_name, f"{timestamp}_v{version}")
    os.makedirs(run_dir, exist_ok=True)

    # Setup Logging
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    logger.info(f"Starting Experiment: {config.get('experiment_name', 'baseline')}")
    logger.info(f"Run directory created at: {run_dir}")

    # Copy config
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Data
    data_path = os.path.join(paths_cfg["processed_input_path"], paths_cfg["input_filename"])
    
    # Determine if temporal features are needed for this model
    include_time = "time" in model_name.lower() and "moment" in model_name.lower()  
    
    train_dataset = NEMADataset(
        file=data_path,
        start_date=data_cfg["train_start_date"],
        end_date=data_cfg["train_end_date"],
        input_len=data_cfg["input_len"],
        output_len=data_cfg["output_len"],
        include_time=include_time
    )
    
    val_dataset = NEMADataset(
        file=data_path,
        start_date=data_cfg["val_start_date"],
        end_date=data_cfg["val_end_date"],
        input_len=data_cfg["input_len"],
        output_len=data_cfg["output_len"],
        include_time=include_time
    )

    test_dataset = NEMADataset(
        file=data_path,
        start_date=data_cfg["test_start_date"],
        end_date=data_cfg["test_end_date"],
        input_len=data_cfg["input_len"],
        output_len=data_cfg["output_len"],
        include_time=include_time
    )

    run_training_experiment(
        run_dir=run_dir,
        model_name=model_name,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        version=version
    )
    
    # if train_cfg['decision_loss']:
    #     drive_dir = os.path.join('/content/drive/MyDrive/DFFT/', model_name + "-decision", dataset_name, f"{timestamp}_v{version}")
    # else:
    #     drive_dir = os.path.join('/content/drive/MyDrive/DFFT/', model_name, dataset_name, f"{timestamp}_v{version}")
    # # Copy the entire directory tree
    # shutil.copytree(run_dir, drive_dir, dirs_exist_ok=True)
    # print(f"{drive_dir} saved to Drive!")

if __name__ == "__main__":
    # 2. Put argparse here so it only runs once when the script is executed
    parser = argparse.ArgumentParser(description="Train a decision-focused model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Model name and variation")
    parser.add_argument("--dataset", type=str, default="NEMA", help="Dataset name")
    parser.add_argument("--version", type=str, default="2", help="Version name")
    args = parser.parse_args()

    # 3. Logic to determine whether to run one model or the batch loop
    if args.model is not None:
        # If you passed a specific model in the terminal, run JUST that one
        print(f"Running single experiment for model: {args.model}")
        main(args.config, args.model, args.dataset, args.version)
    else:
        # If you didn't pass a model, run your hardcoded batch loop
        model_size = ["moment-small", "moment-base", "moment-large"]
        model_features = ["-zeroshot", "-finetuned", "-lora"]
        
        model_list = [size + feature for size in model_size for feature in model_features]
        
        print(f"Running batch experiments for models: {model_list}")
        for current_model in model_list:
            main(args.config, current_model, args.dataset, args.version)