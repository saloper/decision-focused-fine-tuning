import argparse
import sys
import os
from xml.parsers.expat import model
import yaml
import datetime
import shutil

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.dfft.data.dataset import NEMADataset
from src.dfft.training.train import setup_logger, run_training_experiment

def main(config_path="configs/baseline.yaml", model_name="moment-base-frozen", dataset_name="NEMA", version="version1"):
    parser = argparse.ArgumentParser(description="Train a decision-focused model")
    parser.add_argument("--config", type=str, default=config_path, help="Path to config file")
    parser.add_argument("--model", type=str, default=model_name, help="Model name and variation")
    parser.add_argument("--dataset", type=str, default=dataset_name, help="Dataset name")
    parser.add_argument("--version", type=str, default=version, help="Version name")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    # Create Run Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if train_cfg['decision_loss']:
        run_dir = os.path.join(paths_cfg["output_path"], args.model + "-decision", args.dataset, f"{timestamp}_v{args.version}_lr{train_cfg['learning_rate']}_bs{data_cfg['batch_size']}")
    else:
        run_dir = os.path.join(paths_cfg["output_path"], args.model, args.dataset, f"{timestamp}_v{args.version}_lr{train_cfg['learning_rate']}_bs{data_cfg['batch_size']}")
    os.makedirs(run_dir, exist_ok=True)

    # Setup Logging
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    logger.info(f"Starting Experiment: {config.get('experiment_name', 'baseline')}")
    logger.info(f"Run directory created at: {run_dir}")

    # Copy config
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Data
    data_path = os.path.join(paths_cfg["processed_input_path"], paths_cfg["input_filename"])
    
    # Determine if temporal features are needed for this model
    include_time = "time" in args.model.lower() and "moment" in args.model.lower()  # Only MOMENTTime models include temporal features
    
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
        model_name=args.model,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device
    )

if __name__ == "__main__":
    dataset = "NEMA"
    # config = "configs/baseline.yaml"
    config = "configs/decision.yaml"
    model_size = ["moment-small"] #, "moment-base", "moment-large"]
    # model_size = ["timesfm-base"]
    model_features = [ "-zeroshot", "-finetuned","-lora"]
    model_list = []
    for size in model_size:
        for feature in model_features:
            model_list.append(size + feature)
    print("Running experiments for models:", model_list)
    for model in model_list:
        main(config, model, dataset, "1")
