import yaml
import argparse
from src.utils.paths import get_path
from src.data.dataset import NEMADataset

def load_config(filename):
    config_path = get_path(f"configs/{filename}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
        
    return config

def main():
    parser = argparse.ArgumentParser(description="Train a decision-focused model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="baseline.yaml", 
    )
    args = parser.parse_args()
    config = load_config(args.config)

    print(f"Starting Experiment: {config['experiment_name']}")
    
    train = NEMADataset(
        file = get_path(config['paths']['processed_input_path']) / config['paths']['input_filename'],
        start_date = config['data']['train_start_date'],
        end_date = config['data']['train_end_date'],
        input_len = config['data']['input_len'],
        output_len = config['data']['output_len'],
        )

    x,y = train[0]
    print(x.shape, y.shape)

if __name__ == "__main__":
    main()