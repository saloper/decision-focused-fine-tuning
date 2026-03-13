import yaml
import argparse
from dfft.utils.files import get_path, get_config
from dfft.data.dataset import NEMADataset

def main():
    parser = argparse.ArgumentParser(description="Train a decision-focused model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="baseline.yaml", 
    )
    args = parser.parse_args()
    config = get_config(args.config)

    print(f"Starting Experiment: {config['experiment_name']}")
    
    train = NEMADataset(
        file = config['paths']['processed_input_path'] / config['paths']['input_filename'],
        start_date = config['data']['train_start_date'],
        end_date = config['data']['train_end_date'],
        input_len = config['data']['input_len'],
        output_len = config['data']['output_len'],
        )

    x,y = train[0]
    print(x.shape, y.shape)

if __name__ == "__main__":
    main()