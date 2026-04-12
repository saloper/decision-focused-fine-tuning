
from utils import get_latest_experiment_dir
import os
import pandas as pd

import sys

def find_best_epoch_metrics(metrics_csv_path):
    """
    Reads a metrics CSV file (with columns: epoch, train_metric, val_metric, test_metric) and returns
    the train, val, and test metrics for the epoch with the best (lowest) val_metric.
    """
    df = pd.read_csv(metrics_csv_path)
    best_idx = df['val_metric'].idxmin()
    best_row = df.loc[best_idx]
    return {
        'train': best_row['train_metric'],
        'val': best_row['val_metric'],
        'test': best_row['test_metric']
    }


def collect_latest_results(models, dataset, experiments_root):
    """
    For each model, finds the latest experiment run for the given dataset, and collects metrics from results.json or results.parquet.
    Returns a DataFrame with columns: model, run, train, val, test
    """
    records = []
    for model in models:
        latest_dir = get_latest_experiment_dir(model, dataset, experiments_root)
        if latest_dir is None:
            continue
        files = os.listdir(latest_dir)
        metrics = None
        run = os.path.basename(latest_dir)

        # if 'results.parquet' in files:
        #     df = pd.read_parquet(os.path.join(latest_dir, 'results.parquet'))
        #     row = df.iloc[0]
        #     train = row.get('train_metric') or row.get('train')
        #     val = row.get('val_metric') or row.get('val')
        #     test = row.get('test_metric') or row.get('test')
        #     metrics = {'train': train, 'val': val, 'test': test}
        if 'results.json' in files:
            import json
            with open(os.path.join(latest_dir, 'results.json'), 'r') as f:
                data = json.load(f)
            # New structure: get best val/train from training_history, test from test_results
            train_hist = data.get('training_history', {}).get('train', {}).get('mse', [])
            val_hist = data.get('training_history', {}).get('val', {}).get('mse', [])
            test_mse = data.get('test_results', {}).get('mse', None)
            if val_hist and train_hist:
                best_epoch = min(range(len(val_hist)), key=lambda i: val_hist[i])
                best_val = val_hist[best_epoch]
                best_train = train_hist[best_epoch]
            else:
                best_val = None
                best_train = None
            metrics = {'train': best_train, 'val': best_val, 'test': test_mse}
            print("LALA")
        print(f"Model: {model}, Run: {run}, Metrics: {metrics}")
        if metrics and all(v is not None for v in metrics.values()):
            records.append({
                'model': model,
                'run': run,
                'train': metrics['train'],
                'val': metrics['val'],
                'test': metrics['test']
            })
    
    return pd.DataFrame(records).drop(columns=['run'])

if __name__ == "__main__":
    # List of model names to check (from your experiments folder)
    model_size = ["moment-small", "moment-base", "moment-large", "timesfm-base"]
    model_features = ["-zeroshot", "-finetuned", "-lora"]
    model_list = []
    for size in model_size:
        for feature in model_features:
            model_list.append(size + feature)
    model_list.append("moment-small-time")
    model_list.append("moment-small-time-finetuned")
    dataset = 'NEMA'
    experiments_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../results/experiments'))
    df = collect_latest_results(model_list, dataset, experiments_root)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.sort_values('test'))
