from dfft.models.aribtrage import BatteryArbitrage
from dfft.utils.files import get_path
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from joblib import Parallel, delayed
import ast

#Make sure multi-process doesn't cross threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def extract_time(d):
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(d))
    return m.group(1) if m else ""

def get_lastest_exp(base_dir):
    exp_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    exp_dirs = sorted(exp_dirs, key=extract_time)
    return exp_dirs[-1]

def solve_chunk(prices_chunk):
    local_bess = BatteryArbitrage()
    
    chunk_results = []
    for prices in prices_chunk:
        chunk_results.append(local_bess.solve(prices, warm_start=True))
        
    return chunk_results

def solve_dataset(df, num_cores=5, price_key='pred'):
    #Chunk dataset
    chunks = np.array_split(df[price_key].values, num_cores)
    #Multi-process
    results = Parallel(n_jobs=num_cores)(
        delayed(solve_chunk)(chunk) for chunk in chunks
    )
    #Unpack results
    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flat_results).add_prefix(f'{price_key}_')
    results_df.index = df.index
    return df.join(results_df)

def calc_optimization_metrics(model_names=["moment-small", "moment-base", "moment-large"], model_configs=["", "-finetuned", "-lora"]):
    #Define optimization model
    bess = BatteryArbitrage()
    summary = []

    #Loop through models
    for m in model_names:
        for c in model_configs:
            model = m + c
            #Get latest experiment
            base_dir = get_path(f"results\\experiments\\{model}\\NEMA")
            exp_dir = get_lastest_exp(base_dir)

            #Read the results
            csv_file = Path(exp_dir) / "predictions.csv"
            parquet_file = Path(exp_dir) / "results.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file, engine='pyarrow')
            elif csv_file.exists():
                df = pd.read_csv(csv_file, converters={'true' : ast.literal_eval, 'pred': ast.literal_eval})

            #Calculate Ground Truth Metrics
            true_results = [bess.solve(prices) for prices in df['true']]
            true_df = pd.DataFrame(true_results).add_prefix('true_')
            true_df.index = df.index
            df = df.join(true_df)

            #Calculate results for prediction
            pred_results = [bess.solve(prices) for prices in df['pred']]
            pred_df = pd.DataFrame(pred_results).add_prefix('pred_')
            pred_df.index = df.index
            df = df.join(pred_df)

            #Calculate Realized Profit
            true_arr = np.vstack(df['true'])
            discharge_arr = np.vstack(df['pred_discharge_schedule'])
            charge_arr = np.vstack(df['pred_charge_schedule'])
            df['real_profit'] = np.sum(true_arr * (discharge_arr - charge_arr), axis=1)

            #Write un-aggregated file
            path = get_path(f"results\\optimization\\{model}\\NEMA\\" + os.path.basename(exp_dir) + "\\opt_results.parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, engine='pyarrow')

            #Summarize Metrics
            summary_df = df.groupby('split')[['true_profit', 'real_profit']].sum().reset_index()
            summary_df['model_name'] = f'{model}'
            summary.append(summary_df)
    
    final_summary = pd.concat(summary, ignore_index=True)
    return final_summary[['model_name', 'split', 'true_profit', 'real_profit']]



if __name__ == "__main__":
    df = calc_optimization_metrics()
    path = get_path(f"results\\optimization\\summaries\\NEMA_BA.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)