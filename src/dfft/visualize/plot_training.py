def load_results(results_path):
    with open(results_path, "r") as f:
        return json.load(f)

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_latest_experiment_dir, get_experiment_dirs
import plotly.graph_objects as go
import os

# Set a style for all plots
sns.set(style="whitegrid", font_scale=1.2)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def plot_experiments(experiment_dirs, metrics=["mse", "mae"], save_path=None, show=True):
    """
    Args:
        experiment_dirs: list of experiment run directories (each should contain results.json)
        metrics: tuple/list of metrics to plot (must be keys in results.json['training_history'])
        save_path: if provided, saves the figure to this path
        show: if True, displays the plot
    """
    plt.figure(figsize=(10, 6))
    color_idx = 0
    legend_handles = []

    # Find the max last epoch across all experiments for x-axis placement of test metrics
    max_epoch = 0
    train_lengths = {}
    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "results.json")
        if not os.path.exists(results_path):
            continue
        results = load_results(results_path)
        for metric in metrics:
            train_vals = results.get("training_history", {}).get("train", {}).get(metric, [])
            if len(train_vals) > max_epoch:
                max_epoch = len(train_vals)
            train_lengths[(exp_dir, metric)] = len(train_vals)

    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found. Skipping.")
            continue
        results = load_results(results_path)
        model_path = exp_dir.split('\\')
        dataset_name = model_path[-2]
        model_name = model_path[-3]
        exp_name = dataset_name
        trainable = results.get("trainable", True) if "trainable" in results else True
        for i, metric in enumerate(metrics):
            color = COLORS[color_idx % len(COLORS)]
            if trainable:
                train_vals = results["training_history"]["train"][metric]
                h_train, = plt.plot(train_vals, label=f"{model_name} Train {metric.upper()}", color=color, linestyle="-")
                val_vals = results["training_history"]["val"][metric]
                h_val, = plt.plot(val_vals, label=f"{model_name} - Val {metric.upper()}", color=color, linestyle=":")
                if i == 0:
                    legend_handles.extend([h_train, h_val])
            # Plot test metric as a cross at the end of the longest run
            test_val = results.get("test_results", {}).get(metric, None)
            if test_val is not None:
                plt.scatter([max_epoch-1], [test_val],
                            color=color, marker='x', s=100, label=f"{model_name} Test {metric.upper()}")
            color_idx += 1

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Training Graph for {exp_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

import plotly.graph_objects as go
import plotly.colors as pcolors
def plot_experiments_plotly(experiment_dirs, metrics=["mse"], save_path="training_curves.html"):
    fig = go.Figure()
    metric = metrics[0]  # Plotting the first metric
    
    # Plotly's default qualitative color palette
    colors = pcolors.qualitative.Plotly
    
    # PASS 1: Find the global max epoch for x-axis placement of all scatters
    global_max_epoch = 1
    valid_experiments = []
    
    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found. Skipping.")
            continue
            
        results = load_results(results_path)
        trainable = results.get("trainable", True)
        train_vals = results.get("training_history", {}).get("train", {}).get(metric, [])
        
        if trainable and isinstance(train_vals, list) and len(train_vals) > 0:
            if len(train_vals) > global_max_epoch:
                global_max_epoch = len(train_vals)
                
        valid_experiments.append((exp_dir, results))

    # PASS 2: Plot the data
    for idx, (exp_dir, results) in enumerate(valid_experiments):
        # Adjust path parsing based on your OS separator (Windows '\\', Linux/Mac '/')
        model_name = exp_dir.split('\\')[-3] 
        
        # 3. Ensure consistent color for Train, Val, and Test of the SAME model
        color = colors[idx % len(colors)]
        
        trainable = results.get("trainable", True)
        train_vals = results.get("training_history", {}).get("train", {}).get(metric, [])
        val_vals = results.get("training_history", {}).get("val", {}).get(metric, [])
        test_val = results.get("test_results", {}).get(metric, None)

        # Helper to extract a single scalar if zero-shot passes a list of len 1
        def get_scalar(vals):
            if isinstance(vals, list) and len(vals) > 0:
                return vals[-1]
            elif isinstance(vals, (int, float)):
                return vals
            return None

        # 1. Handle Trainable vs Zero-Shot
        if trainable:
            epochs = list(range(1, len(train_vals) + 1))
            
            # Add Train Line (solid)
            if train_vals:
                fig.add_trace(go.Scatter(x=epochs, y=train_vals, mode='lines', 
                                         name=f"{model_name} (Train)", 
                                         line=dict(dash='solid', color=color)))
            # Add Val Line (dashed)
            if val_vals:
                fig.add_trace(go.Scatter(x=epochs, y=val_vals, mode='lines', 
                                         name=f"{model_name} (Val)", 
                                         line=dict(dash='dash', color=color)))
        else:
            # Zero-shot: Plot train/val as circles at global_max_epoch
            train_scalar = get_scalar(train_vals)
            val_scalar = get_scalar(val_vals)
            
            if train_scalar is not None:
                fig.add_trace(go.Scatter(x=[global_max_epoch], y=[train_scalar], mode='markers',
                                         name=f"{model_name} (Train Zero-Shot)", 
                                         marker=dict(size=10, symbol='circle', color=color)))
            if val_scalar is not None:
                # Using 'circle-open' so it visually differs slightly from train
                fig.add_trace(go.Scatter(x=[global_max_epoch], y=[val_scalar], mode='markers',
                                         name=f"{model_name} (Val Zero-Shot)", 
                                         marker=dict(size=10, symbol='circle-open', color=color, line=dict(width=2))))
        
        # 2. Add Test Point at the global_max_epoch across all models
        if test_val is not None:
            fig.add_trace(go.Scatter(x=[global_max_epoch], y=[test_val], mode='markers',
                                     name=f"{model_name} (Test)", 
                                     marker=dict(size=12, symbol='x', color=color)))

    fig.update_layout(title=f"Training Graph - {metric.upper()}",
                      xaxis_title="Epoch", yaxis_title=f"{metric.upper()} Value",
                      hovermode="x unified")
    
    fig.write_html(save_path)
    fig.show()



def example_usage(model_name=["moment-small-finetuned"], dataset_name="NEMA", all_experiments=False):
    """
    By default, plots the latest experiment for the given model/dataset.
    Set all_experiments=True to plot all runs for that model/dataset.
    """
    if all_experiments:
        experiment_dirs = []
        for m in model_name:
            experiment_dirs.extend(get_experiment_dirs(m, dataset_name))
    else:
        experiment_dirs = []
        for m in model_name:
            latest = get_latest_experiment_dir(m, dataset_name)
            if latest:
                experiment_dirs.append(latest)
    if not experiment_dirs:
        print(f"No experiments found for {model_name}/{dataset_name}")
        return
    plot_experiments_plotly(experiment_dirs, metrics=["mse"], save_path="./results/plots/training_curves_old.html")

if __name__ == "__main__":
    # You can change model_name/dataset_name here or call example_usage from another script
    model_size = ["moment-small", "moment-base", "moment-large", "timesfm-base"]
    model_features = ["-zeroshot", "-finetuned", "-lora"]
    model_list = []
    for size in model_size:
        for feature in model_features:
            model_list.append(size + feature)
    model_list.append("moment-small-time")
    model_list.append("moment-small-time-finetuned")
    example_usage(model_list, all_experiments=False)
