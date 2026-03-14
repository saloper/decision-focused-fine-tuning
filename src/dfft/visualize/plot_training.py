def load_results(results_path):
    with open(results_path, "r") as f:
        return json.load(f)

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_latest_experiment_dir, get_experiment_dirs

# Set a style for all plots
sns.set(style="whitegrid", font_scale=1.2)

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def plot_experiments(experiment_dirs, metrics=("mse", "mae"), save_path=None, show=True):
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

    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found. Skipping.")
            continue
        results = load_results(results_path)
        model_path = exp_dir.split('\\')
        print(model_path)
        dataset_name = model_path[-2]
        model_name = model_path[-3]
        exp_name = dataset_name + " " + model_name  # Use last two parts of path for name
        for i, metric in enumerate(metrics):
            color = COLORS[color_idx % len(COLORS)]
            # Plot train
            train_vals = results["training_history"]["train"][metric]
            h_train, = plt.plot(train_vals, label=f"Train {metric.upper()}", color=color, linestyle="-")
            # Plot val
            val_vals = results["training_history"]["val"][metric]
            h_val, = plt.plot(val_vals, label=f"Val {metric.upper()}", color=color, linestyle=":")
            if i == 0:
                legend_handles.extend([h_train, h_val])
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



def example_usage(model_name="moment-small-finetuned", dataset_name="NEMA", all_experiments=False):
    """
    By default, plots the latest experiment for the given model/dataset.
    Set all_experiments=True to plot all runs for that model/dataset.
    """
    if all_experiments:
        experiment_dirs = get_experiment_dirs(model_name, dataset_name)
    else:
        latest = get_latest_experiment_dir(model_name, dataset_name)
        experiment_dirs = [latest] if latest else []
    if not experiment_dirs:
        print(f"No experiments found for {model_name}/{dataset_name}")
        return
    plot_experiments(experiment_dirs, metrics=("mse", "mae"))

if __name__ == "__main__":
    # You can change model_name/dataset_name here or call example_usage from another script
    example_usage("moment-small-lora")
