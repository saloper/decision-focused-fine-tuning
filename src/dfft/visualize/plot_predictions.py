import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .utils import get_latest_experiment_dir, get_experiment_dirs

sns.set(style="whitegrid", font_scale=1.1)

def plot_true_vs_pred_grid(predictions_csv, n_samples=12, random_seed=42, save_path=None, show=True):
    """
    Plots a grid of true vs predicted values for randomly selected samples from predictions.csv.
    Args:
        predictions_csv: Path to predictions.csv
        n_samples: Number of random samples to plot (total subplots)
        random_seed: For reproducibility
        save_path: If provided, saves the figure
        show: If True, displays the plot
    """
    df = pd.read_csv(predictions_csv)
    # Group by split and sample_idx
    grouped = df.groupby(["split", "sample_idx"])
    # Get all unique (split, sample_idx) pairs
    all_keys = list(grouped.groups.keys())
    np.random.seed(random_seed)
    chosen_keys = np.random.choice(len(all_keys), size=min(n_samples, len(all_keys)), replace=False)
    chosen = [all_keys[i] for i in chosen_keys]

    ncols = 4
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, chosen):
        group = grouped.get_group(key)
        true = eval(group.iloc[0]["true"]) if isinstance(group.iloc[0]["true"], str) else group.iloc[0]["true"]
        pred = eval(group.iloc[0]["pred"]) if isinstance(group.iloc[0]["pred"], str) else group.iloc[0]["pred"]
        # If true/pred are lists as strings, convert
        if isinstance(true, str):
            true = eval(true)
        if isinstance(pred, str):
            pred = eval(pred)
        ax.plot(true, label="True", color="tab:blue")
        ax.plot(pred, label="Pred", color="tab:orange", linestyle="--")
        ax.set_title(f"{key[0]} | Sample {key[1]}")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Value")
        ax.legend()
    # Hide unused axes
    for ax in axes[len(chosen):]:
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def example_usage():
    """
    By default, plots the latest experiment's predictions for the given model/dataset.
    Set all_experiments=True to plot all runs for that model/dataset (plots the latest by default).
    """
    model_name = "moment-small-finetuned"
    dataset_name = "NEMA"
    n_samples = 12
    all_experiments = False
    if all_experiments:
        exp_dirs = get_experiment_dirs(model_name, dataset_name)
    else:
        latest = get_latest_experiment_dir(model_name, dataset_name)
        exp_dirs = [latest] if latest else []
    if not exp_dirs:
        print(f"No experiments found for {model_name}/{dataset_name}")
        return
    # By default, plot the latest experiment's predictions.csv
    predictions_csv = os.path.join(exp_dirs[-1], "predictions.csv")
    if not os.path.exists(predictions_csv):
        print(f"No predictions.csv found in {exp_dirs[-1]}")
        return
    plot_true_vs_pred_grid(predictions_csv, n_samples=n_samples)

if __name__ == "__main__":
    example_usage()
