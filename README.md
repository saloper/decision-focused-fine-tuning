## Summary
This repository is dedicated to the exploration of fine-tuning foundation time series models on decsison loss. Specifically, this repository evaluates performance using MOMENT and cvxpylayers to solve a battery arbitrage problem using the ISO-NE NEMA location marginal pricing dataset.


## Instructions for Coding

### Environment Setup

1. Clone this repository.
2. Create and activate a conda environment:
    ```bash
    conda create -n dfft python=3.11
    conda activate dfft
    ```
3. Install PyTorch with CUDA support:
    ```bash
    conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
    ```
4. Install all other dependencies and the package itself:
    ```bash
    pip install -e .
    ```

---

## Adding a New Model: What to Change

To add a new model (or a new variant), update the following locations:

1. **`src/dfft/training/model_registry.py`**
    - Add a new `elif` block in `get_model` for your model/variant.
    - Define its default hyperparameters and set the `trainable` flag as needed.
    - Return the correct model instance and trainable flag.

2. **`src/dfft/models/`**
    - Implement your new model class (e.g., `MyNewForecaster`) in a new or existing file.
    - Ensure it follows the same API as `BaseForecaster`.

3. **`configs/baseline.yaml` or your config file**
    - Add any model-specific parameters under the appropriate section.

4. **(Optional) `scripts/run.py`**
    - If you want to expose your model as a command-line option, add it to the parser's choices or documentation.

5. **(Optional) `src/dfft/training/train.py`**
    - If your model requires special handling in the training loop, add logic as needed.

---

## Project Structure
```text
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml             # Project metadata and dependencies
├── setup_env.bat              # Environment setup script
├── configs/
│   ├── baseline.yaml          # Configuration for baseline models
│   └── decision.yaml          # Configuration for decision-focused learning runs
├── data/
│   ├── processed/
│   │   └── NEMA_LMP.parquet   # Processed NEMA locational marginal pricing dataset
│   └── raw/                   # Raw hourly SMD data (2017-2026) for ISO-NE regions
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│   ├── aggregate_metrics.ipynb
│   ├── explore_metrics.ipynb
│   ├── timesfm.ipynb
│   └── plots/                 # Saved evaluation visualizations (e.g., cumulative profit plots)
├── results/                   # Evaluation results and interactive visualizations
│   └── plots/                 # Interactive HTML training curves
├── scripts/
│   ├── process_data.py        # Script for data processing pipelines
│   ├── run.py                 # General execution script
│   └── runexp.py              # Script for running specific experiments
└── src/
    ├── dfft/                  # Main package source code
    │   ├── data/
    │   │   └── dataset.py     # PyTorch Dataset and DataLoader definitions
    │   ├── models/            # Model architectures and formulations
    │   │   ├── aribtrage.py   # Battery arbitrage optimization formulation
    │   │   ├── base.py        # Base model classes
    │   │   ├── moment.py      # MOMENT foundation model implementation
    │   │   ├── momentLora.py  # LoRA fine-tuning wrappers for MOMENT
    │   │   ├── moment_time.py 
    │   │   └── timesfm.py     # TimesFM model implementation
    │   ├── training/          # Training loops and model management
    │   │   ├── model_registry.py 
    │   │   ├── train.py       # Core training logic
    │   │   └── trainer.py     # Trainer class implementation
    │   ├── utils/
    │   │   └── files.py       # Helper functions for file management
    │   └── visualize/         # Code for generating plots and tables
    │       ├── create_table.py
    │       ├── plot_predictions.py
    │       ├── plot_training.py
    │       └── utils.py
    └── results/               # Experiment logs, outputs, and run configurations
        └── experiments/       # Specific run artifacts (e.g., config.yaml, train.log)
```
## Project Resouces
* [Overleaf Project](https://www.overleaf.com/project/6998cbc5d26b3d55e09f59fb)

## Relevant Papers
### Theory
* [MOMENT](https://arxiv.org/abs/2402.03885)
* [OptNet](https://arxiv.org/abs/1703.00443)
* [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)
* [Decision-Focused Learning for Combinatorial Optimization](https://arxiv.org/abs/1809.05504)
### Classic Studies
* [Economics of electric energy storage for energy arbitrage and regulation in New York](https://www.sciencedirect.com/science/article/pii/S0301421506003545)
* [Estimating the value of electricity storage in PJM: Arbitrage and some welfare effects](https://www.sciencedirect.com/science/article/pii/S0140988308001631)
* [Impact of battery degradation on energy arbitrage revenue of grid-level energy storage](https://www.sciencedirect.com/science/article/pii/S2352152X16303231)
* [Task-based End-to-end Model Learning in Stochastic Optimization](https://arxiv.org/abs/1703.04529)
### Closely Related Work
* [Decision-Focused Fine-Tuning of Time Series Foundation Models for Dispatchable Feeder Optimization](https://arxiv.org/abs/2503.01936v1)
* [Perturbed Decision-Focused Learning for Modeling Strategic Energy Storage](https://arxiv.org/abs/2406.17085)
* [Decision-Focused Learning Enhanced by Automated Feature Engineering for Energy Storage Optimisation](https://arxiv.org/abs/2509.05772)
* [A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage](https://arxiv.org/abs/2505.01551)