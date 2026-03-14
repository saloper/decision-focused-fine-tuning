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
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   └── baseline.yaml          # Configuration files for training and data processing
├── data/
│   └── processed/
│       └── NEMA_LMP.parquet   # Processed datasets 
├── scripts/
│   ├── process_data.py        # Script for data processing pipelines
│   └── train.py               # Main training script
└── src/
    └── dfft/                  # Main package source code
        ├── data/
        │   └── dataset.py     # PyTorch Dataset and DataLoader definitions
        ├── models/
        │   └── moment.py      # Model architectures and forward passes
        └── utils/
            └── files.py       # Helper functions for file management
```
## Project Resouces
* [Overleaf Project](https://www.overleaf.com/project/6998cbc5d26b3d55e09f59fb)

## Relevant Papers
* [MOMENT](https://arxiv.org/abs/2402.03885)
* [OptNet](https://arxiv.org/abs/1703.00443)
* [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)
* [Decision-Focused Learning for Combinatorial Optimization](https://arxiv.org/abs/1809.05504)
* [Task-based End-to-end Model Learning in Stochastic Optimization](https://arxiv.org/abs/1703.04529)
* [A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage](https://arxiv.org/abs/2505.01551)
* [Economics of electric energy storage for energy arbitrage and regulation in New York](https://www.sciencedirect.com/science/article/pii/S0301421506003545)
* [Estimating the value of electricity storage in PJM: Arbitrage and some welfare effects](https://www.sciencedirect.com/science/article/pii/S0140988308001631)
* [Impact of battery degradation on energy arbitrage revenue of grid-level energy storage](https://www.sciencedirect.com/science/article/pii/S2352152X16303231)
