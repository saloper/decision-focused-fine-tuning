## Summary
This repository is dedicated to the exploration of fine-tuning foundation time series models on decsison loss. Specifically, this repository evaluates performance using MOMENT and cvxpylayers to solve a battery arbitrage problem using the ISO-NE NEMA location marginal pricing dataset.

## Project Structure
battery-arbitrage-dfl/
├── configs/                  # YAML files for hyperparameters (model, data, optimizer)
├── data/                     
│   └──raw/                   # Original NEMA ISO-NE data
├── notebooks/                # Jupyter notebooks for EDA, plotting, and quick tests
├── scripts/                  # Executable entry points
│   ├── train.py              # Main training loop script
│   ├── evaluate.py           # Evaluation and backtesting script
│   └── download_data.py      # Script to fetch and prep the NEMA data
├── src/                      
│   ├── data/
│   │   └── dataset.py        # PyTorch Dataset and DataLoader classes
│   ├── models/
│   │   ├── predictive.py     # MOMENT wrappers and other time-series architectures
│   │   └── decision.py       # CVXPY formulations and CvxpyLayer wrappers
│   ├── training/
│   │   ├── losses.py         # Custom decision-focused loss (regret) functions
│   │   └── engine.py         # PyTorch train steps
│   └── utils/
│       └── metrics.py        # Financial metrics (total profit, ROI)
├── .gitignore
├── README.md
└── pyproject.toml            # Dependencies and package metadata

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