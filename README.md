# Automated Quality Control in Manufacturing (Modular NumPy + PyTorch MLP)

## Business Context

This repository contains two fully independent machine learning experiments
implemented in a shared modular codebase:

| Experiment | Dataset | Framework | Task |
|---|---|---|---|
| Wine Quality MLP | UCI Wine Quality (red + white) | NumPy (from scratch) | Regression — predict quality score |
| Adult Income MLP | UCI Adult Income | PyTorch | Binary classification — income > $50K |

The wine experiment frames the model as a **continuous automated quality
proxy** for beverage manufacturing, reducing sensory testing latency from
hours to sub-second inference per batch.

The adult income experiment demonstrates a production-grade PyTorch binary
classifier with full preprocessing (standardisation + one-hot encoding),
mini-batch Adam training, and two comparative experiment configurations.

---

## Repository Structure

```
├── README.md
├── main.py                         ← Wine Quality entry point (NumPy MLP)
├── adult_main.py                   ← Adult Income entry point (PyTorch MLP)
├── requirements.txt
└── src/
    ├── __init__.py
    ├── components/                 ← NumPy MLP primitives
    │   ├── __init__.py
    │   ├── layers.py               ← Layer, Dense, ReLU, MSELoss
    │   └── model.py                ← MLP orchestrator
    ├── torch_components/           ← PyTorch MLP
    │   ├── __init__.py
    │   └── torch_model.py          ← build_mlp, train_model, evaluate_model
    └── utils/
        ├── __init__.py
        ├── data_loader.py          ← WineDatasetPipeline
        └── adult_loader.py         ← AdultIncomePipeline
```

---


## Architectural Overview

### Object-Oriented Layer Abstraction

Every computational unit in the network inherits from the abstract
`Layer` base class defined in `src/components/layers.py`. This enforces
a strict `forward` / `backward` contract across all primitives and
enables structural iteration in the `MLP` orchestrator without any
layer-specific conditional logic.

| Class | File | Responsibility |
|---|---|---|
| `Layer` | `layers.py` | Abstract base — `forward()` and `backward()` contracts |
| `Dense` | `layers.py` | Affine transform **Z = X @ W + b**; owns W, b; applies He init |
| `ReLU` | `layers.py` | Element-wise activation and its gradient mask |
| `MSELoss` | `layers.py` | Scalar regression loss and analytically correct gradient |
| `MLP` | `model.py` | Chains layers; drives forward and backward passes |
| `WineDatasetPipeline` | `data_loader.py` | Ingestion, merging, splitting, scaling |

### He Normal Weight Initialisation

All `Dense` layers initialise their weight matrices using He normal
initialisation, calibrated specifically for ReLU-activated networks:

```
W ~ N(0, sqrt(2 / fan_in))
b  = 0
```

This prevents vanishing and exploding gradients at network
initialisation by keeping the variance of activations stable across
layers of arbitrary depth.

### Mathematical Correction — MSE Gradient

The original NumPy implementation computed the output gradient as:

```
∂L/∂A = (A − Y) / N          ← Incorrect (factor of 2 omitted)
```

The exact analytical derivative of the MSE loss
`L = (1/N) · Σ (A − Y)²` is:

```
∂L/∂A = 2 · (A − Y) / N     ← Correct (implemented in MSELoss)
```

Omitting the factor of 2 halves the true gradient magnitude, making the
effective learning rate inconsistent with its stated value and distorting
convergence speed. This project restores the correct formulation in
`MSELoss.backward()`.

### Mini-Batch Gradient Descent

Full-batch gradient descent is replaced with configurable mini-batch
gradient descent in `main.py`. At the start of each epoch, training
indices are randomly shuffled. The dataset is then consumed in
non-overlapping windows of `batch_size` samples. Each window triggers a
full forward/backward/update cycle. This provides:

- Stochastic noise that helps escape local minima.
- Reduced memory overhead relative to full-batch computation.
- Configurable granularity via the `--batch_size` CLI argument.

### Data Leakage Mitigation — 80 / 10 / 10 Split

`WineDatasetPipeline.build()` applies a two-stage split:

```
Full dataset  (N samples)
    │
    ├─ 80% → Training set      Parameters updated here only
    ├─ 10% → Validation set    Per-epoch monitoring; no parameter updates
    └─ 10% → Test set          Evaluated exactly once after training ends
```

A `StandardScaler` is **fitted exclusively on the training partition**
and subsequently applied as a frozen transform to the validation and test
slices. Fitting the scaler on the full dataset before splitting would
leak the statistical moments of the test distribution into the model,
producing optimistically biased evaluation metrics.

---


## Dataset Placement

Create a `data/` folder at the project root and place datasets there:

```
mlp_quality_control/
└── data/
    ├── winequality_red.csv       ← UCI Wine Quality (red)
    ├── winequality_white.csv     ← UCI Wine Quality (white)
    └── adult_data.csv            ← UCI Adult Income
```

Download links:
- Wine Quality : https://archive.ics.uci.edu/dataset/186/wine+quality
- Adult Income  : https://archive.ics.uci.edu/dataset/2/adult

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Experiments

### Experiment 1 — Wine Quality (NumPy MLP, Regression)

```bash
python main.py \
    --red_csv   data/winequality_red.csv \
    --white_csv data/winequality_white.csv
```

Full configuration:

```bash
python main.py \
    --red_csv       data/winequality_red.csv \
    --white_csv     data/winequality_white.csv \
    --hidden_dims   128 64 32 \
    --learning_rate 0.015 \
    --epochs        500 \
    --batch_size    64 \
    --random_state  42
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--red_csv` | str | required | Path to winequality_red.csv |
| `--white_csv` | str | required | Path to winequality_white.csv |
| `--hidden_dims` | int+ | 64 32 | Hidden layer sizes |
| `--learning_rate` | float | 0.015 | SGD step size |
| `--epochs` | int | 500 | Training epochs |
| `--batch_size` | int | 64 | Mini-batch size |
| `--random_state` | int | 42 | Random seed |

Expected output:

```
Epoch    1/500  |  Train MSE: 1.4823  |  Val MSE: 1.5102
Epoch   50/500  |  Train MSE: 0.8901  |  Val MSE: 0.9210
...
──────────────────────────────────────────────────────
  Final Test-Set Evaluation
──────────────────────────────────────────────────────
  Overall  MSE  : 0.6526
  Overall  RMSE : 0.8078
  Red Wine MSE  : 0.6528
  White Wine MSE: 0.6525
──────────────────────────────────────────────────────
```

---

### Experiment 2 — Adult Income (PyTorch MLP, Binary Classification)

Runs two configurations automatically and prints a side-by-side comparison:

```bash
python adult_main.py --csv data/adult_data.csv
```

Full configuration:

```bash
python adult_main.py \
    --csv          data/adult_data.csv \
    --test_ratio   0.20 \
    --lr           0.001 \
    --random_state 42
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv` | str | required | Path to adult_data.csv |
| `--test_ratio` | float | 0.20 | Test set fraction |
| `--lr` | float | 0.001 | Adam learning rate |
| `--random_state` | int | 42 | Random seed |

Expected output:

```
════════════════════════════════════════════════════════
  Experiment A — Baseline
  epochs=20  batch_size=256  lr=0.001
════════════════════════════════════════════════════════
  Epoch    1/20  |  Loss: 0.3117
  ...

════════════════════════════════════════════════════════
  Experiment B — Extended Training
  epochs=100  batch_size=512  lr=0.001
════════════════════════════════════════════════════════
  ...

  Experiment Comparison
════════════════════════════════════════════════════════
  Metric         Exp A (20/256)   Exp B (100/512)
  ────────────── ──────────────── ────────────────
  accuracy               0.8513           0.8260  ◀
  precision              0.7325           0.6475  ◀
  recall                 0.6345           0.6605
  f1                     0.6800           0.6539  ◀
```

---

## Architectural Overview

### NumPy MLP (Wine Quality)

| Class | Responsibility |
|---|---|
| `Layer` | Abstract base — `forward()` / `backward()` contract |
| `Dense` | Z = X @ W + b; He init; in-place SGD update |
| `ReLU` | Element-wise activation + gradient mask |
| `MSELoss` | Corrected gradient: 2·(A−Y)/N |
| `MLP` | Sequential chain; drives forward + backward |
| `WineDatasetPipeline` | Load, merge, domain indicator, 80/10/10 split, scaler |

### PyTorch MLP (Adult Income)

| Function/Class | Responsibility |
|---|---|
| `build_mlp` | `nn.Sequential`: Dense(128)→ReLU→Dense(64)→ReLU→Dense(1) |
| `train_model` | Adam + BCEWithLogitsLoss; mini-batch loop |
| `evaluate_model` | Accuracy, precision, recall, F1, confusion matrix |
| `AdultIncomePipeline` | Load, NaN drop, standardise, one-hot encode, 80/20 split |

### Mathematical Correction — MSE Gradient

The original NumPy implementation omitted the factor of 2 from the MSE
gradient. The correct derivative of L = (1/N)·Σ(A−Y)² is:

```
∂L/∂A = 2·(A − Y) / N     ← implemented in MSELoss.backward()
```

---

## Metrics and Acceptance Criteria

| Experiment | Primary Metric | Target |
|---|---|---|
| Wine Quality | Test RMSE | ≤ 0.85 |
| Adult Income | F1-score (high-income class) | ≥ 0.68 (Experiment A baseline) |

# OUTPUT
<img width="1162" height="773" alt="Screenshot from 2026-05-28 19-26-39" src="https://github.com/user-attachments/assets/c06577ca-1861-48f9-8ae3-2afa37605c8f" />

<img width="1163" height="883" alt="Screenshot from 2026-05-28 19-26-54" src="https://github.com/user-attachments/assets/ccdbde66-28ec-4d8a-b1c4-4f27eddcd11e" />

<img width="1725" height="953" alt="Screenshot from 2026-05-28 19-29-40" src="https://github.com/user-attachments/assets/a4ffe19e-62a9-40b7-b876-ab720654cf6b" />

<img width="1644" height="644" alt="Screenshot from 2026-05-28 19-29-56" src="https://github.com/user-attachments/assets/ffd12375-6cab-4409-88ea-8ffd46eb0c79" />

<img width="1558" height="1029" alt="Screenshot from 2026-05-28 19-30-17" src="https://github.com/user-attachments/assets/11ab3fb9-5b54-4f77-96e1-3404fc84f302" />

<img width="1553" height="662" alt="Screenshot from 2026-05-28 19-30-27" src="https://github.com/user-attachments/assets/c4f8cfc2-9e5e-43de-82d9-83be2f273702" />
