"""
PyTorch MLP for binary income classification on the Adult Income dataset.

Architecture
------------
Input → Dense(128) → ReLU → Dense(64) → ReLU → Dense(1)

The output is a single raw logit consumed by BCEWithLogitsLoss, which
fuses sigmoid + binary cross-entropy in a numerically stable form:
    loss = -[y * log(σ(z)) + (1 - y) * log(1 - σ(z))]

Evaluation exposes accuracy, precision, recall, F1-score, and a
confusion matrix to account for class imbalance in the dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def build_mlp(input_dim: int) -> nn.Sequential:
    """
    Construct the MLP with two hidden layers and a linear output logit.

    Parameters
    ----------
    input_dim : int
        Number of input features after preprocessing (numerical +
        one-hot encoded categorical).

    Returns
    -------
    nn.Sequential
        Instantiated, untrained PyTorch model.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),   # single raw logit for BCEWithLogitsLoss
    )


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train the model using Adam optimiser and BCEWithLogitsLoss.

    Mini-batch indices are randomly permuted at the start of each epoch
    to decorrelate gradient estimates across iterations.

    Parameters
    ----------
    model      : nn.Module       Instantiated MLP.
    X_train    : torch.Tensor    Float32 feature tensor (N, D).
    y_train    : torch.Tensor    Float32 binary label tensor (N,).
    epochs     : int             Total training epochs. Default: 20.
    batch_size : int             Samples per mini-batch. Default: 256.
    lr         : float           Adam learning rate. Default: 1e-3.

    Returns
    -------
    list[float]
        Per-epoch mean training loss values.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n_samples: int = X_train.shape[0]
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation: torch.Tensor = torch.randperm(n_samples)
        batch_losses: list[float] = []

        for start in range(0, n_samples, batch_size):
            indices = permutation[start : start + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            logits: torch.Tensor = model(X_batch).squeeze()
            loss: torch.Tensor = loss_fn(logits, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())

        mean_loss = float(sum(batch_losses) / len(batch_losses))
        epoch_losses.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  |  Loss: {mean_loss:.4f}")

    return epoch_losses


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict[str, object]:
    """
    Run inference and compute classification metrics.

    Logits are converted to probabilities via sigmoid; threshold 0.5
    is applied to produce binary predictions.

    Parameters
    ----------
    model  : nn.Module       Trained model.
    X_test : torch.Tensor    Float32 feature tensor (N, D).
    y_test : torch.Tensor    Float32 binary label tensor (N,).

    Returns
    -------
    dict with keys:
        accuracy   : float
        precision  : float
        recall     : float
        f1         : float
        confusion_matrix : np.ndarray  shape (2, 2)
    """
    model.eval()
    with torch.no_grad():
        logits: torch.Tensor = model(X_test).squeeze()
        probs: torch.Tensor = torch.sigmoid(logits)
        preds: torch.Tensor = (probs >= 0.5).int()

    y_true = y_test.numpy().astype(int)
    y_pred = preds.numpy()

    accuracy: float = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
