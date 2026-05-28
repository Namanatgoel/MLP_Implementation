# main.py
"""
Execution engine for the Automated Quality Control MLP.

Responsibilities
- Parse CLI hyperparameters.
- Instantiate the WineDatasetPipeline and obtain scaled data splits.
- Construct the MLP from a layer configuration derived from CLI args.
- Run mini-batch gradient descent with per-epoch validation reporting.
- Perform a single final evaluation on the held-out test set.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import mean_squared_error

from src.components.layers import Dense, MSELoss, ReLU
from src.components.model import MLP
from src.utils.data_loader import WineDatasetPipeline, WineSplits


# CLI

def parse_args() -> argparse.Namespace:
    """Define and parse all command-line hyperparameters."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Automated Quality Control — Modular NumPy MLP\n"
            "Predicts wine quality from physicochemical features."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--red_csv",
        type=str,
        required=True,
        help="Path to winequality_red.csv",
    )
    parser.add_argument(
        "--white_csv",
        type=str,
        required=True,
        help="Path to winequality_white.csv",
    )

    # Architecture
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 32],
        metavar="DIM",
        help="Hidden layer neuron counts (space-separated).",
    )

    # Optimisation
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.015,
        help="Gradient-descent step size for all Dense layers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of samples per mini-batch.",
    )

    # Reproducibility
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Global random seed.",
    )

    return parser.parse_args()


# Model factory

def build_model(
    input_dim: int,
    hidden_dims: list[int],
    learning_rate: float,
) -> MLP:
    """
    Construct an MLP from a layer-dimension specification.

    The network topology is:
        Dense(input_dim -> hidden_dims[0]) -> ReLU ->
        Dense(hidden_dims[0] -> hidden_dims[1]) -> ReLU ->
        ... ->
        Dense(hidden_dims[-1] -> 1)   [linear output for regression]

    Parameters
    ----------
    input_dim : int
        Number of input features (derived from the dataset).
    hidden_dims : list[int]
        Neuron counts for each hidden layer.
    learning_rate : float
        Shared learning rate passed to every Dense layer.

    Returns
    -------
    MLP
        Fully instantiated model ready for training.
    """
    layers: list = []
    dims: list[int] = [input_dim] + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(Dense(dims[i], dims[i + 1], learning_rate=learning_rate))
        layers.append(ReLU())

    # Linear output layer — no activation for regression.
    layers.append(Dense(dims[-1], 1, learning_rate=learning_rate))

    return MLP(layers)


# Training loop

def train(
    model: MLP,
    loss_fn: MSELoss,
    splits: WineSplits,
    epochs: int,
    batch_size: int,
) -> dict[str, list[float]]:
    """
    Execute mini-batch gradient descent with per-epoch validation.

    At the start of each epoch the training indices are randomly
    shuffled. The dataset is then iterated in non-overlapping windows
    of size ``batch_size``; a forward pass and backward pass are
    executed for each window, and the Dense layers update their
    parameters in-place during the backward pass.

    After all mini-batches in an epoch are processed, the model is
    evaluated on the validation set without any gradient update.

    Parameters
    ----------
    model : MLP
        Instantiated model.
    loss_fn : MSELoss
        Loss function instance.
    splits : WineSplits
        Data container from the pipeline.
    epochs : int
        Total number of complete passes over the training set.
    batch_size : int
        Number of samples per mini-batch.

    Returns
    -------
    dict[str, list[float]]
        Training history with keys ``"train_loss"`` and ``"val_loss"``.
    """
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    n_train: int = splits.X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Shuffle training indices each epoch.
        indices: np.ndarray = np.random.permutation(n_train)
        epoch_batch_losses: list[float] = []

        # Mini-batch loop.
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start: start + batch_size]
            X_batch: np.ndarray = splits.X_train[batch_idx]
            y_batch: np.ndarray = splits.y_train[batch_idx]

            # Forward pass.
            predictions: np.ndarray = model.forward(X_batch)
            batch_loss: float = loss_fn.forward(predictions, y_batch)
            epoch_batch_losses.append(batch_loss)

            # Backward pass — Dense layers update W and b in-place.
            grad: np.ndarray = loss_fn.backward()
            model.backward(grad)

        # Epoch-level training loss (mean over mini-batches).
        train_loss: float = float(np.mean(epoch_batch_losses))

        # Validation sweep (no gradient update).
        val_predictions: np.ndarray = model.predict(splits.X_val)
        val_loss: float = float(
            mean_squared_error(splits.y_val, val_predictions)
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Progress reporting every 50 epochs and on epoch 1.
        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{epochs}"
                f"  |  Train MSE: {train_loss:.4f}"
                f"  |  Val MSE:   {val_loss:.4f}"
            )

    return history


# Final evaluation

def evaluate_test(
    model: MLP,
    splits: WineSplits,
) -> None:
    """
    Compute and print test-set metrics.

    Executed exactly once after training completes. Per-domain (red /
    white wine) MSE is reported in addition to the aggregate metrics.

    Parameters
    ----------
    model : MLP
        Trained model.
    splits : WineSplits
        Data container holding the held-out test arrays.
    """
    test_preds: np.ndarray = model.predict(splits.X_test)

    overall_mse: float = float(mean_squared_error(splits.y_test, test_preds))
    overall_rmse: float = float(np.sqrt(overall_mse))

    red_mask: np.ndarray = splits.domain_test == 0
    white_mask: np.ndarray = splits.domain_test == 1

    red_mse: float = float(
        mean_squared_error(
            splits.y_test[red_mask], test_preds[red_mask]
        )
    )
    white_mse: float = float(
        mean_squared_error(
            splits.y_test[white_mask], test_preds[white_mask]
        )
    )

    separator: str = "─" * 54
    print(f"\n{separator}")
    print("  Final Test-Set Evaluation")
    print(separator)
    print(f"  Overall  MSE  : {overall_mse:.4f}")
    print(f"  Overall  RMSE : {overall_rmse:.4f}")
    print(f"  Red Wine MSE  : {red_mse:.4f}")
    print(f"  White Wine MSE: {white_mse:.4f}")
    print(separator)


# Entry point

def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    # Data pipeline.
    print("\nLoading and preprocessing dataset ...")
    pipeline = WineDatasetPipeline(
        red_csv_path=args.red_csv,
        white_csv_path=args.white_csv,
        val_ratio=0.10,
        test_ratio=0.10,
        random_state=args.random_state,
    )
    splits: WineSplits = pipeline.build()

    input_dim: int = splits.X_train.shape[1]

    print(f"  Train samples : {splits.X_train.shape[0]}")
    print(f"  Val   samples : {splits.X_val.shape[0]}")
    print(f"  Test  samples : {splits.X_test.shape[0]}")
    print(f"  Input features: {input_dim}")

    # Model construction.
    model: MLP = build_model(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
    )

    layer_dims: list[int] = [input_dim] + args.hidden_dims + [1]
    print(f"\nLayer topology  : {layer_dims}")
    print(f"Learning rate   : {args.learning_rate}")
    print(f"Epochs          : {args.epochs}")
    print(f"Mini-batch size : {args.batch_size}\n")

    # Training.
    loss_fn = MSELoss()
    train(
        model=model,
        loss_fn=loss_fn,
        splits=splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Final evaluation — test set called exactly once.
    evaluate_test(model, splits)


if __name__ == "__main__":
    main()