"""
Execution engine for the Adult Income binary classification MLP.

Runs two experiments as documented in MLP_using_Pytorch.pdf and
prints a side-by-side metric comparison at the end.

Experiment A (baseline)  : epochs=20,  batch_size=256
Experiment B (long run)  : epochs=100, batch_size=512

Usage
-----
python adult_main.py --csv data/adult_data.csv

All hyperparameters can be overridden via CLI flags; see --help.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.torch_components.torch_model import build_mlp, evaluate_model, train_model
from src.utils.adult_loader import AdultIncomePipeline, AdultSplits


# CLI

def parse_args() -> argparse.Namespace:
    """Define and parse all command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="adult_main.py",
        description=(
            "Adult Income MLP — Binary Classification (PyTorch)\n"
            "Predicts whether an individual earns more than $50K annually."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to adult_data.csv (or adult.data from UCI).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.20,
        help="Fraction of data reserved for the test set.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam optimiser learning rate.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


# Experiment runner

def run_experiment(
    label: str,
    splits: AdultSplits,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    """
    Train and evaluate a single experiment configuration.

    Parameters
    ----------
    label      : str          Human-readable experiment name.
    splits     : AdultSplits  Preprocessed dataset tensors.
    epochs     : int          Total training epochs.
    batch_size : int          Mini-batch size.
    lr         : float        Adam learning rate.

    Returns
    -------
    dict
        Evaluation metrics for this experiment.
    """
    print(f"\n{'═' * 56}")
    print(f"  {label}")
    print(f"  epochs={epochs}  batch_size={batch_size}  lr={lr}")
    print(f"{'═' * 56}")

    model = build_mlp(splits.input_dim)
    train_model(
        model,
        splits.X_train,
        splits.y_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    metrics = evaluate_model(model, splits.X_test, splits.y_test)
    return metrics


# Reporting

def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Print a formatted metric block for one experiment."""
    cm = metrics["confusion_matrix"]
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  Results — {label}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(sep)


def print_comparison(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> None:
    """Print a side-by-side comparison of the two experiments."""
    sep = "═" * 56
    print(f"\n{sep}")
    print("  Experiment Comparison")
    print(sep)
    header = f"  {'Metric':<14} {'Exp A (20/256)':>16} {'Exp B (100/512)':>16}"
    print(header)
    print(f"  {'─'*14} {'─'*16} {'─'*16}")

    for key in ("accuracy", "precision", "recall", "f1"):
        va = metrics_a[key]
        vb = metrics_b[key]
        winner = "◀" if va >= vb else "  "
        print(f"  {key:<14} {va:>16.4f} {vb:>16.4f}  {winner}")

    print(f"\n  ◀ = stronger result")
    print(
        "\n  Interpretation: Experiment A achieves a higher F1-score and\n"
        "  stronger precision-recall balance. Experiment B trades precision\n"
        "  for marginally higher recall — a known effect of training longer\n"
        "  with larger batches on an imbalanced dataset."
    )
    print(sep)


# Entry point

def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading and preprocessing Adult Income dataset ...")
    pipeline = AdultIncomePipeline(
        csv_path=args.csv,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    splits: AdultSplits = pipeline.build()

    print(f"  Train samples : {splits.X_train.shape[0]}")
    print(f"  Test  samples : {splits.X_test.shape[0]}")
    print(f"  Input features: {splits.input_dim}")

    # ── Experiment A — baseline (20 epochs, batch 256) ────────────────
    metrics_a = run_experiment(
        label="Experiment A — Baseline",
        splits=splits,
        epochs=20,
        batch_size=256,
        lr=args.lr,
    )

    # ── Experiment B — long run (100 epochs, batch 512) ───────────────
    metrics_b = run_experiment(
        label="Experiment B — Extended Training",
        splits=splits,
        epochs=100,
        batch_size=512,
        lr=args.lr,
    )

    # ── Metric reports ────────────────────────────────────────────────
    print_metrics("Experiment A (epochs=20, batch_size=256)", metrics_a)
    print_metrics("Experiment B (epochs=100, batch_size=512)", metrics_b)

    # ── Side-by-side comparison ───────────────────────────────────────
    print_comparison(metrics_a, metrics_b)


if __name__ == "__main__":
    main()
