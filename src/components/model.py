"""
MLP orchestrator: sequential forward and backward propagation
over an ordered list of Layer objects.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.components.layers import Layer


class MLP:
    """
    Multi-Layer Perceptron orchestrator for regression.

    Accepts an ordered sequence of instantiated ``Layer`` objects and
    drives data forward through each layer in order, then propagates
    error gradients backward through the reversed sequence.

    The class does not own a loss function; loss computation and the
    initial output gradient are the responsibility of the training loop
    in ``main.py``. This separation of concerns keeps the model purely
    structural and loss-agnostic.

    Parameters
    ----------
    layers : List[Layer]
        Ordered list of Layer instances (Dense, ReLU, etc.). The last
        element must be a ``Dense`` layer with ``out_features=1`` for
        scalar regression.

    Example
    -------
    >>> from src.components.layers import Dense, ReLU
    >>> model = MLP([
    ...     Dense(12, 64),
    ...     ReLU(),
    ...     Dense(64, 32),
    ...     ReLU(),
    ...     Dense(32, 1),
    ... ])
    >>> y_hat = model.forward(X_batch)
    >>> model.backward(loss_gradient)
    """

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("layers list must contain at least one Layer.")
        self.layers: List[Layer] = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a sequential forward pass through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (batch_size, in_features).

        Returns
        -------
        np.ndarray
            Model output of shape (batch_size, 1).
        """
        out: np.ndarray = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, output_gradient: np.ndarray) -> None:
        """
        Propagate the loss gradient backward through all layers in
        reverse order. Each ``Dense`` layer updates its own parameters
        internally during this traversal.

        Parameters
        ----------
        output_gradient : np.ndarray
            Initial gradient dL/dA from the loss function,
            shape (batch_size, 1).
        """
        grad: np.ndarray = output_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference without modifying any cached state used by
        backward. Delegates to forward(); provided as a semantically
        explicit alias for evaluation contexts.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, in_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (N, 1).
        """
        return self.forward(X)
