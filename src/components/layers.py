"""
Neural network primitives: abstract base, Dense, ReLU, and MSELoss.
All layers expose a unified forward / backward interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Layer(ABC):
    """
    Abstract base class for all neural network layers.

    Every concrete layer must implement a forward pass that transforms
    an input array and a backward pass that propagates an upstream
    gradient and returns the downstream gradient.
    """

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (batch_size, in_features).

        Returns
        -------
        np.ndarray
            Output array of shape (batch_size, out_features).
        """

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass and return the upstream gradient.

        Parameters
        ----------
        output_gradient : np.ndarray
            Gradient of the loss with respect to this layer's output.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to this layer's input.
        """


class Dense(Layer):
    """
    Fully-connected (linear) layer with He normal weight initialisation.

    Forward pass  : Z = X @ W + b
    Backward pass : dX = dZ @ W.T
                    dW = X.T @ dZ
                    db = sum(dZ, axis=0)

    Weight gradients (dW) and bias gradients (db) are stored as
    instance attributes after each backward call so that an external
    optimiser can inspect or apply them.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    out_features : int
        Number of output neurons.
    learning_rate : float
        Step size for the vanilla gradient-descent parameter update.
        Default: 0.015.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.015,
    ) -> None:
        self.learning_rate: float = learning_rate

        # He normal initialisation — calibrated for ReLU activations.
        # W ~ N(0, sqrt(2 / fan_in))
        self.W: np.ndarray = (
            np.random.randn(in_features, out_features)
            * np.sqrt(2.0 / in_features)
        )
        self.b: np.ndarray = np.zeros((1, out_features))

        # Gradient buffers — populated during backward().
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)

        # Cache for backward pass.
        self._X_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Affine transform: Z = X @ W + b."""
        self._X_cache = X
        return X @ self.W + self.b

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Compute parameter and input gradients; apply SGD update.

        Parameters
        ----------
        output_gradient : np.ndarray
            dL/dZ of shape (batch_size, out_features).

        Returns
        -------
        np.ndarray
            dL/dX of shape (batch_size, in_features).
        """
        if self._X_cache is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        self.dW = self._X_cache.T @ output_gradient
        self.db = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient: np.ndarray = output_gradient @ self.W.T

        # Vanilla gradient-descent in-place parameter update.
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db

        return input_gradient


class ReLU(Layer):
    """
    Rectified Linear Unit activation layer.

    Forward  : A = max(0, Z)
    Backward : dZ = dA * 1(Z > 0)

    The pre-activation tensor Z is cached during the forward pass for
    use in the backward mask computation.
    """

    def __init__(self) -> None:
        self._Z_cache: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply element-wise ReLU."""
        self._Z_cache = X
        return np.maximum(0.0, X)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Propagate gradient through the ReLU non-linearity.

        Parameters
        ----------
        output_gradient : np.ndarray
            dL/dA of shape (batch_size, features).

        Returns
        -------
        np.ndarray
            dL/dZ of the same shape; zeroed where Z <= 0.
        """
        if self._Z_cache is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )
        return output_gradient * (self._Z_cache > 0.0).astype(float)


class MSELoss:
    """
    Mean Squared Error loss function for regression targets.

    Loss     : L = (1 / N) * sum((A - Y)^2)
    Gradient : dL/dA = 2 * (A - Y) / N

    Mathematical note
    -----------------
    The original NumPy implementation computed the output gradient as
    ``(A - Y) / N``, omitting the factor of 2 that is the exact
    analytical derivative of the MSE formula above. This class restores
    the mathematically correct gradient, ensuring that the effective
    learning rate is consistent with the stated scalar and that
    convergence behaviour matches theoretical expectations.
    """

    def __init__(self) -> None:
        self._A_cache: Optional[np.ndarray] = None
        self._Y_cache: Optional[np.ndarray] = None

    def forward(self, A: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute scalar MSE loss.

        Parameters
        ----------
        A : np.ndarray
            Model predictions of shape (batch_size, 1).
        Y : np.ndarray
            Ground-truth targets of shape (batch_size, 1).

        Returns
        -------
        float
            Scalar mean squared error.
        """
        self._A_cache = A
        self._Y_cache = Y
        return float(np.mean((A - Y) ** 2))

    def backward(self) -> np.ndarray:
        """
        Compute the analytically correct MSE gradient.

        Returns
        -------
        np.ndarray
            dL/dA = 2 * (A - Y) / N, shape (batch_size, 1).
        """
        if self._A_cache is None or self._Y_cache is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )
        N: int = self._Y_cache.shape[0]
        return 2.0 * (self._A_cache - self._Y_cache) / N
