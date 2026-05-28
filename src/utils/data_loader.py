"""
WineDatasetPipeline: ingestion, merging, domain encoding,
stratified 80/10/10 splitting, and leak-free feature scaling
for the UCI Wine Quality dataset pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class WineSplits:
    """
    Container for all partitioned arrays produced by
    ``WineDatasetPipeline.build()``.

    Attributes
    ----------
    X_train, X_val, X_test : np.ndarray
        Scaled feature matrices for train / validation / test partitions.
    y_train, y_val, y_test : np.ndarray
        Quality score column vectors of shape (N, 1).
    domain_val, domain_test : np.ndarray
        Flat domain-indicator arrays (0 = red, 1 = white) for the
        validation and test partitions, used for per-type MSE reporting.
    scaler : StandardScaler
        The scaler fitted on the training partition only; retained for
        inference on new production samples.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    domain_val: np.ndarray
    domain_test: np.ndarray
    scaler: StandardScaler = field(repr=False)


class WineDatasetPipeline:
    """
    End-to-end data pipeline for the combined UCI Wine Quality dataset.

    Pipeline stages
    ---------------
    1. Load red and white wine CSVs (semicolon-delimited).
    2. Append a binary domain indicator column (0 -> red, 1 -> white).
    3. Concatenate both datasets into a unified feature matrix.
    4. Apply a stratified 80 / 10 / 10 Train / Validation / Test split
       using two sequential ``train_test_split`` calls.
    5. Fit a ``StandardScaler`` exclusively on the training partition
       and apply the frozen transform to validation and test slices,
       eliminating any leakage of test-distribution statistics.

    Parameters
    ----------
    red_csv_path : str
        Filesystem path to ``winequality_red.csv``.
    white_csv_path : str
        Filesystem path to ``winequality_white.csv``.
    val_ratio : float
        Fraction of the full dataset reserved for validation.
        Default: 0.10.
    test_ratio : float
        Fraction of the full dataset reserved for final evaluation.
        Default: 0.10.
    random_state : int
        Seed for all random operations to ensure reproducibility.
        Default: 42.
    """

    def __init__(
        self,
        red_csv_path: str,
        white_csv_path: str,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.red_csv_path = red_csv_path
        self.white_csv_path = white_csv_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state


    # Private helpers

    @staticmethod
    def _load_single(
        csv_path: str,
        wine_type_label: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load one wine CSV, extract features/targets, and append the
        binary domain indicator column.

        Parameters
        ----------
        csv_path : str
            Path to the semicolon-delimited wine quality CSV.
        wine_type_label : int
            0 for red wine, 1 for white wine.

        Returns
        -------
        X : np.ndarray
            Feature matrix with domain indicator appended, shape (N, D+1).
        y : np.ndarray
            Quality score column vector, shape (N, 1).
        domain : np.ndarray
            Flat domain indicator array, shape (N,).
        """
        df = pd.read_csv(csv_path, sep=";")
        X_features: np.ndarray = df.drop("quality", axis=1).values
        y: np.ndarray = df["quality"].values.reshape(-1, 1)
        domain: np.ndarray = np.full(X_features.shape[0], wine_type_label)
        domain_col = domain.reshape(-1, 1)
        X = np.hstack([X_features, domain_col])
        return X, y, domain

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain: np.ndarray,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        """
        Perform the two-stage 80/10/10 stratified split.

        Stage 1 — Carve out the test partition (10%).
        Stage 2 — Carve out the validation partition (10%) from the
                   remaining 90%.

        Returns
        -------
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        domain_val, domain_test
        """
        # Stage 1: isolate the held-out test set.
        (
            X_dev, X_test,
            y_dev, y_test,
            domain_dev, domain_test,
        ) = train_test_split(
            X, y, domain,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )

        # Stage 2: split the dev pool into train and validation.
        # val_fraction_of_dev converts the absolute val_ratio into a
        # fraction relative to the reduced dev pool size.
        val_fraction_of_dev: float = self.val_ratio / (1.0 - self.test_ratio)

        (
            X_train, X_val,
            y_train, y_val,
            _, domain_val,
        ) = train_test_split(
            X_dev, y_dev, domain_dev,
            test_size=val_fraction_of_dev,
            random_state=self.random_state,
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            domain_val, domain_test,
        )


    def build(self) -> WineSplits:
        """
        Execute the full pipeline and return a ``WineSplits`` dataclass.

        Returns
        -------
        WineSplits
            Partitioned, scaled arrays ready for model consumption.
        """
        # 1. Load individual datasets with domain indicators.
        X_red, y_red, domain_red = self._load_single(
            self.red_csv_path, wine_type_label=0
        )
        X_white, y_white, domain_white = self._load_single(
            self.white_csv_path, wine_type_label=1
        )

        # 2. Concatenate into a unified corpus.
        X_all = np.vstack([X_red, X_white])
        y_all = np.vstack([y_red, y_white])
        domain_all = np.concatenate([domain_red, domain_white])

        # 3. Apply 80 / 10 / 10 split.
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            domain_val, domain_test,
        ) = self._split(X_all, y_all, domain_all)

        # 4. Fit scaler on training partition only; transform all slices.
        scaler = StandardScaler()
        X_train_s: np.ndarray = scaler.fit_transform(X_train)
        X_val_s: np.ndarray = scaler.transform(X_val)
        X_test_s: np.ndarray = scaler.transform(X_test)

        return WineSplits(
            X_train=X_train_s,
            X_val=X_val_s,
            X_test=X_test_s,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            domain_val=domain_val,
            domain_test=domain_test,
            scaler=scaler,
        )
