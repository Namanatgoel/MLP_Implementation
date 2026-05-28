"""
AdultIncomePipeline: ingestion, preprocessing, and stratified splitting
for the UCI Adult Income dataset.

Target
------
Binary classification: income > $50K (1) or <= $50K (0).

Feature handling
----------------
- Numerical columns  : standardised with StandardScaler.
- Categorical columns : one-hot encoded with OneHotEncoder
                        (unknown categories silently ignored at test time).
- Missing values      : rows containing '?' are dropped before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Column names match the UCI Adult dataset header-less format.
_COLUMN_NAMES: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]


@dataclass
class AdultSplits:
    """
    Container for all partitioned tensors produced by
    ``AdultIncomePipeline.build()``.

    Attributes
    ----------
    X_train, X_test : torch.Tensor  Float32 feature tensors.
    y_train, y_test : torch.Tensor  Float32 binary label tensors.
    input_dim       : int           Number of features after encoding.
    """

    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    input_dim: int


class AdultIncomePipeline:
    """
    End-to-end data pipeline for the UCI Adult Income dataset.

    Pipeline stages
    ---------------
    1. Load the CSV with correct column names; replace '?' with NaN
       and drop incomplete rows.
    2. Binarise the income target: '>50K' → 1, '<=50K' → 0.
    3. Separate numerical and categorical feature columns.
    4. Apply a stratified 80/20 Train/Test split (stratify on target
       to preserve class ratio across splits).
    5. Fit StandardScaler on training numerical columns only.
    6. Fit OneHotEncoder on training categorical columns only.
    7. Concatenate scaled numerical and encoded categorical arrays.
    8. Convert to float32 PyTorch tensors.

    Parameters
    ----------
    csv_path     : str   Path to adult_data.csv (or adult.data).
    test_ratio   : float Fraction reserved for test. Default: 0.20.
    random_state : int   Seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        csv_path: str,
        test_ratio: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.csv_path = csv_path
        self.test_ratio = test_ratio
        self.random_state = random_state


    # Private helpers

    def _load(self) -> pd.DataFrame:
        """Load CSV, assign column names, and drop rows with missing values."""
        df = pd.read_csv(
            self.csv_path,
            names=_COLUMN_NAMES,
            na_values=" ?",
            skipinitialspace=True,
        )
        df = df.dropna().reset_index(drop=True)
        df["income"] = (df["income"].str.strip() == ">50K").astype(int)
        return df

    @staticmethod
    def _split_feature_types(
        df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """Return (numerical_cols, categorical_cols) excluding target."""
        num_cols = (
            df.select_dtypes(include=["int64", "float64"])
            .columns
            .drop("income")
            .tolist()
        )
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        return num_cols, cat_cols


    def build(self) -> AdultSplits:
        """
        Execute the full pipeline and return an ``AdultSplits`` dataclass.

        Returns
        -------
        AdultSplits
            Preprocessed, split, tensor-converted dataset ready for
            PyTorch model consumption.
        """
        df = self._load()
        num_cols, cat_cols = self._split_feature_types(df)

        target: np.ndarray = df["income"].values

        # Stratified 80 / 20 split on raw DataFrame rows.
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=target,
            random_state=self.random_state,
        )

        # ── Fit transformers on training slice only ───────────────────
        scaler = StandardScaler()
        X_train_num: np.ndarray = scaler.fit_transform(
            df_train[num_cols].values
        )
        X_test_num: np.ndarray = scaler.transform(df_test[num_cols].values)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat: np.ndarray = encoder.fit_transform(
            df_train[cat_cols].values
        )
        X_test_cat: np.ndarray = encoder.transform(df_test[cat_cols].values)

        # ── Concatenate numerical + categorical ───────────────────────
        X_train = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
        X_test = np.hstack([X_test_num, X_test_cat]).astype(np.float32)

        y_train = df_train["income"].values.astype(np.float32)
        y_test = df_test["income"].values.astype(np.float32)

        # ── Convert to PyTorch tensors ────────────────────────────────
        return AdultSplits(
            X_train=torch.tensor(X_train),
            X_test=torch.tensor(X_test),
            y_train=torch.tensor(y_train),
            y_test=torch.tensor(y_test),
            input_dim=X_train.shape[1],
        )
