"""
Data loader module for Aviator predictor system

Handles loading, validation, and preprocessing of game data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """
    Handles loading and preprocessing of Aviator game data.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize DataLoader.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.data = None
        self.features = None
        self.target = None

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file containing game data.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame containing loaded data
        """
        self.data = pd.read_csv(filepath)
        self._validate_data()
        return self.data

    def _validate_data(self) -> None:
        """
        Validate loaded data has required columns.

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['crash_multiplier', 'players_count', 'total_bet']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def prepare_features(self, data: pd.DataFrame, target_col: str = 'crash_multiplier') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from data.

        Args:
            data: Input DataFrame
            target_col: Target column name

        Returns:
            Tuple of (features array, target array)
        """
        # Remove rows with missing values
        data_clean = data.dropna()

        # Separate features and target
        self.target = data_clean[target_col].values
        self.features = data_clean.drop(columns=[target_col]).values

        return self.features, self.target

    def split_data(self, features: np.ndarray, target: np.ndarray,
                   val_split: float = 0.2, test_split: float = 0.1) -> dict:
        """
        Split data into train, validation, and test sets.

        Args:
            features: Feature array
            target: Target array
            val_split: Validation set fraction
            test_split: Test set fraction

        Returns:
            Dictionary with train, val, test splits
        """
        n_samples = len(features)
        indices = np.arange(n_samples)
        np.random.RandomState(self.random_state).shuffle(indices)

        # Calculate split points
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * val_split)
        train_size = n_samples - test_size - val_size

        # Split indices
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        return {
            'X_train': features[train_idx],
            'X_val': features[val_idx],
            'X_test': features[test_idx],
            'y_train': target[train_idx],
            'y_val': target[val_idx],
            'y_test': target[test_idx],
        }

    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using training set statistics.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            Tuple of normalized arrays
        """
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8  # Avoid division by zero

        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        return X_train_norm, X_val_norm, X_test_norm
