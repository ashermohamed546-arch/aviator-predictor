"""
Feature engineering module for Aviator predictor

Extracts and engineers features from raw game data.
"""

import pandas as pd
import numpy as np
from typing import Union


class FeatureEngineer:
    """
    Extracts and engineers features from game data.
    """

    def __init__(self, lookback_period: int = 10):
        """
        Initialize FeatureEngineer.

        Args:
            lookback_period: Number of previous games to consider
        """
        self.lookback_period = lookback_period

    def engineer_features(self, data: Union[pd.DataFrame, dict]) -> pd.DataFrame:
        """
        Engineer features from raw game data.

        Args:
            data: DataFrame or dict with game data

        Returns:
            DataFrame with engineered features
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        else:
            data = data.copy()

        # Basic features
        features = pd.DataFrame()
        features['players_count'] = data.get('players_count', 0)
        features['total_bet'] = data.get('total_bet', 0)
        features['avg_bet'] = features['total_bet'] / (features['players_count'] + 1)

        # Rolling statistics (if multiple rows)
        if len(data) > 1:
            features['crash_multiplier_mean'] = data['crash_multiplier'].rolling(self.lookback_period, min_periods=1).mean()
            features['crash_multiplier_std'] = data['crash_multiplier'].rolling(self.lookback_period, min_periods=1).std().fillna(0)
            features['crash_multiplier_min'] = data['crash_multiplier'].rolling(self.lookback_period, min_periods=1).min()
            features['crash_multiplier_max'] = data['crash_multiplier'].rolling(self.lookback_period, min_periods=1).max()

            # Lag features
            features['crash_multiplier_lag1'] = data['crash_multiplier'].shift(1).fillna(0)
            features['crash_multiplier_lag2'] = data['crash_multiplier'].shift(2).fillna(0)
            features['crash_multiplier_lag3'] = data['crash_multiplier'].shift(3).fillna(0)

            # Momentum
            features['momentum'] = (data['crash_multiplier'] - data['crash_multiplier'].shift(1)).fillna(0)

            # Trend
            features['trend'] = data['crash_multiplier'].rolling(self.lookback_period, min_periods=1).apply(
                self._calculate_trend, raw=False
            )
        else:
            # Single sample - fill with zeros
            features['crash_multiplier_mean'] = 0
            features['crash_multiplier_std'] = 0
            features['crash_multiplier_min'] = 0
            features['crash_multiplier_max'] = 0
            features['crash_multiplier_lag1'] = 0
            features['crash_multiplier_lag2'] = 0
            features['crash_multiplier_lag3'] = 0
            features['momentum'] = 0
            features['trend'] = 0

        # Fill any remaining NaN
        features = features.fillna(0)

        return features

    @staticmethod
    def _calculate_trend(values: np.ndarray) -> float:
        """
        Calculate trend using simple linear regression.

        Args:
            values: Array of values

        Returns:
            Trend coefficient
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    def normalize_features(self, features: pd.DataFrame, mean: dict = None, std: dict = None) -> tuple:
        """
        Normalize features.

        Args:
            features: Feature DataFrame
            mean: Mean values per feature (calculated if None)
            std: Std values per feature (calculated if None)

        Returns:
            Tuple of (normalized features, mean dict, std dict)
        """
        if mean is None:
            mean = features.mean().to_dict()
        if std is None:
            std = features.std().to_dict()

        features_norm = features.copy()
        for col in features.columns:
            std_val = std.get(col, 1)
            if std_val == 0:
                std_val = 1
            features_norm[col] = (features[col] - mean.get(col, 0)) / std_val

        return features_norm, mean, std
