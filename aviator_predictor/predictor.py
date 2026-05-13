"""
Main Aviator predictor class orchestrating the complete prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import EnsembleModel


class AviatorPredictor:
    """
    Main predictor class for Aviator betting system.

    Combines data loading, feature engineering, and ensemble ML models
    to provide predictions for Aviator game outcomes.
    """

    def __init__(self, rf_n_estimators: int = 100, xgb_n_estimators: int = 100,
                 nn_epochs: int = 50, nn_input_dim: int = 10):
        """
        Initialize Aviator Predictor.

        Args:
            rf_n_estimators: Random Forest trees count
            xgb_n_estimators: XGBoost rounds count
            nn_epochs: Neural Network training epochs
            nn_input_dim: Neural Network input dimension
        """
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer(lookback_period=10)
        self.model = EnsembleModel(input_dim=nn_input_dim)
        self.is_trained = False
        self.feature_mean = None
        self.feature_std = None

    def prepare_data(self, data: Union[str, pd.DataFrame]) -> None:
        """
        Load and prepare data.

        Args:
            data: File path (CSV) or DataFrame
        """
        if isinstance(data, str):
            self.data_loader.load_csv(data)
        else:
            self.data_loader.data = data.copy()
            self.data_loader._validate_data()

    def train(self, val_split: float = 0.2, test_split: float = 0.1,
              random_state: int = 42) -> Dict[str, float]:
        """
        Train the ensemble model.

        Args:
            val_split: Validation set fraction
            test_split: Test set fraction
            random_state: Random seed

        Returns:
            Training metrics
        """
        if self.data_loader.data is None:
            raise ValueError("No data loaded. Call prepare_data first.")

        # Prepare features
        X, y = self.data_loader.prepare_features(self.data_loader.data)

        # Split data
        splits = self.data_loader.split_data(X, y, val_split, test_split)

        # Engineer features
        engineered_features = []
        for features in X:
            feature_dict = {
                'players_count': features[0] if len(features) > 0 else 0,
                'total_bet': features[1] if len(features) > 1 else 0,
            }
            engineered = self.feature_engineer.engineer_features(feature_dict)
            engineered_features.append(engineered.values.flatten())

        X_engineered = np.array(engineered_features)

        # Split engineered features
        X_train = X_engineered[self.data_loader.split_data(X, y, val_split, test_split)['X_train'].astype(bool)]
        X_val = X_engineered[self.data_loader.split_data(X, y, val_split, test_split)['X_val'].astype(bool)]
        X_test = X_engineered[self.data_loader.split_data(X, y, val_split, test_split)['X_test'].astype(bool)]

        # Use simpler split approach
        n_samples = len(X)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_test - n_val

        X_train = X_engineered[:n_train]
        X_val = X_engineered[n_train:n_train + n_val]
        X_test = X_engineered[n_train + n_val:]

        y_train = y[:n_train]
        y_val = y[n_train:n_train + n_val]
        y_test = y[n_train + n_val:]

        # Normalize
        X_train_norm, X_val_norm, X_test_norm = self.data_loader.normalize_features(X_train, X_val, X_test)

        # Train model
        self.model.train(X_train_norm, y_train, X_val_norm, y_val)
        self.is_trained = True

        # Evaluate
        metrics = self.model.evaluate(X_test_norm, y_test)
        return metrics['ensemble']

    def engineer_features(self, data: Union[pd.DataFrame, dict]) -> np.ndarray:
        """
        Engineer features from game data.

        Args:
            data: Game data (DataFrame or dict)

        Returns:
            Engineered features array
        """
        features_df = self.feature_engineer.engineer_features(data)
        return features_df.values

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Make a single prediction.

        Args:
            features: Engineered features

        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        prediction = self.model.predict(features)[0]
        confidence = self._calculate_confidence(prediction)

        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
        }

    def predict_batch(self, features: np.ndarray) -> List[Dict[str, float]]:
        """
        Make batch predictions.

        Args:
            features: Array of engineered features

        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions = self.model.predict(features)
        results = []
        for pred in predictions:
            results.append({
                'prediction': float(pred),
                'confidence': float(self._calculate_confidence(pred)),
            })
        return results

    def evaluate(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        if X_test is None or y_test is None:
            raise ValueError("Test data required for evaluation")

        metrics = self.model.evaluate(X_test, y_test)
        return metrics['ensemble']

    @staticmethod
    def _calculate_confidence(prediction: float, base_confidence: float = 0.65) -> float:
        """
        Calculate confidence score for prediction.

        Args:
            prediction: Model prediction value
            base_confidence: Base confidence level

        Returns:
            Confidence score between 0 and 1
        """
        # Confidence based on prediction magnitude
        confidence = min(abs(prediction) / 5.0, 1.0) * base_confidence + 0.35
        return min(max(confidence, 0.0), 1.0)
