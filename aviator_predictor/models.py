"""
Machine learning models for Aviator prediction.

Includes Random Forest, XGBoost, Neural Network, and Ensemble models.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import warnings

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    keras = None

warnings.filterwarnings('ignore')


class RandomForestModel:
    """
    Random Forest regression model for Aviator prediction.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        y_pred = self.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }


class XGBoostModel:
    """
    XGBoost regression model for Aviator prediction.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            random_state: Random seed
        """
        if xgb is None:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        y_pred = self.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }


class NeuralNetworkModel:
    """
    Neural Network regression model for Aviator prediction.
    """

    def __init__(self, input_dim: int = 10, epochs: int = 50, batch_size: int = 32):
        """
        Initialize Neural Network model.

        Args:
            input_dim: Input dimension
            epochs: Training epochs
            batch_size: Batch size
        """
        if keras is None:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")

        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.is_trained = False
        self.scaler = StandardScaler()

    def _build_model(self):
        """
        Build neural network architecture.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the Neural Network model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)

        self.model.fit(
            X_train_scaled, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0
        )
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        y_pred = self.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }


class EnsembleModel:
    """
    Ensemble model combining Random Forest, XGBoost, and Neural Network.
    """

    def __init__(self, weights: Dict[str, float] = None, input_dim: int = 10):
        """
        Initialize Ensemble model.

        Args:
            weights: Weights for each model
            input_dim: Input dimension for neural network
        """
        self.rf_model = RandomForestModel()
        self.xgb_model = XGBoostModel() if xgb else None
        self.nn_model = NeuralNetworkModel(input_dim=input_dim) if keras else None

        if weights is None:
            self.weights = {'rf': 0.4, 'xgb': 0.35, 'nn': 0.25}
        else:
            self.weights = weights

        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        self.rf_model.train(X_train, y_train)

        if self.xgb_model:
            self.xgb_model.train(X_train, y_train)

        if self.nn_model:
            self.nn_model.train(X_train, y_train, X_val, y_val)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        predictions = np.zeros(len(X))
        total_weight = 0

        # Random Forest prediction
        rf_pred = self.rf_model.predict(X)
        predictions += rf_pred * self.weights['rf']
        total_weight += self.weights['rf']

        # XGBoost prediction
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X)
            predictions += xgb_pred * self.weights['xgb']
            total_weight += self.weights['xgb']

        # Neural Network prediction
        if self.nn_model:
            nn_pred = self.nn_model.predict(X)
            predictions += nn_pred * self.weights['nn']
            total_weight += self.weights['nn']

        return predictions / total_weight

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ensemble and individual models.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with metrics for each model
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        y_pred = self.predict(X_test)

        return {
            'ensemble': {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            },
            'random_forest': self.rf_model.evaluate(X_test, y_test),
            'xgboost': self.xgb_model.evaluate(X_test, y_test) if self.xgb_model else None,
            'neural_network': self.nn_model.evaluate(X_test, y_test) if self.nn_model else None,
        }
