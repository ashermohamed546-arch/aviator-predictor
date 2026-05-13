"""
Unit tests for ML models
"""

import pytest
import numpy as np
from aviator_predictor.models import RandomForestModel, NeuralNetworkModel, EnsembleModel
from aviator_predictor.feature_engineering import FeatureEngineer
from aviator_predictor.data_loader import DataLoader


class TestRandomForestModel:
    """
    Tests for Random Forest model
    """

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test

    def test_initialization(self):
        """Test model initialization"""
        model = RandomForestModel(n_estimators=50)
        assert model.is_trained is False
        assert model.model is not None

    def test_training(self, sample_data):
        """Test model training"""
        X_train, y_train, _, _ = sample_data
        model = RandomForestModel()
        model.train(X_train, y_train)
        assert model.is_trained is True

    def test_prediction(self, sample_data):
        """Test model prediction"""
        X_train, y_train, X_test, _ = sample_data
        model = RandomForestModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (20,)
        assert not np.any(np.isnan(predictions))

    def test_evaluation(self, sample_data):
        """Test model evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        model = RandomForestModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_prediction_without_training(self, sample_data):
        """Test that prediction fails without training"""
        _, _, X_test, _ = sample_data
        model = RandomForestModel()
        with pytest.raises(ValueError):
            model.predict(X_test)


class TestNeuralNetworkModel:
    """
    Tests for Neural Network model
    """

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test

    def test_initialization(self):
        """Test model initialization"""
        model = NeuralNetworkModel(input_dim=10, epochs=10)
        assert model.is_trained is False
        assert model.model is not None

    def test_training(self, sample_data):
        """Test model training"""
        X_train, y_train, _, _ = sample_data
        model = NeuralNetworkModel(input_dim=10, epochs=5)
        model.train(X_train, y_train)
        assert model.is_trained is True

    def test_prediction(self, sample_data):
        """Test model prediction"""
        X_train, y_train, X_test, _ = sample_data
        model = NeuralNetworkModel(input_dim=10, epochs=5)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (20,)
        assert not np.any(np.isnan(predictions))

    def test_evaluation(self, sample_data):
        """Test model evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        model = NeuralNetworkModel(input_dim=10, epochs=5)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics


class TestEnsembleModel:
    """
    Tests for Ensemble model
    """

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_initialization(self):
        """Test ensemble initialization"""
        model = EnsembleModel(input_dim=10)
        assert model.is_trained is False
        assert model.rf_model is not None

    def test_training(self, sample_data):
        """Test ensemble training"""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        model = EnsembleModel(input_dim=10)
        model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained is True

    def test_prediction(self, sample_data):
        """Test ensemble prediction"""
        X_train, y_train, X_val, y_val, X_test, _ = sample_data
        model = EnsembleModel(input_dim=10)
        model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        assert predictions.shape == (20,)
        assert not np.any(np.isnan(predictions))

    def test_evaluation(self, sample_data):
        """Test ensemble evaluation"""
        X_train, y_train, X_val, y_val, X_test, y_test = sample_data
        model = EnsembleModel(input_dim=10)
        model.train(X_train, y_train, X_val, y_val)
        metrics = model.evaluate(X_test, y_test)
        assert 'ensemble' in metrics
        assert 'random_forest' in metrics
        assert all(key in metrics['ensemble'] for key in ['mse', 'rmse', 'mae', 'r2'])


class TestFeatureEngineer:
    """
    Tests for Feature Engineering
    """

    def test_initialization(self):
        """Test feature engineer initialization"""
        engineer = FeatureEngineer(lookback_period=10)
        assert engineer.lookback_period == 10

    def test_engineer_single_sample(self):
        """Test feature engineering for single sample"""
        engineer = FeatureEngineer()
        data = {
            'players_count': 150,
            'total_bet': 5000,
            'crash_multiplier': 2.45
        }
        features = engineer.engineer_features(data)
        assert features.shape[0] == 1
        assert features.shape[1] > 0
        assert not np.any(np.isnan(features))

    def test_normalize_features(self):
        """Test feature normalization"""
        engineer = FeatureEngineer()
        data = {
            'players_count': [100, 200, 150],
            'total_bet': [5000, 10000, 7500],
        }
        features = engineer.engineer_features(data)
        norm_features, mean, std = engineer.normalize_features(features)
        assert norm_features.shape == features.shape
        assert mean is not None
        assert std is not None


class TestDataLoader:
    """
    Tests for Data Loading
    """

    def test_initialization(self):
        """Test data loader initialization"""
        loader = DataLoader()
        assert loader.data is None
        assert loader.features is None
        assert loader.target is None

    def test_split_data(self):
        """Test data splitting"""
        loader = DataLoader()
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        splits = loader.split_data(X, y, val_split=0.2, test_split=0.1)
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        assert len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']) == 100
