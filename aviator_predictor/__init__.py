"""
Aviator Predictor - Machine learning-powered prediction system for Aviator betting
"""

__version__ = '0.1.0'
__author__ = 'Asher Mohamed'
__email__ = 'ashermohamed546@example.com'

from .predictor import AviatorPredictor
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import RandomForestModel, XGBoostModel, NeuralNetworkModel, EnsembleModel

__all__ = [
    'AviatorPredictor',
    'DataLoader',
    'FeatureEngineer',
    'RandomForestModel',
    'XGBoostModel',
    'NeuralNetworkModel',
    'EnsembleModel',
]
