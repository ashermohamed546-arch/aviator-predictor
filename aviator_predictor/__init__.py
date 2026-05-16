"""
Aviator Predictor - Machine learning-powered prediction system for Aviator betting
"""

__version__ = '0.2.0'
__author__ = 'Asher Mohamed'
__email__ = 'ashermohamed546@example.com'

from .predictor import AviatorPredictor
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import RandomForestModel, XGBoostModel, NeuralNetworkModel, EnsembleModel
from .multi_site_predictor import MultiSiteAviatorPredictor
from .site_scrapers import SiteScraperFactory, BetpawasScraper, BongoBongoScraper

__all__ = [
    'AviatorPredictor',
    'DataLoader',
    'FeatureEngineer',
    'RandomForestModel',
    'XGBoostModel',
    'NeuralNetworkModel',
    'EnsembleModel',
    'MultiSiteAviatorPredictor',
    'SiteScraperFactory',
    'BetpawasScraper',
    'BongoBongoScraper',
]
