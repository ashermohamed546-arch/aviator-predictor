"""
Multi-site Aviator predictor for Betpawa, Bongo Bongo UG, and other platforms.

Handles scraping, data collection, and unified predictions across multiple sites.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import json

from .predictor import AviatorPredictor
from .site_scrapers import SiteScraperFactory, SiteScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSiteAviatorPredictor:
    """
    Multi-site Aviator predictor supporting Betpawa, Bongo Bongo UG, and others.
    """

    def __init__(self, sites: List[str] = None, max_workers: int = 4):
        """
        Initialize multi-site predictor.

        Args:
            sites: List of sites to use ('betpawa', 'bongo_bongo_ug', '1xbet', 'bet365')
            max_workers: Maximum concurrent threads for scraping
        """
        self.sites = sites or ['betpawa', 'bongo_bongo_ug']
        self.max_workers = max_workers
        self.scrapers: Dict[str, SiteScraper] = {}
        self.predictors: Dict[str, AviatorPredictor] = {}
        self.game_data: Dict[str, List[Dict]] = {}
        self.combined_data = None

        # Initialize scrapers for requested sites
        self._initialize_scrapers()
        # Initialize predictors for each site
        self._initialize_predictors()

    def _initialize_scrapers(self) -> None:
        """
        Initialize scrapers for all sites.
        """
        for site in self.sites:
            scraper = SiteScraperFactory.create_scraper(site)
            if scraper:
                self.scrapers[site] = scraper
                logger.info(f"Initialized scraper for {site}")
            else:
                logger.warning(f"Failed to initialize scraper for {site}")

    def _initialize_predictors(self) -> None:
        """
        Initialize predictors for each site.
        """
        for site in self.sites:
            self.predictors[site] = AviatorPredictor()
            logger.info(f"Initialized predictor for {site}")

    def fetch_all_game_data(self, limit: int = 100) -> Dict[str, List[Dict]]:
        """
        Fetch game data from all sites concurrently.

        Args:
            limit: Number of games per site

        Returns:
            Dictionary mapping sites to game data
        """
        self.game_data = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for site, scraper in self.scrapers.items():
                future = executor.submit(scraper.fetch_game_history, limit)
                futures[future] = site

            for future in as_completed(futures):
                site = futures[future]
                try:
                    games = future.result()
                    self.game_data[site] = games
                    logger.info(f"Fetched {len(games)} games from {site}")
                except Exception as e:
                    logger.error(f"Error fetching data from {site}: {str(e)}")
                    self.game_data[site] = []

        return self.game_data

    def fetch_live_data_all(self) -> Dict[str, Dict]:
        """
        Fetch live data from all sites.

        Returns:
            Dictionary mapping sites to live game data
        """
        live_data = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for site, scraper in self.scrapers.items():
                future = executor.submit(scraper.fetch_live_data)
                futures[future] = site

            for future in as_completed(futures):
                site = futures[future]
                try:
                    data = future.result()
                    if data:
                        live_data[site] = data
                        logger.info(f"Fetched live data from {site}")
                except Exception as e:
                    logger.error(f"Error fetching live data from {site}: {str(e)}")

        return live_data

    def combine_all_game_data(self) -> pd.DataFrame:
        """
        Combine game data from all sites into a single DataFrame.

        Returns:
            Combined DataFrame with all site data
        """
        all_games = []
        for site, games in self.game_data.items():
            all_games.extend(games)

        if not all_games:
            logger.warning("No game data available to combine")
            return pd.DataFrame()

        self.combined_data = pd.DataFrame(all_games)
        logger.info(f"Combined {len(self.combined_data)} games from all sites")
        return self.combined_data

    def train_all_models(self, combined: bool = True) -> Dict[str, Dict]:
        """
        Train models for all sites.

        Args:
            combined: If True, use combined data for all models

        Returns:
            Dictionary of training metrics per site
        """
        metrics = {}

        if combined:
            # Train on combined data from all sites
            if self.combined_data is None:
                self.combine_all_game_data()

            if self.combined_data is None or len(self.combined_data) == 0:
                logger.error("No combined data available for training")
                return {}

            for site in self.sites:
                try:
                    self.predictors[site].prepare_data(self.combined_data)
                    site_metrics = self.predictors[site].train()
                    metrics[site] = site_metrics
                    logger.info(f"Trained model for {site}: {site_metrics}")
                except Exception as e:
                    logger.error(f"Error training model for {site}: {str(e)}")
                    metrics[site] = {}
        else:
            # Train separate models for each site
            for site, games in self.game_data.items():
                if not games:
                    logger.warning(f"No data for {site}, skipping training")
                    continue

                try:
                    data_df = pd.DataFrame(games)
                    self.predictors[site].prepare_data(data_df)
                    site_metrics = self.predictors[site].train()
                    metrics[site] = site_metrics
                    logger.info(f"Trained model for {site}: {site_metrics}")
                except Exception as e:
                    logger.error(f"Error training model for {site}: {str(e)}")
                    metrics[site] = {}

        return metrics

    def predict_all_sites(self, game_data: Dict) -> Dict[str, Dict]:
        """
        Make predictions for a game across all sites.

        Args:
            game_data: Game data dictionary with features

        Returns:
            Dictionary mapping sites to predictions
        """
        predictions = {}

        for site in self.sites:
            try:
                predictor = self.predictors[site]
                if not predictor.is_trained:
                    logger.warning(f"Predictor for {site} not trained")
                    continue

                features = predictor.engineer_features(game_data)
                prediction = predictor.predict(features)
                prediction['site'] = site
                predictions[site] = prediction
                logger.info(f"Prediction for {site}: {prediction}")
            except Exception as e:
                logger.error(f"Error making prediction for {site}: {str(e)}")
                predictions[site] = {'error': str(e)}

        return predictions

    def get_consensus_prediction(self, game_data: Dict,
                                 threshold: float = 0.65) -> Dict:
        """
        Get consensus prediction across all sites.

        Args:
            game_data: Game data dictionary
            threshold: Confidence threshold for consensus

        Returns:
            Consensus prediction with agreement metrics
        """
        predictions = self.predict_all_sites(game_data)

        if not predictions:
            return {'error': 'No predictions available'}

        # Extract valid predictions
        valid_predictions = [
            p for p in predictions.values()
            if 'prediction' in p and p.get('confidence', 0) >= threshold
        ]

        if not valid_predictions:
            return {'error': 'No high-confidence predictions'}

        # Calculate consensus
        avg_prediction = np.mean([p['prediction'] for p in valid_predictions])
        avg_confidence = np.mean([p['confidence'] for p in valid_predictions])
        std_prediction = np.std([p['prediction'] for p in valid_predictions])

        # Calculate agreement score
        agreement = 1.0 / (1.0 + std_prediction) if std_prediction > 0 else 1.0

        return {
            'consensus_prediction': float(avg_prediction),
            'confidence': float(avg_confidence),
            'agreement': float(agreement),
            'num_sites': len(valid_predictions),
            'std_deviation': float(std_prediction),
            'individual_predictions': predictions,
        }

    def save_game_data(self, filepath: str) -> None:
        """
        Save collected game data to file.

        Args:
            filepath: Path to save CSV file
        """
        if self.combined_data is not None:
            self.combined_data.to_csv(filepath, index=False)
            logger.info(f"Saved game data to {filepath}")
        else:
            logger.warning("No combined data to save")

    def generate_report(self) -> Dict:
        """
        Generate summary report of all models and predictions.

        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'sites': self.sites,
            'total_games': sum(len(games) for games in self.game_data.values()),
            'games_per_site': {site: len(games) for site, games in self.game_data.items()},
            'trained_sites': [site for site in self.sites if self.predictors[site].is_trained],
            'supported_sites': SiteScraperFactory.get_supported_sites(),
        }
        return report

    def update_models_continuous(self, interval_seconds: int = 3600) -> None:
        """
        Continuously update models with new data.

        Args:
            interval_seconds: Update interval in seconds (default: 1 hour)
        """
        import time
        logger.info(f"Starting continuous model updates every {interval_seconds}s")
        try:
            while True:
                logger.info("Fetching new game data...")
                self.fetch_all_game_data(limit=50)
                self.combine_all_game_data()
                metrics = self.train_all_models(combined=True)
                logger.info(f"Models updated with metrics: {metrics}")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Continuous updates stopped")
