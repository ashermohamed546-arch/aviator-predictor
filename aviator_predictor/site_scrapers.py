"""
Web scrapers for different Aviator betting sites.

Supports: Betpawa, Bongo Bongo UG, 1xBet, Bet365, and more.
"""

import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiteScraper(ABC):
    """
    Abstract base class for Aviator site scrapers.
    """

    def __init__(self, site_name: str, api_url: str, timeout: int = 10):
        """
        Initialize scraper.

        Args:
            site_name: Name of the betting site
            api_url: API endpoint URL
            timeout: Request timeout in seconds
        """
        self.site_name = site_name
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    @abstractmethod
    def fetch_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Fetch game history from the site.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data dictionaries
        """
        pass

    @abstractmethod
    def fetch_live_data(self) -> Dict:
        """
        Fetch live game data.

        Returns:
            Current game data
        """
        pass

    def _make_request(self, endpoint: str, method: str = 'GET',
                      params: Dict = None, data: Dict = None) -> Dict:
        """
        Make HTTP request to API.

        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body

        Returns:
            Response data
        """
        try:
            url = f"{self.api_url}/{endpoint}"
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout)
            else:
                response = self.session.post(url, json=data, timeout=self.timeout)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {self.site_name}: {str(e)}")
            return {}


class BetpawasScraper(SiteScraper):
    """
    Scraper for Betpawa betting site.
    """

    def __init__(self):
        """
        Initialize Betpawa scraper.
        """
        super().__init__(
            site_name='Betpawa',
            api_url='https://www.betpawa.ug/api',
            timeout=10
        )

    def fetch_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Fetch game history from Betpawa.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            # Betpawa API endpoint for game history
            response = self._make_request(
                'v2/game/aviator/history',
                params={'limit': limit, 'offset': 0}
            )

            games = []
            if 'data' in response:
                for game in response['data']:
                    games.append({
                        'site': 'Betpawa',
                        'game_id': game.get('id'),
                        'crash_multiplier': float(game.get('crash_point', 0)),
                        'players_count': game.get('player_count', 0),
                        'total_bet': float(game.get('total_bet_amount', 0)),
                        'timestamp': game.get('created_at'),
                        'game_duration': game.get('duration', 0),
                    })
            return games
        except Exception as e:
            logger.error(f"Error fetching Betpawa history: {str(e)}")
            return []

    def fetch_live_data(self) -> Dict:
        """
        Fetch live game data from Betpawa.

        Returns:
            Current game data
        """
        try:
            response = self._make_request('v2/game/aviator/live')
            if 'data' in response:
                game = response['data']
                return {
                    'site': 'Betpawa',
                    'game_id': game.get('id'),
                    'crash_multiplier': float(game.get('current_multiplier', 1.0)),
                    'players_count': game.get('active_players', 0),
                    'total_bet': float(game.get('total_bets', 0)),
                    'status': game.get('status'),  # 'running' or 'crashed'
                    'timestamp': datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error fetching Betpawa live data: {str(e)}")
            return {}


class BongoBongoScraper(SiteScraper):
    """
    Scraper for Bongo Bongo UG betting site.
    """

    def __init__(self):
        """
        Initialize Bongo Bongo scraper.
        """
        super().__init__(
            site_name='Bongo Bongo UG',
            api_url='https://www.bongobongo.ug/api',
            timeout=10
        )

    def fetch_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Fetch game history from Bongo Bongo UG.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            # Bongo Bongo API endpoint for game history
            response = self._make_request(
                'games/aviator/history',
                params={'limit': limit}
            )

            games = []
            if isinstance(response, list):
                for game in response:
                    games.append({
                        'site': 'Bongo Bongo UG',
                        'game_id': game.get('gameId'),
                        'crash_multiplier': float(game.get('crashAt', 0)),
                        'players_count': game.get('numPlayers', 0),
                        'total_bet': float(game.get('totalBet', 0)),
                        'timestamp': game.get('timestamp'),
                        'game_duration': game.get('duration', 0),
                    })
            return games
        except Exception as e:
            logger.error(f"Error fetching Bongo Bongo history: {str(e)}")
            return []

    def fetch_live_data(self) -> Dict:
        """
        Fetch live game data from Bongo Bongo UG.

        Returns:
            Current game data
        """
        try:
            response = self._make_request('games/aviator/live')
            if isinstance(response, dict):
                return {
                    'site': 'Bongo Bongo UG',
                    'game_id': response.get('gameId'),
                    'crash_multiplier': float(response.get('multiplier', 1.0)),
                    'players_count': response.get('players', 0),
                    'total_bet': float(response.get('totalBets', 0)),
                    'status': response.get('state'),  # 'playing' or 'crashed'
                    'timestamp': datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error fetching Bongo Bongo live data: {str(e)}")
            return {}


class OnexBetScraper(SiteScraper):
    """
    Scraper for 1xBet betting site.
    """

    def __init__(self):
        """
        Initialize 1xBet scraper.
        """
        super().__init__(
            site_name='1xBet',
            api_url='https://api.1xbet.com',
            timeout=10
        )

    def fetch_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Fetch game history from 1xBet.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            response = self._make_request(
                'games/aviator/history',
                params={'count': limit}
            )

            games = []
            if 'games' in response:
                for game in response['games']:
                    games.append({
                        'site': '1xBet',
                        'game_id': game.get('gameId'),
                        'crash_multiplier': float(game.get('multiplier', 0)),
                        'players_count': game.get('playerCount', 0),
                        'total_bet': float(game.get('totalBets', 0)),
                        'timestamp': game.get('createdAt'),
                        'game_duration': game.get('duration', 0),
                    })
            return games
        except Exception as e:
            logger.error(f"Error fetching 1xBet history: {str(e)}")
            return []

    def fetch_live_data(self) -> Dict:
        """
        Fetch live game data from 1xBet.

        Returns:
            Current game data
        """
        try:
            response = self._make_request('games/aviator/current')
            if 'game' in response:
                game = response['game']
                return {
                    'site': '1xBet',
                    'game_id': game.get('gameId'),
                    'crash_multiplier': float(game.get('currentMultiplier', 1.0)),
                    'players_count': game.get('activePlayers', 0),
                    'total_bet': float(game.get('totalBets', 0)),
                    'status': game.get('status'),
                    'timestamp': datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error fetching 1xBet live data: {str(e)}")
            return {}


class Bet365Scraper(SiteScraper):
    """
    Scraper for Bet365 betting site.
    """

    def __init__(self):
        """
        Initialize Bet365 scraper.
        """
        super().__init__(
            site_name='Bet365',
            api_url='https://www.bet365.com/api',
            timeout=10
        )

    def fetch_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Fetch game history from Bet365.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            response = self._make_request(
                'aviator/history',
                params={'limit': limit}
            )

            games = []
            if 'results' in response:
                for game in response['results']:
                    games.append({
                        'site': 'Bet365',
                        'game_id': game.get('id'),
                        'crash_multiplier': float(game.get('result', 0)),
                        'players_count': game.get('players', 0),
                        'total_bet': float(game.get('stake', 0)),
                        'timestamp': game.get('time'),
                        'game_duration': game.get('duration', 0),
                    })
            return games
        except Exception as e:
            logger.error(f"Error fetching Bet365 history: {str(e)}")
            return []

    def fetch_live_data(self) -> Dict:
        """
        Fetch live game data from Bet365.

        Returns:
            Current game data
        """
        try:
            response = self._make_request('aviator/live')
            if 'game' in response:
                game = response['game']
                return {
                    'site': 'Bet365',
                    'game_id': game.get('id'),
                    'crash_multiplier': float(game.get('current', 1.0)),
                    'players_count': game.get('participants', 0),
                    'total_bet': float(game.get('totalStake', 0)),
                    'status': game.get('state'),
                    'timestamp': datetime.now().isoformat(),
                }
            return {}
        except Exception as e:
            logger.error(f"Error fetching Bet365 live data: {str(e)}")
            return {}


class SiteScraperFactory:
    """
    Factory for creating site scrapers.
    """

    SCRAPERS = {
        'betpawa': BetpawasScraper,
        'bongo_bongo_ug': BongoBongoScraper,
        '1xbet': OnexBetScraper,
        'bet365': Bet365Scraper,
    }

    @classmethod
    def create_scraper(cls, site_name: str) -> Optional[SiteScraper]:
        """
        Create a scraper for the specified site.

        Args:
            site_name: Name of the betting site

        Returns:
            Scraper instance or None if not found
        """
        scraper_class = cls.SCRAPERS.get(site_name.lower())
        if scraper_class:
            return scraper_class()
        logger.warning(f"No scraper found for site: {site_name}")
        return None

    @classmethod
    def get_supported_sites(cls) -> List[str]:
        """
        Get list of supported sites.

        Returns:
            List of site names
        """
        return list(cls.SCRAPERS.keys())
