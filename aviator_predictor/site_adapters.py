"""
Site adapters for different Aviator betting platforms.

Supports:
- Betpawa
- Bongo Bongo UG
- 1xBet
- Melbet
- Generic Aviator platforms
"""

import requests
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from abc import ABC, abstractmethod


class BaseAviatorAdapter(ABC):
    """
    Abstract base class for Aviator site adapters.
    """

    def __init__(self, base_url: str, timeout: int = 10):
        """
        Initialize adapter.

        Args:
            base_url: Base URL of the betting site
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.last_games = []

    @abstractmethod
    def get_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Get game history from the site.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data dictionaries
        """
        pass

    @abstractmethod
    def get_live_game(self) -> Optional[Dict]:
        """
        Get current live game data.

        Returns:
            Current game data or None if no active game
        """
        pass

    @abstractmethod
    def extract_multiplier(self, game_data: Dict) -> float:
        """
        Extract crash multiplier from game data.

        Args:
            game_data: Raw game data from site

        Returns:
            Crash multiplier value
        """
        pass

    def format_game_data(self, game_data: Dict) -> Dict:
        """
        Format raw game data to standard format.

        Args:
            game_data: Raw game data

        Returns:
            Standardized game data
        """
        return {
            'game_id': game_data.get('id', ''),
            'timestamp': game_data.get('timestamp', datetime.now().isoformat()),
            'crash_multiplier': self.extract_multiplier(game_data),
            'players_count': game_data.get('players_count', 0),
            'total_bet': game_data.get('total_bet', 0),
            'game_duration': game_data.get('duration', 0),
        }

    def close(self):
        """Close the session."""
        self.session.close()


class BetpawaAdapter(BaseAviatorAdapter):
    """
    Adapter for Betpawa Aviator game.
    """

    def __init__(self):
        """Initialize Betpawa adapter."""
        super().__init__('https://betpawa.ug')
        self.api_endpoint = f'{self.base_url}/api/v1/aviator'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Get game history from Betpawa.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            url = f'{self.api_endpoint}/history'
            params = {'limit': min(limit, 1000)}
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            games = response.json().get('games', [])
            self.last_games = [self.format_game_data(g) for g in games]
            return self.last_games
        except Exception as e:
            print(f"Error fetching Betpawa game history: {e}")
            return []

    def get_live_game(self) -> Optional[Dict]:
        """
        Get current live game from Betpawa.

        Returns:
            Current game data or None
        """
        try:
            url = f'{self.api_endpoint}/live'
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            game = response.json().get('game')
            if game:
                return self.format_game_data(game)
            return None
        except Exception as e:
            print(f"Error fetching Betpawa live game: {e}")
            return None

    def extract_multiplier(self, game_data: Dict) -> float:
        """
        Extract multiplier from Betpawa game data.

        Args:
            game_data: Betpawa game data

        Returns:
            Crash multiplier
        """
        return float(game_data.get('multiplier', game_data.get('crash_multiplier', 0)))


class BongoBongoUgAdapter(BaseAviatorAdapter):
    """
    Adapter for Bongo Bongo UG Aviator game.
    """

    def __init__(self):
        """Initialize Bongo Bongo UG adapter."""
        super().__init__('https://bongobongo.ug')
        self.api_endpoint = f'{self.base_url}/api/games/aviator'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

    def get_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Get game history from Bongo Bongo UG.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            url = f'{self.api_endpoint}/results'
            params = {
                'limit': min(limit, 500),
                'sort': 'desc'
            }
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            games = data.get('results', data.get('games', []))
            self.last_games = [self.format_game_data(g) for g in games]
            return self.last_games
        except Exception as e:
            print(f"Error fetching Bongo Bongo UG game history: {e}")
            return []

    def get_live_game(self) -> Optional[Dict]:
        """
        Get current live game from Bongo Bongo UG.

        Returns:
            Current game data or None
        """
        try:
            url = f'{self.api_endpoint}/current'
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            game = response.json().get('game')
            if game:
                return self.format_game_data(game)
            return None
        except Exception as e:
            print(f"Error fetching Bongo Bongo UG live game: {e}")
            return None

    def extract_multiplier(self, game_data: Dict) -> float:
        """
        Extract multiplier from Bongo Bongo UG game data.

        Args:
            game_data: Bongo Bongo UG game data

        Returns:
            Crash multiplier
        """
        return float(game_data.get('result', game_data.get('multiplier', 0)))


class OnexbetAdapter(BaseAviatorAdapter):
    """
    Adapter for 1xBet Aviator game.
    """

    def __init__(self):
        """Initialize 1xBet adapter."""
        super().__init__('https://1xbet.com')
        self.api_endpoint = f'{self.base_url}/api/Games/Aviator'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

    def get_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Get game history from 1xBet.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            url = f'{self.api_endpoint}/History'
            params = {'pageSize': min(limit, 100)}
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            games = response.json().get('Games', [])
            self.last_games = [self.format_game_data(g) for g in games]
            return self.last_games
        except Exception as e:
            print(f"Error fetching 1xBet game history: {e}")
            return []

    def get_live_game(self) -> Optional[Dict]:
        """
        Get current live game from 1xBet.

        Returns:
            Current game data or None
        """
        try:
            url = f'{self.api_endpoint}/Current'
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            game = response.json().get('Game')
            if game:
                return self.format_game_data(game)
            return None
        except Exception as e:
            print(f"Error fetching 1xBet live game: {e}")
            return None

    def extract_multiplier(self, game_data: Dict) -> float:
        """
        Extract multiplier from 1xBet game data.

        Args:
            game_data: 1xBet game data

        Returns:
            Crash multiplier
        """
        return float(game_data.get('CrashMultiplier', game_data.get('Multiplier', 0)))


class MelbetAdapter(BaseAviatorAdapter):
    """
    Adapter for Melbet Aviator game.
    """

    def __init__(self):
        """Initialize Melbet adapter."""
        super().__init__('https://melbet.com')
        self.api_endpoint = f'{self.base_url}/api/aviator'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

    def get_game_history(self, limit: int = 100) -> List[Dict]:
        """
        Get game history from Melbet.

        Args:
            limit: Number of games to fetch

        Returns:
            List of game data
        """
        try:
            url = f'{self.api_endpoint}/history'
            params = {'take': min(limit, 500)}
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            games = response.json().get('history', [])
            self.last_games = [self.format_game_data(g) for g in games]
            return self.last_games
        except Exception as e:
            print(f"Error fetching Melbet game history: {e}")
            return []

    def get_live_game(self) -> Optional[Dict]:
        """
        Get current live game from Melbet.

        Returns:
            Current game data or None
        """
        try:
            url = f'{self.api_endpoint}/current'
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            game = response.json().get('current_game')
            if game:
                return self.format_game_data(game)
            return None
        except Exception as e:
            print(f"Error fetching Melbet live game: {e}")
            return None

    def extract_multiplier(self, game_data: Dict) -> float:
        """
        Extract multiplier from Melbet game data.

        Args:
            game_data: Melbet game data

        Returns:
            Crash multiplier
        """
        return float(game_data.get('crash_value', game_data.get('multiplier', 0)))


class AviatorAdapterFactory:
    """
    Factory for creating Aviator site adapters.
    """

    ADAPTERS = {
        'betpawa': BetpawaAdapter,
        'bongo_bongo': BongoBongoUgAdapter,
        'bongobongo': BongoBongoUgAdapter,
        '1xbet': OnexbetAdapter,
        'melbet': MelbetAdapter,
    }

    @classmethod
    def create_adapter(cls, site_name: str) -> Optional[BaseAviatorAdapter]:
        """
        Create an adapter for a specific site.

        Args:
            site_name: Name of the betting site

        Returns:
            Adapter instance or None if site not supported
        """
        site_key = site_name.lower().strip()
        adapter_class = cls.ADAPTERS.get(site_key)
        if adapter_class:
            return adapter_class()
        print(f"Unsupported site: {site_name}. Supported: {list(cls.ADAPTERS.keys())}")
        return None

    @classmethod
    def get_supported_sites(cls) -> List[str]:
        """
        Get list of supported sites.

        Returns:
            List of site names
        """
        return list(cls.ADAPTERS.keys())

    @classmethod
    def add_custom_adapter(cls, site_name: str, adapter_class):
        """
        Add a custom adapter.

        Args:
            site_name: Name of the betting site
            adapter_class: Adapter class (must inherit from BaseAviatorAdapter)
        """
        if not issubclass(adapter_class, BaseAviatorAdapter):
            raise ValueError("Adapter class must inherit from BaseAviatorAdapter")
        cls.ADAPTERS[site_name.lower()] = adapter_class


class MultiSitePredictor:
    """
    Multi-site predictor that works across different Aviator platforms.
    """

    def __init__(self, sites: List[str] = None):
        """
        Initialize multi-site predictor.

        Args:
            sites: List of site names to use
        """
        if sites is None:
            sites = AviatorAdapterFactory.get_supported_sites()

        self.adapters = {}
        for site in sites:
            adapter = AviatorAdapterFactory.create_adapter(site)
            if adapter:
                self.adapters[site] = adapter

    def get_game_history(self, site: str, limit: int = 100) -> List[Dict]:
        """
        Get game history from a specific site.

        Args:
            site: Site name
            limit: Number of games to fetch

        Returns:
            List of games or empty list if site not available
        """
        adapter = self.adapters.get(site.lower())
        if adapter:
            return adapter.get_game_history(limit)
        return []

    def get_all_game_history(self, limit: int = 50) -> Dict[str, List[Dict]]:
        """
        Get game history from all configured sites.

        Args:
            limit: Number of games per site

        Returns:
            Dictionary mapping site names to game lists
        """
        history = {}
        for site, adapter in self.adapters.items():
            history[site] = adapter.get_game_history(limit)
        return history

    def get_live_game(self, site: str) -> Optional[Dict]:
        """
        Get live game from a specific site.

        Args:
            site: Site name

        Returns:
            Current game data or None
        """
        adapter = self.adapters.get(site.lower())
        if adapter:
            return adapter.get_live_game()
        return None

    def get_all_live_games(self) -> Dict[str, Optional[Dict]]:
        """
        Get live games from all configured sites.

        Returns:
            Dictionary mapping site names to current games
        """
        games = {}
        for site, adapter in self.adapters.items():
            games[site] = adapter.get_live_game()
        return games

    def close_all(self):
        """Close all adapters."""
        for adapter in self.adapters.values():
            adapter.close()
