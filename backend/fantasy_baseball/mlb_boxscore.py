"""
MLB Stats API integration for fetching box scores and resolving decisions.
"""

import logging
import re
import unicodedata
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """
    Normalize player name for matching.
    
    Handles:
    - Unicode normalization (José → Jose)
    - Case normalization
    - Common suffixes (Jr., III, etc.)
    - Extra whitespace
    """
    if not name:
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Normalize unicode (remove accents)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove common suffixes
    suffixes = [r'\s+jr\.?', r'\s+sr\.?', r'\s+iii', r'\s+ii', r'\s+iv', r'\s+v']
    for suffix in suffixes:
        name = re.sub(suffix, '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def fuzzy_name_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    """
    Fuzzy match two player names.
    
    Uses normalized comparison with edit distance ratio.
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    # Exact match after normalization
    if n1 == n2:
        return True
    
    # Try simple substring match for last name
    parts1 = n1.split()
    parts2 = n2.split()
    
    if len(parts1) >= 1 and len(parts2) >= 1:
        # Last name match (most reliable)
        if parts1[-1] == parts2[-1]:
            # First name initial match (e.g., "J. Smith" vs "John Smith")
            if len(parts1) >= 2 and len(parts2) >= 2:
                if parts1[0][0] == parts2[0][0]:
                    return True
    
    # Calculate similarity ratio
    try:
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, n1, n2).ratio()
        return similarity >= threshold
    except ImportError:
        return False


class MLBBoxScoreFetcher:
    """Fetch player stats from MLB Stats API."""
    
    BASE_URL = "https://statsapi.mlb.com/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_player_stats_for_date(
        self, 
        player_name: str, 
        team_abbr: str, 
        game_date: str
    ) -> Optional[Dict[str, float]]:
        """
        Get batting stats for a player on a specific date.
        
        Handles doubleheaders by aggregating stats across all games.
        
        Args:
            player_name: Full player name (e.g., "Pete Alonso")
            team_abbr: Team abbreviation (e.g., "NYM")
            game_date: YYYY-MM-DD
            
        Returns:
            Dict with hr, r, rbi, sb, avg or None if no game/no stats
        """
        try:
            # Find ALL games for this team on this date (handles doubleheaders)
            game_pks = self._find_all_game_pks(team_abbr, game_date)
            if not game_pks:
                logger.debug(f"No games found for {team_abbr} on {game_date}")
                return None
            
            # Aggregate stats across all games
            aggregated_stats = None
            for game_pk in game_pks:
                box_score = self._fetch_box_score(game_pk)
                if not box_score:
                    continue
                
                stats = self._extract_player_stats(box_score, player_name, team_abbr)
                if stats:
                    if aggregated_stats is None:
                        aggregated_stats = stats.copy()
                    else:
                        # Sum counting stats
                        for key in ['hr', 'r', 'rbi', 'sb', 'h', 'ab', '2b', '3b', 'bb']:
                            aggregated_stats[key] += stats.get(key, 0)
                        # Recalculate average
                        if aggregated_stats['ab'] > 0:
                            aggregated_stats['avg'] = round(aggregated_stats['h'] / aggregated_stats['ab'], 3)
            
            return aggregated_stats
            
        except Exception as e:
            logger.warning(f"Failed to fetch stats for {player_name}: {e}")
            return None
    
    def get_all_stats_for_date(self, game_date: str) -> Dict[str, Dict[str, float]]:
        """
        Get all player stats for all games on a date.
        
        Returns:
            Dict mapping "Player Name" -> stats dict
        """
        all_stats = {}
        
        try:
            # Get all games for the date
            games = self._get_games_for_date(game_date)
            
            for game in games:
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue
                
                # Check if game is final
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    logger.debug(f"Game {game_pk} not final yet")
                    continue
                
                # Fetch box score
                box_score = self._fetch_box_score(game_pk)
                if not box_score:
                    continue
                
                # Extract all player stats from this game
                game_stats = self._extract_all_players_stats(box_score)
                all_stats.update(game_stats)
                
        except Exception as e:
            logger.error(f"Failed to fetch all stats for {game_date}: {e}")
        
        return all_stats
    
    def _find_game_pk(self, team_abbr: str, game_date: str) -> Optional[int]:
        """Find game PK for a team on a specific date."""
        games = self._find_all_game_pks(team_abbr, game_date)
        return games[0] if games else None
    
    def _find_all_game_pks(self, team_abbr: str, game_date: str) -> List[int]:
        """Find all game PKs for a team on a specific date (handles doubleheaders)."""
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "teamId": self._get_team_id(team_abbr),
        }
        
        game_pks = []
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for date_info in data.get("dates", []):
                for game in date_info.get("games", []):
                    game_pk = game.get("gamePk")
                    if game_pk:
                        game_pks.append(game_pk)
                    
        except Exception as e:
            logger.warning(f"Failed to find games for {team_abbr}: {e}")
        
        return game_pks
    
    def _get_games_for_date(self, game_date: str) -> List[Dict]:
        """Get all games for a date."""
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
        }
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            games = []
            for date_info in data.get("dates", []):
                games.extend(date_info.get("games", []))
            return games
            
        except Exception as e:
            logger.error(f"Failed to fetch games for {game_date}: {e}")
            return []
    
    def _fetch_box_score(self, game_pk: int) -> Optional[Dict]:
        """Fetch box score for a game."""
        url = f"{self.BASE_URL}/game/{game_pk}/boxscore"
        
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch box score for {game_pk}: {e}")
            return None
    
    def _extract_player_stats(
        self, 
        box_score: Dict, 
        player_name: str, 
        team_abbr: str
    ) -> Optional[Dict[str, float]]:
        """Extract stats for a specific player from box score using fuzzy matching."""
        normalized_search = normalize_name(player_name)
        
        # Try both teams
        for team_type in ["home", "away"]:
            team_data = box_score.get("teams", {}).get(team_type, {})
            players = team_data.get("players", {})
            
            for player_id, player_data in players.items():
                info = player_data.get("person", {})
                name = info.get("fullName", "")
                
                # Try exact match first (fast path)
                if name.lower() == player_name.lower():
                    return self._parse_hitting_stats(player_data)
                
                # Try fuzzy match
                if fuzzy_name_match(name, player_name):
                    logger.debug(f"Fuzzy matched '{player_name}' to '{name}'")
                    return self._parse_hitting_stats(player_data)
        
        logger.debug(f"Could not find player '{player_name}' (normalized: '{normalized_search}')")
        return None
    
    def _extract_all_players_stats(self, box_score: Dict) -> Dict[str, Dict[str, float]]:
        """Extract all player stats from a box score."""
        all_stats = {}
        
        for team_type in ["home", "away"]:
            team_data = box_score.get("teams", {}).get(team_type, {})
            players = team_data.get("players", {})
            
            for player_id, player_data in players.items():
                info = player_data.get("person", {})
                name = info.get("fullName", "")
                
                stats = self._parse_hitting_stats(player_data)
                if stats:  # Only include if they have hitting stats
                    all_stats[name] = stats
        
        return all_stats
    
    def _parse_hitting_stats(self, player_data: Dict) -> Optional[Dict[str, float]]:
        """Parse hitting stats from player data."""
        stats = player_data.get("stats", {}).get("batting", {})
        
        if not stats:
            return None
        
        at_bats = stats.get("atBats", 0)
        hits = stats.get("hits", 0)
        
        return {
            "hr": float(stats.get("homeRuns", 0)),
            "r": float(stats.get("runs", 0)),
            "rbi": float(stats.get("rbi", 0)),
            "sb": float(stats.get("stolenBases", 0)),
            "avg": round(hits / at_bats, 3) if at_bats > 0 else 0.0,
            "h": float(hits),
            "ab": float(at_bats),
            "2b": float(stats.get("doubles", 0)),
            "3b": float(stats.get("triples", 0)),
            "bb": float(stats.get("baseOnBalls", 0)),
        }
    
    def _get_team_id(self, team_abbr: str) -> Optional[int]:
        """Map team abbreviation to MLB team ID."""
        # Common abbreviations to MLB team IDs
        team_ids = {
            "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
            "CHC": 112, "CWS": 145, "CIN": 113, "CLE": 114,
            "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
            "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
            "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
            "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
            "SEA": 136, "STL": 138, "TB": 139, "TEX": 140,
            "TOR": 141, "WAS": 120,
            # Full names
            "DIAMONDBACKS": 109, "BRAVES": 144, "ORIOLES": 110, "RED SOX": 111,
            "CUBS": 112, "WHITE SOX": 145, "REDS": 113, "GUARDIANS": 114,
            "ROCKIES": 115, "TIGERS": 116, "ASTROS": 117, "ROYALS": 118,
            "ANGELS": 108, "DODGERS": 119, "MARLINS": 146, "BREWERS": 158,
            "TWINS": 142, "METS": 121, "YANKEES": 147, "ATHLETICS": 133,
            "PHILLIES": 143, "PIRATES": 134, "PADRES": 135, "GIANTS": 137,
            "MARINERS": 136, "CARDINALS": 138, "RAYS": 139, "RANGERS": 140,
            "BLUE JAYS": 141, "NATIONALS": 120,
        }
        return team_ids.get(team_abbr.upper())


# Singleton instance
_mlb_fetcher: Optional[MLBBoxScoreFetcher] = None


def get_mlb_fetcher() -> MLBBoxScoreFetcher:
    """Get singleton MLB fetcher instance."""
    global _mlb_fetcher
    if _mlb_fetcher is None:
        _mlb_fetcher = MLBBoxScoreFetcher()
    return _mlb_fetcher
