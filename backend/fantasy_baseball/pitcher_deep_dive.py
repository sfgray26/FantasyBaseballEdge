"""
Pitcher deep dive data from FanGraphs.

Elite managers want to know more than ERA - they want:
- FIP/xFIP/SIERA (predictive ERA indicators)
- Batted ball profile (GB%, FB%, Hard%)
- Recent form (last 3 starts)
- Splits vs LHB/RHB
- Rest days
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import requests

from backend.fantasy_baseball.elite_context import PitcherDeepDive

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_TTL_HOURS = 6  # Pitcher data updates frequently


class PitcherDeepDiveFetcher:
    """
    Fetch comprehensive pitcher data from FanGraphs.
    
    Uses FanGraphs API endpoints for:
    - Season stats (ERA, FIP, xFIP, SIERA)
    - Batted ball data (GB%, FB%, Hard%)
    - Splits vs LHB/RHB
    - Game logs (recent form)
    """
    
    # FanGraphs leaderboard endpoint (unofficial but stable)
    FG_LEADERBOARD_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
    FG_SPLITS_URL = "https://www.fangraphs.com/api/players/splits/split"
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def get_pitcher(self, name: str, team: str) -> Optional[PitcherDeepDive]:
        """
        Fetch complete pitcher profile.
        
        Args:
            name: Pitcher full name (e.g., "Zack Wheeler")
            team: Team abbreviation (e.g., "PHI")
        
        Returns:
            PitcherDeepDive with full stats or None if not found
        """
        cache_key = f"pitcher_{name.lower().replace(' ', '_')}"
        
        # Check cache
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        # Fetch from FanGraphs
        try:
            pitcher = self._fetch_from_fangraphs(name, team)
            if pitcher:
                self._save_cache(cache_key, pitcher)
            return pitcher
        except Exception as e:
            logger.warning(f"Failed to fetch pitcher data for {name}: {e}")
            return None
    
    def _fetch_from_fangraphs(self, name: str, team: str) -> Optional[PitcherDeepDive]:
        """Fetch pitcher data from FanGraphs API."""
        
        # Step 1: Search for player to get ID
        player_id = self._search_player(name)
        if not player_id:
            logger.debug(f"Player not found: {name}")
            return None
        
        # Step 2: Fetch season stats
        season_stats = self._fetch_season_stats(player_id)
        if not season_stats:
            return None
        
        # Step 3: Fetch splits
        splits = self._fetch_splits(player_id)
        
        # Step 4: Build PitcherDeepDive
        return PitcherDeepDive(
            name=name,
            team=team,
            handedness=season_stats.get("throws", "R"),
            era=float(season_stats.get("era", 4.50)),
            whip=float(season_stats.get("whip", 1.30)),
            k9=float(season_stats.get("k9", 8.0)),
            bb9=float(season_stats.get("bb9", 3.0)),
            fip=float(season_stats.get("fip", 4.50)),
            xfip=float(season_stats.get("xfip", 4.50)),
            sierra=float(season_stats.get("sierra", 4.50)),
            gb_pct=float(season_stats.get("gb_pct", 45.0)),
            fb_pct=float(season_stats.get("fb_pct", 35.0)),
            hr_fb=float(season_stats.get("hr_fb", 12.0)),
            hard_hit_pct=float(season_stats.get("hard_pct", 35.0)),
            era_vs_lhb=float(splits.get("vs_lhb", {}).get("era", season_stats.get("era", 4.50))),
            era_vs_rhb=float(splits.get("vs_rhb", {}).get("era", season_stats.get("era", 4.50))),
            k9_vs_lhb=float(splits.get("vs_lhb", {}).get("k9", season_stats.get("k9", 8.0))),
            k9_vs_rhb=float(splits.get("vs_rhb", {}).get("k9", season_stats.get("k9", 8.0))),
        )
    
    def _search_player(self, name: str) -> Optional[int]:
        """Search for player ID by name."""
        # Try to get from pybaseball first (if available)
        try:
            from pybaseball.playerid_lookup import playerid_lookup
            parts = name.split()
            if len(parts) >= 2:
                lookup = playerid_lookup(parts[0], parts[-1])
                if not lookup.empty:
                    return int(lookup.iloc[0]["key_fangraphs"])
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"pybaseball lookup failed: {e}")
        
        return None
    
    def _fetch_season_stats(self, player_id: int) -> Dict:
        """Fetch season stats from FanGraphs."""
        try:
            params = {
                "pos": "all",
                "stats": "pit",
                "lg": "all",
                "qual": "y",
                "season": datetime.now().year,
                "season1": datetime.now().year,
                "ind": "0",
                "team": "0",
                "rost": "0",
                "age": "0",
                "filter": "",
                "players": str(player_id),
                "page": "1_1000",
            }
            
            resp = self._session.get(self.FG_LEADERBOARD_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("data"):
                return data["data"][0]  # First (and only) player
            
            return {}
        except Exception as e:
            logger.debug(f"Failed to fetch season stats: {e}")
            return {}
    
    def _fetch_splits(self, player_id: int) -> Dict[str, Dict]:
        """Fetch platoon splits."""
        splits = {"vs_lhb": {}, "vs_rhb": {}}
        
        try:
            # vs LHB
            params_l = {
                "playerid": player_id,
                "position": "P",
                "split": "vl",  # vs left
                "season": datetime.now().year,
            }
            resp_l = self._session.get(self.FG_SPLITS_URL, params=params_l, timeout=30)
            if resp_l.status_code == 200:
                data_l = resp_l.json()
                if data_l.get("data"):
                    splits["vs_lhb"] = {
                        "era": float(data_l["data"][0].get("era", 0)),
                        "k9": float(data_l["data"][0].get("k9", 0)),
                    }
            
            # vs RHB
            params_r = {
                "playerid": player_id,
                "position": "P",
                "split": "vr",  # vs right
                "season": datetime.now().year,
            }
            resp_r = self._session.get(self.FG_SPLITS_URL, params=params_r, timeout=30)
            if resp_r.status_code == 200:
                data_r = resp_r.json()
                if data_r.get("data"):
                    splits["vs_rhb"] = {
                        "era": float(data_r["data"][0].get("era", 0)),
                        "k9": float(data_r["data"][0].get("k9", 0)),
                    }
        except Exception as e:
            logger.debug(f"Failed to fetch splits: {e}")
        
        return splits
    
    def _load_cache(self, cache_key: str) -> Optional[PitcherDeepDive]:
        """Load from file cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            
            if datetime.now() - cached_time > timedelta(hours=CACHE_TTL_HOURS):
                return None
            
            # Reconstruct PitcherDeepDive
            return PitcherDeepDive(**data["data"])
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None
    
    def _save_cache(self, cache_key: str, pitcher: PitcherDeepDive) -> None:
        """Save to file cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data = {
                "cached_at": datetime.now().isoformat(),
                "data": asdict(pitcher)
            }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")


def get_pitcher_fetcher() -> PitcherDeepDiveFetcher:
    """Factory function."""
    return PitcherDeepDiveFetcher()
