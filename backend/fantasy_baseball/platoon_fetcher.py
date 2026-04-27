"""
Platoon split fetching from FanGraphs via pybaseball.

Caches results to avoid repeated API calls.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_TTL_DAYS = 7  # Platoon splits update weekly


@dataclass
class PlatoonSplits:
    """Player's platoon split performance (wOBA vs each hand)."""
    player_name: str
    player_id: Optional[int] = None
    vs_lhp: float = 0.0  # wOBA vs LHP
    vs_rhp: float = 0.0  # wOBA vs RHP
    sample_vs_l: int = 0  # PA sample size
    sample_vs_r: int = 0
    last_updated: Optional[datetime] = None
    
    @property
    def split_delta(self) -> float:
        """Positive = hits LHP better, Negative = hits RHP better."""
        if self.sample_vs_l < 50 or self.sample_vs_r < 50:
            return 0.0  # Insufficient sample
        return self.vs_lhp - self.vs_rhp
    
    @property
    def is_reliable(self) -> bool:
        """Whether we have enough sample to trust the split."""
        return self.sample_vs_l >= 50 and self.sample_vs_r >= 50


class PlatoonSplitFetcher:
    """
    Fetch platoon splits from FanGraphs via pybaseball.
    
    Uses the 'splits' leaderboards which provide:
    - wOBA vs LHP
    - wOBA vs RHP  
    - PA vs each hand
    """
    
    # FanGraphs team ID mapping for MLB Stats API abbreviations
    TEAM_ID_MAP = {
        "ARI": 15, "ATL": 16, "BAL": 2, "BOS": 3, "CHC": 17,
        "CIN": 18, "CLE": 5, "COL": 19, "CWS": 4, "DET": 6,
        "HOU": 21, "KC": 7, "LAA": 1, "LAD": 22, "MIA": 20,
        "MIL": 23, "MIN": 8, "NYM": 25, "NYY": 9, "OAK": 10,
        "PHI": 26, "PIT": 27, "SD": 29, "SEA": 11, "SF": 30,
        "STL": 28, "TB": 12, "TEX": 13, "TOR": 14, "WAS": 24,
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, PlatoonSplits] = {}
    
    def get_splits(self, player_name: str, year: Optional[int] = None) -> Optional[PlatoonSplits]:
        """
        Get platoon splits for a player.
        
        Checks memory cache -> file cache -> FanGraphs API
        """
        cache_key = self._normalize_name(player_name)
        
        # Memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # File cache
        cached = self._load_from_cache(cache_key, year)
        if cached:
            self._memory_cache[cache_key] = cached
            return cached
        
        # Fetch from FanGraphs
        splits = self._fetch_from_fangraphs(player_name, year)
        if splits:
            self._save_to_cache(cache_key, splits, year)
            self._memory_cache[cache_key] = splits
        
        return splits
    
    def get_splits_batch(
        self, 
        player_names: list[str], 
        year: Optional[int] = None
    ) -> Dict[str, PlatoonSplits]:
        """Fetch splits for multiple players efficiently."""
        results = {}
        
        for name in player_names:
            splits = self.get_splits(name, year)
            if splits:
                results[name] = splits
        
        return results
    
    def _fetch_from_fangraphs(
        self, 
        player_name: str, 
        year: Optional[int] = None
    ) -> Optional[PlatoonSplits]:
        """
        Fetch platoon splits from FanGraphs via pybaseball.
        
        Uses batting_stats with split handedness filter.
        """
        try:
            # Import pybaseball here to handle optional dependency
            from pybaseball import batting_stats
            from pybaseball.playerid_lookup import playerid_lookup
            
            current_year = year or datetime.now().year
            
            # Look up player ID
            lookup = playerid_lookup(player_name.split()[0], player_name.split()[-1])
            if lookup.empty:
                logger.debug(f"Player not found: {player_name}")
                return None
            
            # Get FanGraphs ID
            fg_id = lookup.iloc[0].get("key_fangraphs")
            if not fg_id:
                return None
            
            # Fetch batting stats with split
            # Note: pybaseball's batting_stats supports 'versus' parameter for splits
            vs_lhb = batting_stats(
                current_year, 
                current_year,
                split_season=False,
                qual=1,  # Minimum PA
                vs="L"  # vs LHP
            )
            
            vs_rhb = batting_stats(
                current_year,
                current_year, 
                split_season=False,
                qual=1,
                vs="R"  # vs RHP
            )
            
            # Find player in each split
            lhp_row = vs_lhb[vs_lhb['IDfg'] == int(fg_id)]
            rhp_row = vs_rhb[vs_rhb['IDfg'] == int(fg_id)]
            
            if lhp_row.empty and rhp_row.empty:
                return None
            
            # Extract wOBA and PA
            vs_lhp = float(lhp_row.iloc[0].get('wOBA', 0)) if not lhp_row.empty else 0.0
            vs_rhp = float(rhp_row.iloc[0].get('wOBA', 0)) if not rhp_row.empty else 0.0
            sample_l = int(lhp_row.iloc[0].get('PA', 0)) if not lhp_row.empty else 0
            sample_r = int(rhp_row.iloc[0].get('PA', 0)) if not rhp_row.empty else 0
            
            return PlatoonSplits(
                player_name=player_name,
                player_id=int(fg_id),
                vs_lhp=vs_lhp,
                vs_rhp=vs_rhp,
                sample_vs_l=sample_l,
                sample_vs_r=sample_r,
                last_updated=datetime.now()
            )
            
        except ImportError:
            logger.warning("pybaseball not installed; platoon splits unavailable")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch platoon splits for {player_name}: {e}")
            return None
    
    def _load_from_cache(
        self, 
        cache_key: str, 
        year: Optional[int]
    ) -> Optional[PlatoonSplits]:
        """Load splits from file cache if fresh."""
        year_str = str(year or datetime.now().year)
        cache_file = self.cache_dir / f"platoon_{cache_key}_{year_str}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['last_updated'])
            
            # Check if cache is fresh
            if datetime.now() - cached_time > timedelta(days=CACHE_TTL_DAYS):
                return None
            
            return PlatoonSplits(
                player_name=data['player_name'],
                player_id=data.get('player_id'),
                vs_lhp=data['vs_lhp'],
                vs_rhp=data['vs_rhp'],
                sample_vs_l=data['sample_vs_l'],
                sample_vs_r=data['sample_vs_r'],
                last_updated=cached_time
            )
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(
        self, 
        cache_key: str, 
        splits: PlatoonSplits, 
        year: Optional[int]
    ) -> None:
        """Save splits to file cache."""
        year_str = str(year or datetime.now().year)
        cache_file = self.cache_dir / f"platoon_{cache_key}_{year_str}.json"
        
        try:
            data = {
                'player_name': splits.player_name,
                'player_id': splits.player_id,
                'vs_lhp': splits.vs_lhp,
                'vs_rhp': splits.vs_rhp,
                'sample_vs_l': splits.sample_vs_l,
                'sample_vs_r': splits.sample_vs_r,
                'last_updated': splits.last_updated.isoformat() if splits.last_updated else datetime.now().isoformat()
            }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize player name for cache key."""
        return name.lower().replace(" ", "_").replace(".", "")


def get_platoon_fetcher() -> PlatoonSplitFetcher:
    """Factory function for PlatoonSplitFetcher."""
    return PlatoonSplitFetcher()
