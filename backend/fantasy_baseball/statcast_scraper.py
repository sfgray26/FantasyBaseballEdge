"""
Baseball Savant (Statcast) Data Scraper

Uses pybaseball library (unofficial Python wrapper for Baseball Savant)
or direct API calls to Statcast to retrieve:
- Statcast batting leaderboards
- Statcast pitching leaderboards  
- Spray charts
- Pitch-level data
- Player lookup

Alternative: Use baseballr package data exports or manual CSV downloads
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Baseball Savant API endpoints
BASEBALL_SAVANT_API = "https://baseballsavant.mlb.com"


def _cache_key(endpoint: str, params: dict) -> str:
    """Generate cache filename from endpoint and params."""
    param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"{endpoint}_{param_str}.json"


def _get_cached_or_fetch(endpoint: str, params: dict, use_cache: bool = True) -> dict:
    """Get data from cache or fetch from API."""
    cache_file = CACHE_DIR / _cache_key(endpoint, params)
    
    if use_cache and cache_file.exists():
        logger.debug(f"Using cached data: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Fetch from API
    url = f"{BASEBALL_SAVANT_API}/{endpoint}"
    if params:
        url += f"?{urlencode(params)}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Cache the response
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return {}


def get_statcast_batting_leaderboards(
    year: int = 2025,
    min_pa: int = 200,
    stat_type: str = "expected_statistics"
) -> List[Dict]:
    """
    Get Statcast batting leaderboards from Baseball Savant.
    
    stat_type options:
    - expected_statistics (xBA, xSLG, xwOBA, etc)
    - exit_velocity_barrels (barrel%, exit velo, hard hit%)
    - plate_discipline (O-Swing%, Z-Contact%, etc)
    - batted_ball_profile (GB%, FB%, LD%, Pull%)
    - sprint_speed (sprint speed, bolts)
    """
    params = {
        "year": year,
        "min_pa": min_pa,
        "type": stat_type,
    }
    
    data = _get_cached_or_fetch("leaderboard/statcast", params)
    return data.get("leaderboard", [])


def get_statcast_pitching_leaderboards(
    year: int = 2025,
    min_ip: int = 50,
    stat_type: str = "expected_statistics"
) -> List[Dict]:
    """
    Get Statcast pitching leaderboards.
    
    stat_type options:
    - expected_statistics (xERA, xwOBA allowed, etc)
    - pitch_arsenal_stats (Stuff+, Location+, Pitching+)
    - plate_discipline (Whiff%, Chase%, CSW%)
    - batted_ball_profile (Barrel% allowed, Hard Hit% allowed)
    - velocity (fastball velo, spin rates)
    """
    params = {
        "year": year,
        "min_ip": min_ip,
        "type": stat_type,
    }
    
    data = _get_cached_or_fetch("leaderboard/statcast_pitching", params)
    return data.get("leaderboard", [])


def get_player_statcast_batting(
    player_name: str,
    player_id: Optional[int] = None,
    year: int = 2025
) -> Dict:
    """Get detailed Statcast batting stats for a specific player."""
    # Would use player ID lookup or search
    params = {
        "player_id": player_id or player_name,
        "year": year,
    }
    
    return _get_cached_or_fetch("player/statcast_batting", params)


def get_player_statcast_pitching(
    player_name: str,
    player_id: Optional[int] = None,
    year: int = 2025
) -> Dict:
    """Get detailed Statcast pitching stats for a specific player."""
    params = {
        "player_id": player_id or player_name,
        "year": year,
    }
    
    return _get_cached_or_fetch("player/statcast_pitching", params)


def search_player_id(name: str) -> Optional[int]:
    """Search for MLBAM player ID by name."""
    params = {"search": name}
    data = _get_cached_or_fetch("player/search", params, use_cache=False)
    
    if data.get("players"):
        return data["players"][0].get("mlbam_id")
    return None


# ---------------------------------------------------------------------------
# CSV Export Functions (for manual Baseball Savant downloads)
# ---------------------------------------------------------------------------

def parse_statcast_batting_csv(csv_path: Path) -> List[Dict]:
    """
    Parse a Baseball Savant batting CSV export.
    Users can download these from:
    https://baseballsavant.mlb.com/leaderboard/statcast
    """
    import csv
    
    players = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = {
                "name": row.get("last_name, first_name", row.get("player_name", "")).strip(),
                "team": row.get("team", "").strip(),
                "player_id": int(row.get("player_id", 0)) if row.get("player_id") else 0,
                "pa": int(row.get("pa", 0)) if row.get("pa") else 0,
                
                # Batted ball quality
                "barrel_pct": float(row.get("barrel_batted_rate", 0)) if row.get("barrel_batted_rate") else 0.0,
                "exit_velo_avg": float(row.get("exit_velocity_avg", 0)) if row.get("exit_velocity_avg") else 0.0,
                "hard_hit_pct": float(row.get("hard_hit_percent", 0)) if row.get("hard_hit_percent") else 0.0,
                "sweet_spot_pct": float(row.get("sweet_spot_percent", 0)) if row.get("sweet_spot_percent") else 0.0,
                
                # Expected stats
                "xba": float(row.get("xba", 0)) if row.get("xba") else 0.0,
                "xslg": float(row.get("xslg", 0)) if row.get("xslg") else 0.0,
                "xwoba": float(row.get("xwoba", 0)) if row.get("xwoba") else 0.0,
                "woba": float(row.get("woba", 0)) if row.get("woba") else 0.0,
                "xwoba_diff": float(row.get("xwoba", 0)) - float(row.get("woba", 0)) if row.get("xwoba") and row.get("woba") else 0.0,
                
                # Plate discipline
                "o_swing_pct": float(row.get("oz_swing_percent", 0)) if row.get("oz_swing_percent") else 0.0,
                "z_swing_pct": float(row.get("z_swing_percent", 0)) if row.get("z_swing_percent") else 0.0,
                "o_contact_pct": float(row.get("oz_contact_percent", 0)) if row.get("oz_contact_percent") else 0.0,
                "z_contact_pct": float(row.get("iz_contact_percent", 0)) if row.get("iz_contact_percent") else 0.0,
                "swstr_pct": float(row.get("whiff_percent", 0)) if row.get("whiff_percent") else 0.0,
                
                # Batted ball profile
                "gb_pct": float(row.get("groundballs_percent", 0)) if row.get("groundballs_percent") else 0.0,
                "fb_pct": float(row.get("flyballs_percent", 0)) if row.get("flyballs_percent") else 0.0,
                "ld_pct": float(row.get("linedrives_percent", 0)) if row.get("linedrives_percent") else 0.0,
                "pull_pct": float(row.get("pull_percent", 0)) if row.get("pull_percent") else 0.0,
                "oppo_pct": float(row.get("oppo_percent", 0)) if row.get("oppo_percent") else 0.0,
            }
            players.append(player)
    
    return players


def parse_statcast_pitching_csv(csv_path: Path) -> List[Dict]:
    """
    Parse a Baseball Savant pitching CSV export.
    """
    import csv
    
    players = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = {
                "name": row.get("last_name, first_name", row.get("player_name", "")).strip(),
                "team": row.get("team", "").strip(),
                "player_id": int(row.get("player_id", 0)) if row.get("player_id") else 0,
                "ip": float(row.get("ip", 0)) if row.get("ip") else 0.0,
                
                # Pitch quality
                "stuff_plus": float(row.get("stuff_plus", 100)) if row.get("stuff_plus") else 100.0,
                "location_plus": float(row.get("location_plus", 100)) if row.get("location_plus") else 100.0,
                "pitching_plus": float(row.get("pitching_plus", 100)) if row.get("pitching_plus") else 100.0,
                
                # Velocity
                "fb_velo_avg": float(row.get("n_ff_formatted", 0)) if row.get("n_ff_formatted") else 0.0,
                "spin_rate_fb": int(row.get("ff_spin", 0)) if row.get("ff_spin") else 0,
                
                # Whiff metrics
                "whiff_pct": float(row.get("whiff_percent", 0)) if row.get("whiff_percent") else 0.0,
                "chase_pct": float(row.get("oz_swing_percent", 0)) if row.get("oz_swing_percent") else 0.0,
                "csw_pct": float(row.get("csw_percent", 0)) if row.get("csw_percent") else 0.0,
                
                # Contact quality allowed
                "barrel_allowed_pct": float(row.get("barrel_batted_rate", 0)) if row.get("barrel_batted_rate") else 0.0,
                "hard_hit_allowed_pct": float(row.get("hard_hit_percent", 0)) if row.get("hard_hit_percent") else 0.0,
                
                # Expected stats
                "xera": float(row.get("xera", 0)) if row.get("xera") else 0.0,
                "era": float(row.get("era", 0)) if row.get("era") else 0.0,
                "xwoba_allowed": float(row.get("xwoba", 0)) if row.get("xwoba") else 0.0,
            }
            
            # Calculate xERA difference if both values present
            if player["xera"] > 0 and player["era"] > 0:
                player["xera_diff"] = player["xera"] - player["era"]
            else:
                player["xera_diff"] = 0.0
            
            players.append(player)
    
    return players


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------

DOWNLOAD_INSTRUCTIONS = """
BASEBALL SAVANT DATA DOWNLOAD INSTRUCTIONS
==========================================

Since Baseball Savant doesn't have a public API, follow these steps
to download CSV files and place them in data/cache/:

1. BATTING - Expected Statistics:
   https://baseballsavant.mlb.com/leaderboard/expected_statistics
   - Select year: 2025
   - Min PA: 100
   - Download CSV
   - Save as: data/cache/statcast_batting_expected_2025.csv

2. BATTING - Exit Velocity & Barrels:
   https://baseballsavant.mlb.com/leaderboard/statcast
   - Leaderboard: Exit Velocity & Barrels
   - Year: 2025, Min PA: 100
   - Download CSV
   - Save as: data/cache/statcast_batting_ev_2025.csv

3. BATTING - Sprint Speed:
   https://baseballsavant.mlb.com/leaderboard/sprint_speed
   - Year: 2025, Min PA: 50
   - Download CSV
   - Save as: data/cache/statcast_batting_speed_2025.csv

4. PITCHING - Expected Statistics:
   https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher
   - Year: 2025, Min IP: 30
   - Download CSV
   - Save as: data/cache/statcast_pitching_expected_2025.csv

5. PITCHING - Pitch Arsenal Stats:
   https://baseballsavant.mlb.com/leaderboard/pitch_arsenal
   - Year: 2025
   - Download CSV
   - Save as: data/cache/statcast_pitching_arsenal_2025.csv

Then run: python -m backend.fantasy_baseball.statcast_scraper
"""


if __name__ == "__main__":
    print(DOWNLOAD_INSTRUCTIONS)
