"""
MCMC Calibration Layer — B5 Implementation

Bridges Yahoo roster data + PlayerDailyMetric z-scores with the MCMC simulator.

Flow:
  Yahoo Roster → PlayerDailyMetric z-scores → cat_scores → MCMC simulator

Key design decisions:
1. Players on the player_board use their pre-computed cat_scores
2. Players not on the board get proxy cat_scores derived from z_score_recent
3. Proxy scores are distributed across categories based on position type
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from backend.models import PlayerDailyMetric, SessionLocal
from backend.fantasy_baseball.player_board import build_board, get_or_create_projection

logger = logging.getLogger(__name__)

# Category keys expected by MCMC simulator
BATTER_CATS = ["hr", "r", "rbi", "nsb", "avg", "ops", "tb", "h"]
PITCHER_CATS = ["k_pit", "era", "whip", "w", "nsv", "qs", "k9"]

# Position-based category weight templates (rough approximation of importance)
_BATTER_CAT_WEIGHTS: Dict[str, float] = {
    "hr": 1.0,
    "r": 1.0,
    "rbi": 1.0,
    "nsb": 0.8,  # Slightly less weighted
    "avg": 0.9,
    "ops": 1.0,
    "tb": 0.9,
    "h": 0.7,
}

_PITCHER_CAT_WEIGHTS: Dict[str, float] = {
    "k_pit": 1.0,
    "era": 1.0,
    "whip": 1.0,
    "w": 0.9,
    "nsv": 0.8,
    "qs": 0.9,
    "k9": 0.8,
}

# In-memory cache for player board (refreshed on first call)
_player_board_cache: Optional[List[dict]] = None
_player_board_lookup: Optional[Dict[str, dict]] = None


def _get_player_board() -> tuple[List[dict], Dict[str, dict]]:
    """Get or build the player board and name lookup."""
    global _player_board_cache, _player_board_lookup
    if _player_board_cache is None:
        _player_board_cache = build_board()
        _player_board_lookup = {
            p["name"].lower().replace(".", "").replace("'", ""): p
            for p in _player_board_cache
        }
    return _player_board_cache, _player_board_lookup or {}


def _is_pitcher(positions: List[str]) -> bool:
    """Check if player is a pitcher based on positions."""
    pitcher_positions = {"SP", "RP", "P"}
    return any(p in pitcher_positions for p in positions)


def _get_player_z_score_from_db(
    player_name: str,
    db: Optional[Session] = None,
) -> Optional[float]:
    """
    Fetch z_score_recent from PlayerDailyMetric for a player.
    
    Returns None if no recent data available.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        recent_cutoff = date.today() - timedelta(days=3)
        metric = (
            db.query(PlayerDailyMetric)
            .filter(
                PlayerDailyMetric.player_name.ilike(player_name),
                PlayerDailyMetric.sport == "mlb",
                PlayerDailyMetric.metric_date >= recent_cutoff,
            )
            .order_by(PlayerDailyMetric.metric_date.desc())
            .first()
        )
        
        if metric and metric.z_score_recent is not None:
            return float(metric.z_score_recent)
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch z-score for {player_name}: {e}")
        return None
    finally:
        if close_db and db is not None:
            db.close()


def _build_proxy_cat_scores(
    player_name: str,
    positions: List[str],
    total_z: float,
) -> Dict[str, float]:
    """
    Build proxy cat_scores from a total z-score when per-category data unavailable.
    
    Distributes the total z-score across categories based on position-type templates.
    """
    is_pit = _is_pitcher(positions)
    cats = PITCHER_CATS if is_pit else BATTER_CATS
    weights = _PITCHER_CAT_WEIGHTS if is_pit else _BATTER_CAT_WEIGHTS
    
    # Normalize weights
    total_weight = sum(weights.get(c, 1.0) for c in cats)
    if total_weight == 0:
        total_weight = 1.0
    
    # Distribute total_z proportionally
    # Scale factor: empirical tuning to make proxy scores roughly comparable
    # to player_board scores (which are weighted z-scores)
    scale = 0.3  # Proxy scores are noisier, so scale down
    
    cat_scores = {}
    for cat in cats:
        w = weights.get(cat, 1.0)
        cat_scores[cat] = round(total_z * w / total_weight * scale, 3)
    
    return cat_scores


def _get_starts_this_week(
    player: dict,
    pitcher_start_cache: Optional[Dict[str, int]] = None,
) -> int:
    """
    Estimate pitcher starts this week.
    
    Uses the has_start/pitcher_slot info if available, otherwise defaults to 1 for SPs.
    """
    positions = player.get("positions", [])
    
    # Only SPs have multiple starts
    if "SP" not in positions:
        return 1
    
    # Check if we have explicit start info from lineup optimizer
    if player.get("has_start") and player.get("pitcher_slot") == "SP":
        # Try to get from cache if provided
        name = player.get("name", "")
        if pitcher_start_cache and name in pitcher_start_cache:
            return min(pitcher_start_cache[name], 2)
        
        # Default: most SPs get 1 start, estimate 2-start based on rotation
        # This is simplified — in production would use actual probables
        return 1
    
    return 1


def convert_yahoo_roster_to_mcmc_format(
    roster: List[dict],
    db: Optional[Session] = None,
    pitcher_start_cache: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """
    Convert Yahoo roster format to MCMC simulator format.
    
    Each output player dict has:
        - name: str
        - positions: List[str]
        - starts_this_week: int
        - cat_scores: Dict[str, float] — z-scores per category
    
    Args:
        roster: List of Yahoo player dicts from get_roster()
        db: Optional database session
        pitcher_start_cache: Optional map of pitcher name -> estimated starts
    
    Returns:
        List of player dicts ready for mcmc_simulator.simulate_weekly_matchup()
    """
    _, board_lookup = _get_player_board()
    result: List[dict] = []
    
    for yahoo_player in roster:
        name = yahoo_player.get("name", "")
        positions = yahoo_player.get("positions", [])
        
        if not name:
            continue
        
        # 1. Try to get from player board (best quality cat_scores)
        name_key = name.lower().replace(".", "").replace("'", "")
        board_player = board_lookup.get(name_key)
        
        if board_player and board_player.get("cat_scores"):
            cat_scores = board_player["cat_scores"]
            # Use board's total z_score as fallback if available
            total_z = board_player.get("z_score", 0.0)
        else:
            # 2. Try to get z_score from PlayerDailyMetric
            db_z = _get_player_z_score_from_db(name, db)
            
            if db_z is not None:
                total_z = db_z
                cat_scores = _build_proxy_cat_scores(name, positions, total_z)
            else:
                # 3. Use player_board's proxy generator as last resort
                proxy = get_or_create_projection(yahoo_player)
                total_z = proxy.get("z_score", 0.0)
                cat_scores = _build_proxy_cat_scores(name, positions, total_z)
        
        # Get pitcher starts estimate
        starts = _get_starts_this_week(yahoo_player, pitcher_start_cache)
        
        result.append({
            "name": name,
            "positions": positions,
            "starts_this_week": starts,
            "cat_scores": cat_scores,
        })
    
    return result


def calculate_matchup_win_probability(
    my_roster: List[dict],
    opponent_roster: List[dict],
    db: Optional[Session] = None,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate win probability for a weekly H2H matchup.
    
    This is the main entry point for the dashboard service.
    
    Args:
        my_roster: Yahoo roster dicts for my team
        opponent_roster: Yahoo roster dicts for opponent
        db: Optional database session
        n_sims: Number of Monte Carlo simulations
        seed: Optional RNG seed for reproducibility
    
    Returns:
        Dict with win_prob, category_win_probs, expected_cats_won, etc.
    """
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup
    
    # Convert rosters to MCMC format
    my_mcmc = convert_yahoo_roster_to_mcmc_format(my_roster, db)
    opp_mcmc = convert_yahoo_roster_to_mcmc_format(opponent_roster, db)
    
    # Run simulation
    result = simulate_weekly_matchup(
        my_roster=my_mcmc,
        opponent_roster=opp_mcmc,
        n_sims=n_sims,
        seed=seed,
    )
    
    return result


def invalidate_board_cache():
    """Invalidate the player board cache (call after board updates)."""
    global _player_board_cache, _player_board_lookup
    _player_board_cache = None
    _player_board_lookup = None
