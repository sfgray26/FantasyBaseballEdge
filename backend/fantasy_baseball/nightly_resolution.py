"""
Nightly job to resolve fantasy baseball decisions from the previous day.

This should run after all games have completed (around midnight ET).
"""

import logging
from datetime import datetime, timedelta

from backend.fantasy_baseball.decision_tracker import get_decision_tracker, PlayerDecision
from backend.fantasy_baseball.mlb_boxscore import get_mlb_fetcher

logger = logging.getLogger(__name__)


def resolve_yesterdays_decisions() -> dict:
    """
    Resolve all pending decisions from yesterday with actual MLB stats.
    
    Returns summary of resolutions.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tracker = get_decision_tracker()
    mlb = get_mlb_fetcher()
    
    # Load pending decisions
    pending = []
    for decision in tracker._load_decisions_for_date(yesterday):
        if decision.outcome == "pending":
            pending.append(decision)
    
    if not pending:
        logger.info(f"No pending decisions to resolve for {yesterday}")
        return {"date": yesterday, "resolved": 0, "skipped": 0, "failed": 0}
    
    logger.info(f"Resolving {len(pending)} decisions for {yesterday}")
    
    # Fetch all stats for the date (more efficient than per-player)
    all_stats = mlb.get_all_stats_for_date(yesterday)
    logger.info(f"Fetched stats for {len(all_stats)} players from MLB API")
    
    resolved = 0
    failed = 0
    no_game = 0
    
    for decision in pending:
        try:
            # Look up stats by player name
            player_name = decision.player_name
            
            if player_name in all_stats:
                stats = all_stats[player_name]
                tracker.resolve_decision(
                    decision.decision_id,
                    actual_stats=stats,
                    game_happened=True
                )
                resolved += 1
                logger.debug(f"Resolved {player_name}: {stats}")
            else:
                # Check if game was postponed or player didn't play
                tracker.resolve_decision(
                    decision.decision_id,
                    actual_stats={"hr": 0, "r": 0, "rbi": 0, "sb": 0, "avg": 0},
                    game_happened=False  # No stats = no game or DNP
                )
                no_game += 1
                logger.debug(f"No stats for {player_name} - marking as no game")
                
        except Exception as e:
            logger.warning(f"Failed to resolve {decision.player_name}: {e}")
            failed += 1
    
    result = {
        "date": yesterday,
        "total_pending": len(pending),
        "resolved": resolved,
        "no_game": no_game,
        "failed": failed,
    }
    
    logger.info(f"Resolution complete: {resolved} resolved, {no_game} no game, {failed} failed")
    return result


def resolve_specific_date(date: str) -> dict:
    """
    Resolve decisions for a specific date.
    
    Args:
        date: YYYY-MM-DD
    """
    tracker = get_decision_tracker()
    mlb = get_mlb_fetcher()
    
    pending = [d for d in tracker._load_decisions_for_date(date) if d.outcome == "pending"]
    
    if not pending:
        return {"date": date, "message": "No pending decisions", "resolved": 0}
    
    all_stats = mlb.get_all_stats_for_date(date)
    
    resolved = 0
    for decision in pending:
        if decision.player_name in all_stats:
            tracker.resolve_decision(
                decision.decision_id,
                actual_stats=all_stats[decision.player_name],
                game_happened=True
            )
            resolved += 1
    
    return {
        "date": date,
        "total_pending": len(pending),
        "resolved": resolved,
    }


def get_pending_resolutions(date: str = None) -> list:
    """
    Get list of decisions pending resolution for a date.
    
    Args:
        date: YYYY-MM-DD, defaults to yesterday
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    tracker = get_decision_tracker()
    decisions = tracker._load_decisions_for_date(date)
    
    pending = [
        {
            "decision_id": d.decision_id,
            "player_name": d.player_name,
            "team": d.team,
            "recommended_action": d.recommended_action,
            "confidence": d.confidence,
            "opponent": d.opponent,
        }
        for d in decisions
        if d.outcome == "pending"
    ]
    
    return pending


if __name__ == "__main__":
    # Run resolution
    result = resolve_yesterdays_decisions()
    print(f"Resolution result: {result}")
