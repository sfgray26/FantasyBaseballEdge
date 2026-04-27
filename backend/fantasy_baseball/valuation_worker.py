"""
ARCH-001 Phase 2 -- Player Valuation Worker.

Runs once daily at 06:00 ET (registered in daily_ingestion.py).
Pre-computes PlayerValuationReport for every rostered player and upserts
into player_valuation_cache. The API never blocks on this computation.

GUARDIAN FREEZE: Do NOT import betting_model or analysis.
ADR-005: This module is write-only toward the cache; API layer is read-only.
"""
import asyncio
import logging
import uuid
from datetime import datetime, date
from typing import List, Optional
from zoneinfo import ZoneInfo

from backend.models import SessionLocal, PlayerValuationCache
from backend.contracts import (
    PlayerValuationReport,
    CategoryProjection,
    UncertaintyRange,
    AuditTrail,
    DataSource,
)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Tier-based std_dev approximation for UncertaintyRange
# Used when we lack full MCMC distribution
_TIER_STD_DEV = {
    "elite": 0.15,
    "above_avg": 0.20,
    "average": 0.25,
    "below_avg": 0.30,
    "unknown": 0.35,
}


def _make_uncertainty(point: float, tier: str = "average") -> UncertaintyRange:
    """Parametric approximation of uncertainty using tier-based std_dev."""
    sd = _TIER_STD_DEV.get(tier, 0.25)
    z80 = 1.282 * sd
    z95 = 1.960 * sd
    return UncertaintyRange(
        point_estimate=point,
        lower_80=round(point - z80, 4),
        upper_80=round(point + z80, 4),
        lower_95=round(point - z95, 4),
        upper_95=round(point + z95, 4),
        std_dev=round(sd, 4),
        sample_size=0,  # 0 = parametric approximation, not sampled
    )


def _build_report(player: dict, target_date: str, now_et: datetime) -> PlayerValuationReport:
    """
    Build a PlayerValuationReport from a player dict.

    player dict keys (populated by _assemble_player_data):
        player_id, player_name, positions, team,
        category_projections: list of {category, value, z_score}
        composite_z: float
        matchup_quality: float (0-1)
        start_probability: float (0-1)
        recent_form_delta: float
        platoon_flag: Optional[str]
        park_factor: Optional[float]
        data_sources: list of DataSource
        warnings: list of str
    """
    cat_projections = [
        CategoryProjection(
            category=cp["category"],
            projection=_make_uncertainty(cp["value"]),
            z_score=cp.get("z_score", 0.0),
        )
        for cp in player.get("category_projections", [])
    ]
    composite_value = _make_uncertainty(player.get("composite_z", 0.0))
    audit = AuditTrail(
        created_at=now_et,
        model_version="arch001-v1",
        data_sources=player.get("data_sources", [DataSource.PLAYER_BOARD]),
        data_as_of=now_et,
        computation_ms=0,
        warnings=player.get("warnings", []),
    )
    return PlayerValuationReport(
        report_id=str(uuid.uuid4()),
        player_id=player["player_id"],
        player_name=player["player_name"],
        target_date=target_date,
        category_projections=cat_projections,
        composite_value=composite_value,
        matchup_quality=player.get("matchup_quality", 0.5),
        start_probability=player.get("start_probability", 1.0),
        recent_form_delta=player.get("recent_form_delta", 0.0),
        platoon_flag=player.get("platoon_flag"),
        park_factor=player.get("park_factor"),
        audit=audit,
    )


def _upsert_report(db, report: PlayerValuationReport, league_key: str, now_et: datetime) -> None:
    """Upsert a single PlayerValuationReport into player_valuation_cache."""
    from sqlalchemy import text
    target_date = date.fromisoformat(report.target_date)

    # Soft-delete any existing valid entry for this player+date+league
    db.execute(
        text("""
            UPDATE player_valuation_cache
            SET invalidated_at = :now
            WHERE player_id = :pid
              AND target_date = :tdate
              AND league_key = :lkey
              AND invalidated_at IS NULL
        """),
        {
            "now": now_et.replace(tzinfo=None),
            "pid": report.player_id,
            "tdate": target_date,
            "lkey": league_key,
        },
    )

    # Insert fresh record
    cache_row = PlayerValuationCache(
        id=str(uuid.uuid4()),
        player_id=report.player_id,
        player_name=report.player_name,
        target_date=target_date,
        league_key=league_key,
        report=report.dict(),
        computed_at=now_et.replace(tzinfo=None),
        invalidated_at=None,
        data_as_of=report.audit.data_as_of.replace(tzinfo=None),
    )
    db.add(cache_row)


async def _fetch_rostered_players(league_key: str) -> List[dict]:
    """
    Fetch all rostered players for a league via Yahoo API.
    Returns list of {player_id, player_name, positions, team}.
    Degrades gracefully: returns [] on any error.
    """
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient
        client = ResilientYahooClient()
        # get_league_rosters is sync -- run in thread pool
        rosters = await asyncio.to_thread(client.get_league_rosters, league_key)
        players = []
        for team_roster in rosters:
            for p in team_roster.get("players", []):
                players.append({
                    "player_id": str(p.get("player_id") or p.get("player_key", "")),
                    "player_name": p.get("name") or p.get("full_name", ""),
                    "positions": p.get("eligible_positions", []),
                    "team": p.get("editorial_team_abbr", ""),
                })
        return players
    except Exception as exc:
        logger.warning("valuation_worker: Yahoo roster fetch failed (%s) -- using empty list", exc)
        return []


async def _fetch_statcast_metrics(player_ids: List[str]) -> dict:
    """
    Fetch recent Statcast metrics for players.
    Returns {player_id: {metric_key: value}}.
    Degrades gracefully: returns {} on any error.
    """
    try:
        # Import here to respect GUARDIAN FREEZE boundary
        from backend.fantasy_baseball.statcast_ingestion import get_player_metrics_bulk
        metrics = await asyncio.to_thread(get_player_metrics_bulk, player_ids)
        return metrics or {}
    except Exception as exc:
        logger.warning("valuation_worker: Statcast fetch failed (%s) -- using empty metrics", exc)
        return {}


def _assemble_player_data(player: dict, statcast: dict) -> dict:
    """
    Merge Yahoo roster data + Statcast metrics into valuation input dict.
    All fields have safe defaults so missing data never raises.
    """
    pid = player["player_id"]
    player_name = player.get("player_name", pid)
    sc = statcast.get(pid, {})
    missing_fields: list = []

    # Build category projections from available metrics
    cat_projections = []

    # Batting categories
    for cat, key in [("AVG", "avg"), ("HR", "hr"), ("RBI", "rbi"), ("R", "r"), ("SB", "sb")]:
        val = sc.get(key, 0.0)
        if val is not None:
            z_score = sc.get(f"{key}_z")
            if z_score is None:
                logger.warning(
                    "valuation: player %s missing %s_z (z-score) -- defaulting to 0.0",
                    player_name, key,
                )
                missing_fields.append(f"{key}_z")
                z_score = 0.0
            cat_projections.append({
                "category": cat,
                "value": float(val),
                "z_score": z_score,
            })

    # Pitching categories
    for cat, key in [("W", "wins"), ("ERA", "era"), ("WHIP", "whip"), ("K", "k_per_9"), ("SV", "saves")]:
        val = sc.get(key)
        if val is not None:
            z_score = sc.get(f"{key}_z")
            if z_score is None:
                logger.warning(
                    "valuation: player %s missing %s_z (z-score) -- defaulting to 0.0",
                    player_name, key,
                )
                missing_fields.append(f"{key}_z")
                z_score = 0.0
            cat_projections.append({
                "category": cat,
                "value": float(val),
                "z_score": z_score,
            })

    # Composite value: average of available z-scores
    z_scores = [cp["z_score"] for cp in cat_projections if cp["z_score"] != 0.0]
    if not z_scores:
        logger.warning(
            "valuation: player %s has no non-zero z-scores -- composite_z defaults to 0.0 (tier=average)",
            player_name,
        )
        missing_fields.append("composite_z")
    composite_z = sum(z_scores) / len(z_scores) if z_scores else 0.0

    # Determine tier from composite z-score
    if composite_z >= 1.5:
        tier = "elite"
    elif composite_z >= 0.5:
        tier = "above_avg"
    elif composite_z >= -0.5:
        tier = "average"
    elif composite_z >= -1.5:
        tier = "below_avg"
    else:
        tier = "unknown"

    matchup_quality = sc.get("matchup_quality")
    if matchup_quality is None:
        logger.warning(
            "valuation: player %s missing matchup_quality -- defaulting to 0.5",
            player_name,
        )
        missing_fields.append("matchup_quality")
        matchup_quality = 0.5

    start_probability = sc.get("start_probability")
    if start_probability is None:
        logger.warning(
            "valuation: player %s missing start_probability -- defaulting to 1.0",
            player_name,
        )
        missing_fields.append("start_probability")
        start_probability = 1.0

    recent_form_delta = sc.get("recent_form_delta")
    if recent_form_delta is None:
        logger.warning(
            "valuation: player %s missing recent_form_delta -- defaulting to 0.0",
            player_name,
        )
        missing_fields.append("recent_form_delta")
        recent_form_delta = 0.0

    data_sources = [DataSource.PLAYER_BOARD]
    if sc:
        data_sources.append(DataSource.STATCAST)

    return {
        **player,
        "category_projections": cat_projections,
        "composite_z": composite_z,
        "tier": tier,
        "matchup_quality": matchup_quality,
        "start_probability": start_probability,
        "recent_form_delta": recent_form_delta,
        "platoon_flag": sc.get("platoon_flag"),
        "park_factor": sc.get("park_factor"),
        "data_sources": data_sources,
        "warnings": [],
        "_missing_fields": missing_fields,
    }


async def run_valuation_worker(league_key: str) -> dict:
    """
    Main entry point. Fetches rosters, merges Statcast, upserts cache.

    Returns summary dict: {players_processed, players_skipped, league_key, target_date}

    Failure modes:
    - Yahoo fetch fails -> empty roster -> 0 players processed (API serves stale)
    - Statcast fails -> empty metrics -> projections use defaults (API serves degraded but not broken)
    - DB write fails -> logs error, raises so advisory lock is released properly
    """
    now_et = datetime.now(_ET)
    target_date = now_et.strftime("%Y-%m-%d")

    logger.info("valuation_worker: starting for league=%s date=%s", league_key, target_date)
    start_ms = int(datetime.now(_ET).timestamp() * 1000)

    # I/O call 1: Yahoo roster (async, thread pool)
    players = await _fetch_rostered_players(league_key)
    if not players:
        logger.warning("valuation_worker: no players fetched for league=%s -- cache not updated", league_key)
        return {"players_processed": 0, "players_skipped": 0, "league_key": league_key, "target_date": target_date}

    # I/O call 2: Statcast metrics (async, thread pool)
    player_ids = [p["player_id"] for p in players]
    statcast = await _fetch_statcast_metrics(player_ids)

    # In-memory assembly
    assembled = [_assemble_player_data(p, statcast) for p in players]

    total_assembled = len(assembled)
    degraded_count = sum(1 for p in assembled if p.get("_missing_fields"))
    complete_count = total_assembled - degraded_count
    elapsed_ms = int(datetime.now(_ET).timestamp() * 1000) - start_ms
    logger.info(
        "valuation_worker: assembled %d players -- %d complete, %d degraded (missing fields) in %dms",
        total_assembled, complete_count, degraded_count, elapsed_ms,
    )

    # Upsert all reports in a single transaction
    db = SessionLocal()
    players_processed = 0
    players_skipped = 0
    try:
        for player_data in assembled:
            try:
                report = _build_report(player_data, target_date, now_et)
                _upsert_report(db, report, league_key, now_et)
                players_processed += 1
            except Exception as exc:
                logger.warning(
                    "valuation_worker: skipping player %s (%s)",
                    player_data.get("player_id"),
                    exc,
                )
                players_skipped += 1
        db.commit()
        logger.info(
            "valuation_worker: complete -- processed=%d skipped=%d league=%s",
            players_processed, players_skipped, league_key,
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "players_processed": players_processed,
        "players_skipped": players_skipped,
        "league_key": league_key,
        "target_date": target_date,
    }
