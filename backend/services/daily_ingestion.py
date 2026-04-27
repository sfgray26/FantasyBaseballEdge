"""
DailyIngestionOrchestrator — EPIC-2 data pipeline coordinator.

Owns all MLB/CBB data polling jobs that run independently of the nightly
CBB analysis scheduler. Each job acquires a PostgreSQL advisory lock before
running, which prevents duplicate execution across Railway replicas.

ADR-001: Every job MUST use _with_advisory_lock.
ADR-004: This file is additive only. Never import betting_model or analysis.
"""

import asyncio
import json
import logging
import os
import time
import math
import traceback
import unicodedata
from datetime import datetime, date, timedelta
from typing import Optional, Any
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.models import (
    SessionLocal,
    PlayerDailyMetric,
    ProjectionSnapshot,
    PlayerValuationCache,
    ProjectionCacheEntry,
    MLBTeam,
    MLBGameLog,
    MLBOddsSnapshot,
    MLBPlayerStats,
    PlayerRollingStats,
    PlayerScore,
    PlayerMomentum,
    PlayerIDMapping,
    SimulationResult as SimulationResultORM,
    DecisionResult as DecisionResultORM,
    BacktestResult as BacktestResultORM,
    DecisionExplanation as DecisionExplanationORM,
    DailySnapshot as DailySnapshotORM,
    PositionEligibility,
    ProbablePitcherSnapshot,
    DataIngestionLog,
    engine,
)
from backend.services.explainability_layer import ExplanationInput, explain_batch
from backend.services.probable_pitcher_fallback import (
    build_recent_starter_candidates,
    infer_probable_pitcher_for_team,
)
from backend.services.snapshot_engine import SnapshotInput, build_snapshot
from backend.services.backtesting_harness import (
    BacktestInput,
    evaluate_cohort,
    summarize,
    load_golden_baseline,
    save_golden_baseline,
    BASELINE_PATH,
)
from backend.services.simulation_engine import simulate_all_players, REMAINING_GAMES_DEFAULT
from backend.services.decision_engine import (
    PlayerDecisionInput,
    optimize_lineup,
    optimize_waivers,
)
from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
from backend.utils.time_utils import now_et, today_et

logger = logging.getLogger(__name__)


# Module-level mirror: RoS projections fetched by fangraphs_ros (100_012)
# and also persisted to projection_cache_entries for cross-process durability.
_ROS_CACHE: dict = {}
_ROS_CACHE_KEY = "fangraphs_ros"
_ROS_CACHE_TABLE_READY = False


# ---------------------------------------------------------------------------
# Advisory lock IDs — must match HANDOFF.md LOCK_IDS table
# ---------------------------------------------------------------------------

LOCK_IDS = {
    "mlb_odds":    100_001,
    "statcast":    100_002,
    "rolling_z":   100_003,
    "cbb_ratings": 100_004,
    "clv":         100_005,
    "cleanup":     100_006,
    "waiver_scan": 100_007,
    "mlb_brief":   100_008,
    "openclaw_perf":  100_009,
    "openclaw_sweep": 100_010,
    "valuation_cache": 100_011,
    "fangraphs_ros":   100_012,
    "yahoo_adp_injury": 100_013,
    "ensemble_update": 100_014,
    "projection_freshness": 100_015,
    "mlb_game_log":         100_016,
    "mlb_box_stats":        100_017,
    "rolling_windows":      100_018,
    "player_scores":        100_019,
    "player_momentum":      100_020,
    "ros_simulation":       100_021,
    "decision_optimization": 100_022,
    "backtesting":           100_023,
    "explainability":        100_024,
    "snapshot":              100_025,
    "statsapi_supplement":   100_026,
    "position_eligibility": 100_027,
    "probable_pitchers":     100_028,
    "player_id_mapping":     100_029,
    "vorp":                  100_030,
}


def _ensure_projection_cache_table() -> None:
    """Create the durable projection cache table if the migration has not run yet."""
    global _ROS_CACHE_TABLE_READY
    if _ROS_CACHE_TABLE_READY:
        return
    ProjectionCacheEntry.__table__.create(bind=engine, checkfirst=True)
    _ROS_CACHE_TABLE_READY = True


def _serialize_ros_frames(frames: Optional[dict]) -> dict[str, list[dict[str, Any]]]:
    """Convert a dict of pandas DataFrames into JSON-safe row lists."""
    if not frames:
        return {}

    serialized: dict[str, list[dict[str, Any]]] = {}
    for system_key, frame in frames.items():
        if frame is None:
            continue
        serialized[system_key] = json.loads(frame.to_json(orient="records", date_format="iso"))
    return serialized


def _parse_innings_pitched(ip: Optional[Any]) -> Optional[float]:
    """
    Convert BDL innings pitched format to decimal.

    BDL returns IP as "6.2" (6 innings + 2 outs) or "7" (7 innings) or 0.2 (2 outs).
    Convert to decimal: 6.2 → 6.667, 7 → 7.0, 0.2 → 0.667.

    Args:
        ip: Innings pitched from BDL API (str like "6.2", float, int, or None)

    Returns:
        Decimal innings pitched (6.2 → 6.667) or None if input is None/invalid

    Examples:
        >>> _parse_innings_pitched("6.2")
        6.667
        >>> _parse_innings_pitched(7)
        7.0
        >>> _parse_innings_pitched(None)
        None
    """
    if ip is None:
        return None

    # If already a number, return as float
    if isinstance(ip, (int, float)):
        return float(ip)

    # Parse "6.2" format: 6 innings + 2 outs
    if isinstance(ip, str):
        parts = ip.split(".")
        try:
            innings = int(parts[0])
            outs = int(parts[1]) if len(parts) > 1 else 0
            return innings + (outs / 3.0)
        except (ValueError, IndexError):
            return None

    return None


def _validate_mlb_stats(stat) -> bool:
    """
    Validate MLB stat row before database insertion.

    Prevents data quality issues by rejecting rows with impossible values.
    Logs warnings for rejected rows to aid in debugging.

    Args:
        stat: MLBPlayerStats object from BDL API

    Returns:
        True if stat row is valid, False otherwise
    """
    errors = []

    # Check ERA range (0-100)
    if stat.era is not None and (stat.era < 0 or stat.era > 100):
        errors.append(f"Invalid ERA: {stat.era}")

    # Check AVG range (0-1.0)
    if stat.avg is not None and (stat.avg < 0 or stat.avg > 1.0):
        errors.append(f"Invalid AVG: {stat.avg}")

    # Validate innings_pitched format
    if stat.ip is not None:
        try:
            ip_decimal = _parse_innings_pitched(stat.ip)
            if ip_decimal is None:
                errors.append(f"Invalid IP format: {stat.ip}")
        except Exception as e:
            errors.append(f"Invalid IP format: {stat.ip}")

    if errors:
        logger.warning(
            "mlb_box_stats: Validation failed for player %d: %s",
            stat.bdl_player_id, ", ".join(errors)
        )
        return False

    return True


def _deserialize_ros_frames(payload: Optional[dict]) -> dict[str, Any]:
    """Rebuild pandas DataFrames from persisted RoS row payloads."""
    if not payload:
        return {}

    import pandas as pd

    restored: dict[str, Any] = {}
    for system_key, rows in payload.items():
        restored[system_key] = pd.DataFrame(rows or [])
    return restored


def _store_persisted_ros_cache(
    bat_raw: Optional[dict],
    pit_raw: Optional[dict],
    fetched_at: datetime,
) -> None:
    """Persist raw Fangraphs payloads so downstream jobs survive process restarts."""
    _ensure_projection_cache_table()
    db = SessionLocal()
    try:
        entry = (
            db.query(ProjectionCacheEntry)
            .filter(ProjectionCacheEntry.cache_key == _ROS_CACHE_KEY)
            .first()
        )
        payload = {
            "bat": _serialize_ros_frames(bat_raw),
            "pit": _serialize_ros_frames(pit_raw),
        }
        if entry is None:
            entry = ProjectionCacheEntry(
                cache_key=_ROS_CACHE_KEY,
                payload=payload,
                fetched_at=fetched_at,
            )
            db.add(entry)
        else:
            entry.payload = payload
            entry.fetched_at = fetched_at
        db.commit()
    finally:
        db.close()


def _load_persisted_ros_cache(include_payload: bool = True) -> tuple[Optional[dict], Optional[dict], Optional[datetime]]:
    """Load the last persisted Fangraphs payload and fetched timestamp."""
    _ensure_projection_cache_table()
    db = SessionLocal()
    try:
        entry = (
            db.query(ProjectionCacheEntry)
            .filter(ProjectionCacheEntry.cache_key == _ROS_CACHE_KEY)
            .first()
        )
        if entry is None:
            return None, None, None
        if not include_payload:
            return None, None, entry.fetched_at
        payload = entry.payload or {}
        return (
            _deserialize_ros_frames(payload.get("bat")),
            _deserialize_ros_frames(payload.get("pit")),
            entry.fetched_at,
        )
    finally:
        db.close()


def _extract_blend_rows(blend_df: Any, metric_map: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    """Normalize a blend dataframe into upsert rows and count skipped entries."""
    if blend_df is None:
        return [], 0

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in blend_df.iterrows():
        player_id = row.get("player_id", "")
        if not player_id:
            skipped += 1
            continue

        metrics = {dest: row.get(src) for src, dest in metric_map.items()}

        def _is_missing_metric(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, float) and math.isnan(value):
                return True
            return False

        if all(_is_missing_metric(value) for value in metrics.values()):
            skipped += 1
            continue

        rows.append(
            {
                "player_id": player_id,
                "player_name": row.get("name", player_id),
                **metrics,
            }
        )

    return rows, skipped


# ---------------------------------------------------------------------------
# Advisory lock helper
# ---------------------------------------------------------------------------

async def _with_advisory_lock(lock_id: int, job_name: str, coro):
    """
    Acquire a session-level PostgreSQL advisory lock, run coro(), then release.
    If the lock is already held (another replica running the same job), skip
    execution and return None.

    Enhanced with comprehensive execution logging for observability.
    """
    job_start = time.monotonic()
    started_at = now_et()
    logger.info("JOB START: %s (lock %d) at %s", job_name, lock_id, started_at.isoformat())

    db = SessionLocal()
    log_row: Optional[DataIngestionLog] = None
    try:
        result = db.execute(
            text("SELECT pg_try_advisory_lock(:lid)"), {"lid": lock_id}
        ).scalar()
        if not result:
            logger.warning("JOB SKIPPED: %s (lock %d) - advisory lock held by another worker", job_name, lock_id)
            _persist_ingestion_log(
                db,
                job_type=job_name,
                target_date=today_et(),
                status="SKIPPED",
                started_at=started_at,
                completed_at=now_et(),
                processing_time_seconds=time.monotonic() - job_start,
                summary_stats={"skip_reason": "advisory_lock_held", "lock_id": lock_id},
                warning_details=[{"type": "advisory_lock", "message": "Lock held by another worker"}],
            )
            return None

        logger.info("JOB LOCK ACQUIRED: %s (lock %d)", job_name, lock_id)
        log_row = DataIngestionLog(
            job_type=job_name,
            target_date=today_et(),
            status="RUNNING",
            started_at=started_at,
            summary_stats={"lock_id": lock_id},
            error_details=[],
            warning_details=[],
        )
        db.add(log_row)
        db.commit()

        try:
            result = await coro()
            job_end = time.monotonic()
            elapsed_seconds = job_end - job_start
            elapsed_ms = int(elapsed_seconds * 1000)
            _persist_ingestion_log(
                db,
                log_row=log_row,
                job_type=job_name,
                target_date=today_et(),
                status=_normalize_ingestion_log_status(result.get("status") if isinstance(result, dict) else None),
                started_at=started_at,
                completed_at=now_et(),
                records_processed=_extract_processed_records(result),
                records_failed=_extract_failed_records(result),
                validation_errors=_extract_validation_count(result, "validation_errors"),
                validation_warnings=_extract_validation_count(result, "validation_warnings"),
                processing_time_seconds=elapsed_seconds,
                data_quality_score=_extract_data_quality_score(result),
                error_details=_extract_error_details(result),
                warning_details=_extract_warning_details(result),
                summary_stats=_sanitize_for_json(result),
                error_message=_extract_error_message(result),
            )
            logger.info("JOB COMPLETE: %s (lock %d) - status=%s, elapsed_ms=%d",
                       job_name, lock_id, result.get("status", "unknown"), elapsed_ms)
            return result

        except Exception as exc:
            job_end = time.monotonic()
            elapsed_seconds = job_end - job_start
            elapsed_ms = int(elapsed_seconds * 1000)
            _persist_ingestion_log(
                db,
                log_row=log_row,
                job_type=job_name,
                target_date=today_et(),
                status="FAILED",
                started_at=started_at,
                completed_at=now_et(),
                processing_time_seconds=elapsed_seconds,
                error_message=str(exc),
                error_details=[{"type": exc.__class__.__name__, "message": str(exc)}],
                stack_trace=traceback.format_exc(),
            )
            logger.error("JOB FAILED: %s (lock %d) - exception=%s, elapsed_ms=%d",
                        job_name, lock_id, str(exc), elapsed_ms, exc_info=True)
            raise

    finally:
        try:
            db.execute(text("SELECT pg_advisory_unlock(:lid)"), {"lid": lock_id})
        except Exception:
            pass
        db.close()


def _sanitize_for_json(value: Any) -> Any:
    """Convert common Python objects into JSON-safe payloads for audit logging."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    return str(value)


def _normalize_ingestion_log_status(raw_status: Optional[str]) -> str:
    """Map job return statuses to the durable log vocabulary."""
    normalized = (raw_status or "SUCCESS").strip().upper()
    if normalized in {"SUCCESS", "PARTIAL", "FAILED", "SKIPPED", "RUNNING"}:
        return normalized
    if normalized in {"ERROR", "FAIL", "FAILURE"}:
        return "FAILED"
    if normalized in {"OK", "DONE"}:
        return "SUCCESS"
    return normalized


def _extract_processed_records(result: Any) -> int:
    """Best-effort extraction of the primary processed-record count from a job result."""
    if not isinstance(result, dict):
        return 0

    direct_keys = (
        "records_processed",
        "records",
        "rows_written",
        "rows_upserted",
        "rows_patched",
        "records_written",
        "records_updated",
        "players_simulated",
        "n_players",
        "n_players_scored",
        "n_explained",
        "bat_rows",
        "pit_rows",
    )
    for key in direct_keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return int(value)

    composite_keys = ("lineup_decisions", "waiver_decisions")
    composite_total = 0
    found_composite = False
    for key in composite_keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            composite_total += int(value)
            found_composite = True
    if found_composite:
        return composite_total

    scored_window_keys = ("scored_7d", "scored_14d", "scored_30d")
    scored_total = 0
    found_scored = False
    for key in scored_window_keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            scored_total += int(value)
            found_scored = True
    if found_scored:
        return scored_total

    return 0


def _extract_failed_records(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    value = result.get("records_failed")
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _extract_validation_count(result: Any, key: str) -> int:
    if not isinstance(result, dict):
        return 0
    value = result.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _extract_data_quality_score(result: Any) -> Optional[float]:
    if not isinstance(result, dict):
        return None
    value = result.get("data_quality_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_error_message(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    for key in ("error_message", "error", "reason"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_error_details(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    details = result.get("error_details")
    if isinstance(details, list):
        return _sanitize_for_json(details)
    message = _extract_error_message(result)
    if message:
        return [{"message": message}]
    return []


def _extract_warning_details(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    details = result.get("warning_details")
    if isinstance(details, list):
        return _sanitize_for_json(details)
    warnings = result.get("warnings")
    if isinstance(warnings, list):
        return [{"message": str(item)} for item in _sanitize_for_json(warnings)]
    return []


def _persist_ingestion_log(
    db,
    *,
    job_type: str,
    target_date: date,
    status: str,
    started_at: datetime,
    completed_at: Optional[datetime] = None,
    records_processed: int = 0,
    records_failed: int = 0,
    processing_time_seconds: Optional[float] = None,
    validation_errors: int = 0,
    validation_warnings: int = 0,
    data_quality_score: Optional[float] = None,
    error_details: Optional[list[dict[str, Any]]] = None,
    warning_details: Optional[list[dict[str, Any]]] = None,
    summary_stats: Optional[Any] = None,
    error_message: Optional[str] = None,
    stack_trace: Optional[str] = None,
    log_row: Optional[DataIngestionLog] = None,
) -> None:
    """Persist or update the durable audit row for an ingestion job without affecting job success."""
    try:
        if log_row is None:
            log_row = DataIngestionLog(
                job_type=job_type,
                target_date=target_date,
                status=status,
                started_at=started_at,
            )
            db.add(log_row)

        log_row.status = status
        log_row.target_date = target_date
        log_row.started_at = started_at
        log_row.completed_at = completed_at
        log_row.records_processed = records_processed
        log_row.records_failed = records_failed
        log_row.processing_time_seconds = processing_time_seconds
        log_row.validation_errors = validation_errors
        log_row.validation_warnings = validation_warnings
        log_row.data_quality_score = data_quality_score
        log_row.error_details = _sanitize_for_json(error_details or [])
        log_row.warning_details = _sanitize_for_json(warning_details or [])
        log_row.summary_stats = _sanitize_for_json(summary_stats or {})
        log_row.error_message = error_message
        log_row.stack_trace = stack_trace
        db.commit()
    except Exception as log_exc:
        db.rollback()
        logger.warning("Failed to persist DataIngestionLog for %s: %s", job_type, log_exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DailyIngestionOrchestrator:
    """
    Coordinates background data-ingestion jobs on their own AsyncIOScheduler.
    Registered separately from the main CBB scheduler so ingestion can be
    disabled without affecting the nightly analysis pipeline.
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler(
            job_defaults={"misfire_grace_time": 300},
        )
        self._job_status: dict[str, dict] = {}
        self._openclaw: Optional[Any] = None
        # H2 fix: league params computed by _compute_player_scores (14d window),
        # consumed by _run_ros_simulation to enable composite risk metrics.
        self._league_means: Optional[dict] = None
        self._league_stds: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register all jobs and start the scheduler. Called once from lifespan()."""
        tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")

        # MLB game log ingestion: daily 1 AM ET
        # Fetches yesterday (final scores) + today (schedule). Runs before rolling_z (3 AM).
        self._scheduler.add_job(
            self._ingest_mlb_game_log,
            CronTrigger(hour=1, minute=0, timezone=tz),
            id="mlb_game_log",
            name="MLB Game Log Ingestion",
            replace_existing=True,
        )

        # MLB player box stats ingestion: daily 2 AM ET (1 hour after game-log at 1 AM)
        # Fetches yesterday + today box stats. Runs before rolling_z (3 AM) and fangraphs_ros (3 AM).
        self._scheduler.add_job(
            self._ingest_mlb_box_stats,
            CronTrigger(hour=2, minute=0, timezone=tz),
            id="mlb_box_stats",
            name="MLB Box Stats Ingestion",
            replace_existing=True,
        )

        # Supplement BDL counting stats with MLB Stats API: daily 2:30 AM ET
        # Fills NULL ab/h/r/doubles/triples/so/sb/cs from statsapi.boxscore_data().
        # Runs after mlb_box_stats (2 AM) and before rolling_windows (3 AM).
        self._scheduler.add_job(
            self._supplement_statsapi_counting_stats,
            CronTrigger(hour=2, minute=30, timezone=tz),
            id="statsapi_supplement",
            name="StatsAPI Counting Stats Supplement",
            replace_existing=True,
        )

        # Rolling window computation: daily 3 AM ET (after box stats at 2 AM)
        # Computes 7/14/30-day decay-weighted windows per player. Runs before Z-score calc (4 AM).
        self._scheduler.add_job(
            self._compute_rolling_windows,
            CronTrigger(hour=3, minute=0, timezone=tz),
            id="rolling_windows",
            name="Player Rolling Window Computation",
            replace_existing=True,
        )

        # Player Z-score scoring: daily 4 AM ET (after rolling windows at 3 AM)
        # Computes league Z-scores + 0-100 percentile ranks per player per window.
        self._scheduler.add_job(
            self._compute_player_scores,
            CronTrigger(hour=4, minute=0, timezone=tz),
            id="player_scores",
            name="Player Z-Score Scoring",
            replace_existing=True,
        )

        # VORP computation: daily 4:30 AM ET (after player_scores at 4 AM)
        # Computes VORP = composite_z - replacement_z(scarcest_position) per player.
        self._scheduler.add_job(
            self._compute_vorp,
            CronTrigger(hour=4, minute=30, timezone=tz),
            id="vorp",
            name="VORP Computation",
            replace_existing=True,
        )

        # Player momentum signals: daily 5 AM ET (after player_scores at 4 AM)
        # Computes delta-Z = Z_14d - Z_30d and assigns SURGING/HOT/STABLE/COLD/COLLAPSING.
        self._scheduler.add_job(
            self._compute_player_momentum,
            CronTrigger(hour=5, minute=0, timezone=tz),
            id="player_momentum",
            name="Player Momentum Signal Computation",
            replace_existing=True,
        )

        # RoS Monte Carlo simulation: daily 6 AM ET (after player_momentum at 5 AM)
        # Runs 1000-sim ROS projection per player from 14d rolling window.
        self._scheduler.add_job(
            self._run_ros_simulation,
            CronTrigger(hour=6, minute=0, timezone=tz),
            id="ros_simulation",
            name="RoS Monte Carlo Simulation",
            replace_existing=True,
        )

        # Decision optimization: daily 7 AM ET (after ros_simulation at 6 AM)
        # Runs greedy lineup + waiver analysis from player_scores + simulation_results.
        self._scheduler.add_job(
            self._run_decision_optimization,
            CronTrigger(hour=7, minute=0, timezone=tz),
            id="decision_optimization",
            name="Decision Engine Optimization",
            replace_existing=True,
        )

        # Backtesting harness: daily 8 AM ET (after decision_optimization at 7 AM)
        # Evaluates P16 simulation projections vs actual mlb_player_stats outcomes.
        self._scheduler.add_job(
            self._run_backtesting,
            CronTrigger(hour=8, minute=0, timezone=tz),
            id="backtesting",
            name="Backtesting Harness",
            replace_existing=True,
        )

        # Explainability: daily 9 AM ET (after backtesting at 8 AM)
        # Generates human-readable decision traces from all P14-P18 signals.
        self._scheduler.add_job(
            self._run_explainability,
            CronTrigger(hour=9, minute=0, timezone=tz),
            id="explainability",
            name="Explainability Engine",
            replace_existing=True,
        )

        # Snapshot: daily 10 AM ET (after explainability at 9 AM -- final pipeline stage)
        # Captures complete daily state: counts, health status, top players, regression flag.
        self._scheduler.add_job(
            self._run_snapshot,
            CronTrigger(hour=10, minute=0, timezone=tz),
            id="snapshot",
            name="Daily Snapshot",
            replace_existing=True,
        )

        # MLB odds polling: every 5 min, restricted to 10 AM - 11 PM ET
        self._scheduler.add_job(
            self._poll_mlb_odds,
            IntervalTrigger(minutes=5, timezone=tz),
            id="mlb_odds",
            name="MLB Odds Poll",
            replace_existing=True,
        )

        # Statcast enrichment: every 6 hours
        self._scheduler.add_job(
            self._update_statcast,
            IntervalTrigger(hours=6),
            id="statcast",
            name="Statcast Update",
            replace_existing=True,
        )

        # Rolling z-scores: daily 4 AM ET
        self._scheduler.add_job(
            self._calc_rolling_zscores,
            CronTrigger(hour=4, minute=0, timezone=tz),
            id="rolling_z",
            name="Rolling Z-Score Calc",
            replace_existing=True,
        )

        # CLV attribution: daily 11 PM ET
        self._scheduler.add_job(
            self._compute_clv,
            CronTrigger(hour=23, minute=0, timezone=tz),
            id="clv",
            name="Daily CLV Attribution",
            replace_existing=True,
        )

        # Metric cleanup: daily 3:30 AM ET
        self._scheduler.add_job(
            self._cleanup_old_metrics,
            CronTrigger(hour=3, minute=30, timezone=tz),
            id="cleanup",
            name="Old Metric Cleanup",
            replace_existing=True,
        )

        # Player valuation cache: daily 6 AM ET
        # Only runs if FANTASY_LEAGUES env var is set (comma-separated league keys)
        _fantasy_leagues = os.getenv("FANTASY_LEAGUES", "")
        if _fantasy_leagues:
            self._scheduler.add_job(
                self._refresh_valuation_cache,
                CronTrigger(hour=6, minute=0, timezone=tz),
                id="valuation_cache",
                name="Player Valuation Cache Refresh",
                replace_existing=True,
            )

        # FanGraphs RoS projections: daily 3 AM ET (before ensemble at 5 AM)
        self._scheduler.add_job(
            self._fetch_fangraphs_ros,
            CronTrigger(hour=3, minute=0, timezone=tz),
            id="fangraphs_ros",
            name="FanGraphs RoS Fetch",
            replace_existing=True,
        )

        # Yahoo ADP + injury feed: every 4 hours
        self._scheduler.add_job(
            self._poll_yahoo_adp_injury,
            IntervalTrigger(hours=4),
            id="yahoo_adp_injury",
            name="Yahoo ADP & Injury Poll",
            replace_existing=True,
        )

        # Ensemble blend update: daily 5 AM ET (after RoS fetch at 3 AM)
        self._scheduler.add_job(
            self._update_ensemble_blend,
            CronTrigger(hour=5, minute=0, timezone=tz),
            id="ensemble_update",
            name="Ensemble Blend Update",
            replace_existing=True,
        )

        # Projection freshness SLA gate: hourly
        self._scheduler.add_job(
            self._check_projection_freshness,
            IntervalTrigger(hours=1),
            id="projection_freshness",
            name="Projection Freshness Check",
            replace_existing=True,
        )

        # Player ID mapping sync: 7:00 AM ET (before probable_pitchers needs IDs)
        self._scheduler.add_job(
            self._sync_player_id_mapping,
            CronTrigger(hour=7, minute=0, timezone=tz),
            id="player_id_mapping",
            name="Player ID Mapping Sync",
            replace_existing=True,
        )

        # Position eligibility sync: 7:15 AM ET (after player_id_mapping)
        self._scheduler.add_job(
            self._sync_position_eligibility,
            CronTrigger(hour=7, minute=15, timezone=tz),
            id="position_eligibility",
            name="Position Eligibility Sync",
            replace_existing=True,
        )

        # Probable pitchers sync: 8:30 AM, 4:00 PM, 8:00 PM ET
        # Pitchers announced at varying times throughout the day
        self._scheduler.add_job(
            self._sync_probable_pitchers,
            CronTrigger(hour=8, minute=30, timezone=tz),
            id="probable_pitchers_morning",
            name="Probable Pitchers Sync (Morning)",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._sync_probable_pitchers,
            CronTrigger(hour=16, minute=0, timezone=tz),
            id="probable_pitchers_afternoon",
            name="Probable Pitchers Sync (Afternoon)",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._sync_probable_pitchers,
            CronTrigger(hour=20, minute=0, timezone=tz),
            id="probable_pitchers_evening",
            name="Probable Pitchers Sync (Evening)",
            replace_existing=True,
        )

        # Initialise status dict so get_status() never returns missing keys
        _all_job_ids = ["mlb_game_log", "mlb_box_stats", "rolling_windows", "player_scores",
                        "vorp", "player_momentum", "ros_simulation", "decision_optimization",
                        "backtesting", "explainability", "snapshot",
                        "mlb_odds", "statcast",
                        "rolling_z", "clv", "cleanup", "fangraphs_ros", "yahoo_adp_injury",
                        "ensemble_update", "projection_freshness",
                        "player_id_mapping", "position_eligibility",
                        "probable_pitchers_morning", "probable_pitchers_afternoon", "probable_pitchers_evening"]
        if _fantasy_leagues:
            _all_job_ids.append("valuation_cache")
        for job_id in _all_job_ids:
            if job_id not in self._job_status:
                self._job_status[job_id] = {
                    "name": job_id,
                    "enabled": True,
                    "last_run": None,
                    "last_status": None,
                    "next_run": None,
                }

        self._scheduler.start()
        # Populate next_run now that jobs are scheduled
        for job_id in _all_job_ids:
            self._job_status[job_id]["next_run"] = self._get_next_run(job_id)

    def get_status(self) -> dict:
        """Return per-job status dict for /admin/ingestion/status."""
        # Refresh next_run on every call
        for job_id in list(self._job_status.keys()):
            self._job_status[job_id]["next_run"] = self._get_next_run(job_id)
        return dict(self._job_status)

    async def run_job(self, job_id: str) -> dict:
        """
        Manually execute a single ingestion job by ID.

        Supported IDs (pipeline order):
          mlb_game_log -> mlb_box_stats -> rolling_windows -> player_scores
          -> player_momentum -> ros_simulation -> decision_optimization

        Returns the job's result dict.  Raises ValueError for unknown job_id.
        """
        _handlers = {
            "mlb_game_log":    self._ingest_mlb_game_log,
            "mlb_box_stats":   self._ingest_mlb_box_stats,
            "statcast":        self._update_statcast,
            "rolling_windows": self._compute_rolling_windows,
            "player_scores":   self._compute_player_scores,
            "vorp":            self._compute_vorp,
            "player_momentum": self._compute_player_momentum,
            "ros_simulation":        self._run_ros_simulation,
            "decision_optimization": self._run_decision_optimization,
            "backtesting":           self._run_backtesting,
            "explainability":        self._run_explainability,
            "snapshot":              self._run_snapshot,
            "player_id_mapping":     self._sync_player_id_mapping,
            "position_eligibility":  self._sync_position_eligibility,
            "probable_pitchers_morning":   self._sync_probable_pitchers,
            "probable_pitchers_afternoon": self._sync_probable_pitchers,
            "probable_pitchers_evening":   self._sync_probable_pitchers,
        }
        handler = _handlers.get(job_id)
        if handler is None:
            raise ValueError(f"Unknown job_id: {job_id!r}. Valid: {sorted(_handlers)}")
        return await handler()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_run(self, job_id: str) -> Optional[str]:
        """Return ISO next-run string for a scheduled job, or None."""
        try:
            job = self._scheduler.get_job(job_id)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        except Exception:
            pass
        return None

    def _record_job_run(self, job_id: str, status: str, records: int = 0, elapsed_ms: Optional[int] = None) -> None:
        """Update in-memory job status after a run."""
        run_info = {
            "name": job_id,
            "enabled": True,
            "last_run": now_et().isoformat(),
            "last_status": status,
            "next_run": self._get_next_run(job_id),
            "records_processed": records,
            "last_elapsed_ms": elapsed_ms,
        }
        self._job_status[job_id] = run_info

        # Propagate probable_pitchers status to all schedule variants so the
        # morning/afternoon/evening entries don't stay stale (None/None).
        # _sync_probable_pitchers always records to "probable_pitchers" regardless
        # of which variant triggered it; this keeps the three variant keys in sync.
        if job_id == "probable_pitchers":
            for variant in (
                "probable_pitchers_morning",
                "probable_pitchers_afternoon",
                "probable_pitchers_evening",
            ):
                if variant in self._job_status:
                    self._job_status[variant]["last_run"] = run_info["last_run"]
                    self._job_status[variant]["last_status"] = run_info["last_status"]

    # ------------------------------------------------------------------
    # Job handlers
    # ------------------------------------------------------------------

    async def _poll_mlb_odds(self) -> dict:
        """
        Fetch MLB spread odds via BallDontLie GOAT client (lock 100_001).

        Every poll (every 5 min):
          1. get_mlb_games(today) -> list[MLBGame]
          2. For each game: ensure mlb_team + mlb_game_log rows exist (idempotent upsert)
          3. For each game: get_mlb_odds(game_id) -> list[MLBBettingOdd]
          4. Upsert each odd into mlb_odds_snapshot on (game_id, vendor, snapshot_window)

        snapshot_window = now_et() rounded DOWN to the 30-min bucket.
        This makes each poll idempotent — re-running within the same 30-min window
        updates the row rather than inserting a duplicate.

        All data is Pydantic-validated. Typed attribute access only. No dict.get().
        raw_payload = odd.model_dump() on every upsert (dual-write).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("_poll_mlb_odds: BDL init failed -- %s", exc)
                self._record_job_run("mlb_odds", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            date_str = today_et().isoformat()
            games = await asyncio.to_thread(bdl.get_mlb_games, date_str)

            if not games:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info("_poll_mlb_odds: 0 games on %s (BDL)", date_str)
                self._record_job_run("mlb_odds", "success", 0)
                return {"status": "success", "records": 0, "elapsed_ms": elapsed}

            # snapshot_window: round current time DOWN to 30-min bucket (idempotent key)
            now = now_et()
            snapshot_window = now.replace(
                minute=(now.minute // 30) * 30, second=0, microsecond=0
            )

            total_odds = 0
            games_with_odds = 0
            db = SessionLocal()
            try:
                for game in games:
                    # Step 1: ensure dimension + fact rows exist before odds FK write
                    for team in (game.home_team, game.away_team):
                        stmt = pg_insert(MLBTeam.__table__).values(
                            team_id=team.id,
                            abbreviation=team.abbreviation,
                            name=team.name,
                            display_name=team.display_name,
                            short_name=team.short_display_name,
                            location=team.location,
                            slug=team.slug,
                            league=team.league,
                            division=team.division,
                            ingested_at=now,
                        ).on_conflict_do_update(
                            index_elements=["team_id"],
                            set_=dict(
                                abbreviation=team.abbreviation,
                                name=team.name,
                                display_name=team.display_name,
                                short_name=team.short_display_name,
                                location=team.location,
                                slug=team.slug,
                                league=team.league,
                                division=team.division,
                            ),
                        )
                        db.execute(stmt)

                    dt_utc = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
                    game_date_et = dt_utc.astimezone(ZoneInfo("America/New_York")).date()
                    is_active = game.status in {"STATUS_FINAL", "STATUS_IN_PROGRESS"}

                    game_stmt = pg_insert(MLBGameLog.__table__).values(
                        game_id=game.id,
                        game_date=game_date_et,
                        season=game.season,
                        season_type=game.season_type,
                        status=game.status,
                        home_team_id=game.home_team.id,
                        away_team_id=game.away_team.id,
                        home_runs=game.home_team_data.runs if is_active else None,
                        away_runs=game.away_team_data.runs if is_active else None,
                        home_hits=game.home_team_data.hits if is_active else None,
                        away_hits=game.away_team_data.hits if is_active else None,
                        home_errors=game.home_team_data.errors if is_active else None,
                        away_errors=game.away_team_data.errors if is_active else None,
                        venue=game.venue,
                        attendance=(game.attendance or None) if is_active else None,
                        period=game.period,
                        raw_payload=game.model_dump(),
                        ingested_at=now,
                        updated_at=now,
                    ).on_conflict_do_update(
                        index_elements=["game_id"],
                        set_=dict(
                            status=game.status,
                            home_runs=game.home_team_data.runs if is_active else None,
                            away_runs=game.away_team_data.runs if is_active else None,
                            home_hits=game.home_team_data.hits if is_active else None,
                            away_hits=game.away_team_data.hits if is_active else None,
                            home_errors=game.home_team_data.errors if is_active else None,
                            away_errors=game.away_team_data.errors if is_active else None,
                            attendance=(game.attendance or None) if is_active else None,
                            period=game.period,
                            raw_payload=game.model_dump(),
                            updated_at=now,
                        ),
                    )
                    db.execute(game_stmt)

                    # Step 2: fetch and persist odds
                    odds = await asyncio.to_thread(bdl.get_mlb_odds, game.id)
                    if not odds:
                        continue

                    games_with_odds += 1
                    for odd in odds:
                        # Skip books that lack a full spread+total line.
                        # MLBOddsSnapshot columns are NOT NULL by design; the moneyline
                        # is still captured in raw_payload via other odds rows for the game.
                        if not (odd.has_spread and odd.has_total):
                            logger.debug(
                                "skip mlb odds row: game=%s vendor=%s missing spread/total",
                                odd.game_id, odd.vendor,
                            )
                            continue
                        payload = odd.model_dump()
                        stmt = pg_insert(MLBOddsSnapshot.__table__).values(
                            odds_id=odd.id,
                            game_id=odd.game_id,
                            vendor=odd.vendor,
                            snapshot_window=snapshot_window,
                            spread_home=odd.spread_home_value,
                            spread_away=odd.spread_away_value,
                            spread_home_odds=odd.spread_home_odds,
                            spread_away_odds=odd.spread_away_odds,
                            ml_home_odds=odd.moneyline_home_odds,
                            ml_away_odds=odd.moneyline_away_odds,
                            total=odd.total_value,
                            total_over_odds=odd.total_over_odds,
                            total_under_odds=odd.total_under_odds,
                            raw_payload=payload,
                        ).on_conflict_do_update(
                            index_elements=["game_id", "vendor", "snapshot_window"],
                            set_=dict(
                                odds_id=odd.id,
                                spread_home=odd.spread_home_value,
                                spread_away=odd.spread_away_value,
                                spread_home_odds=odd.spread_home_odds,
                                spread_away_odds=odd.spread_away_odds,
                                ml_home_odds=odd.moneyline_home_odds,
                                ml_away_odds=odd.moneyline_away_odds,
                                total=odd.total_value,
                                total_over_odds=odd.total_over_odds,
                                total_under_odds=odd.total_under_odds,
                                raw_payload=payload,
                            ),
                        )
                        db.execute(stmt)
                        total_odds += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_poll_mlb_odds DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_odds", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "_poll_mlb_odds: %d games, %d with odds, %d snapshots in %dms (window=%s)",
                len(games), games_with_odds, total_odds, elapsed,
                snapshot_window.strftime("%H:%M"),
            )
            self._record_job_run("mlb_odds", "success", total_odds)
            return {
                "status": "success",
                "records": total_odds,
                "games": len(games),
                "games_with_odds": games_with_odds,
                "snapshot_window": snapshot_window.isoformat(),
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_odds"], "mlb_odds", _run)

    async def _ingest_mlb_game_log(self) -> dict:
        """
        Daily MLB game-log ingestion (lock 100_016).

        Fetches two dates per run:
          - yesterday_et(): finalize scores for games that finished overnight
          - today_et():     seed today's scheduled games

        For each game:
          1. Upsert mlb_team (home + away) -- dimension before fact, FK dependency
          2. Upsert mlb_game_log on game_id:
             - Immutable: game_date, season, team_ids, venue, ingested_at
             - Updated:   status, scores, attendance, period, raw_payload, updated_at

        Scores (home_runs, away_runs, hits, errors) written only when game has started
        (STATUS_IN_PROGRESS or STATUS_FINAL). Pre-game rows store NULL for score fields.

        Anomaly check: logs WARNING if BDL returns 0 games for a date.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("mlb_game_log: BDL init failed -- %s", exc)
                self._record_job_run("mlb_game_log", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            today = today_et()
            yesterday = today - timedelta(days=1)
            dates = [yesterday.isoformat(), today.isoformat()]

            total_games = 0
            db = SessionLocal()
            try:
                for date_str in dates:
                    games = await asyncio.to_thread(bdl.get_mlb_games, date_str)

                    if not games:
                        logger.warning(
                            "mlb_game_log: 0 games returned for %s -- off-day or BDL error",
                            date_str,
                        )
                        continue

                    for game in games:
                        # Step 1: upsert both teams (dimension before fact -- FK dependency)
                        for team in (game.home_team, game.away_team):
                            stmt = pg_insert(MLBTeam.__table__).values(
                                team_id=team.id,
                                abbreviation=team.abbreviation,
                                name=team.name,
                                display_name=team.display_name,
                                short_name=team.short_display_name,
                                location=team.location,
                                slug=team.slug,
                                league=team.league,
                                division=team.division,
                                ingested_at=now_et(),
                            ).on_conflict_do_update(
                                index_elements=["team_id"],
                                set_=dict(
                                    abbreviation=team.abbreviation,
                                    name=team.name,
                                    display_name=team.display_name,
                                    short_name=team.short_display_name,
                                    location=team.location,
                                    slug=team.slug,
                                    league=team.league,
                                    division=team.division,
                                ),
                            )
                            db.execute(stmt)

                        # Step 2: convert UTC ISO 8601 game timestamp to ET date
                        dt_utc = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
                        game_date_et = dt_utc.astimezone(ZoneInfo("America/New_York")).date()

                        # Scores are meaningful only when game has started
                        is_active = game.status in {"STATUS_FINAL", "STATUS_IN_PROGRESS"}
                        home_runs   = game.home_team_data.runs   if is_active else None
                        away_runs   = game.away_team_data.runs   if is_active else None
                        home_hits   = game.home_team_data.hits   if is_active else None
                        away_hits   = game.away_team_data.hits   if is_active else None
                        home_errors = game.home_team_data.errors if is_active else None
                        away_errors = game.away_team_data.errors if is_active else None
                        attendance  = (game.attendance or None)  if is_active else None

                        # Step 3: upsert game log -- idempotent on game_id
                        now = now_et()
                        payload = game.model_dump()
                        stmt = pg_insert(MLBGameLog.__table__).values(
                            game_id=game.id,
                            game_date=game_date_et,
                            season=game.season,
                            season_type=game.season_type,
                            status=game.status,
                            home_team_id=game.home_team.id,
                            away_team_id=game.away_team.id,
                            home_runs=home_runs,
                            away_runs=away_runs,
                            home_hits=home_hits,
                            away_hits=away_hits,
                            home_errors=home_errors,
                            away_errors=away_errors,
                            venue=game.venue,
                            attendance=attendance,
                            period=game.period,
                            raw_payload=payload,
                            ingested_at=now,
                            updated_at=now,
                        ).on_conflict_do_update(
                            index_elements=["game_id"],
                            set_=dict(
                                status=game.status,
                                home_runs=home_runs,
                                away_runs=away_runs,
                                home_hits=home_hits,
                                away_hits=away_hits,
                                home_errors=home_errors,
                                away_errors=away_errors,
                                attendance=attendance,
                                period=game.period,
                                raw_payload=payload,
                                updated_at=now,
                            ),
                        )
                        db.execute(stmt)
                        total_games += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("mlb_game_log DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_game_log", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "mlb_game_log: %d games upserted across %s in %dms",
                total_games, dates, elapsed,
            )
            self._record_job_run("mlb_game_log", "success", total_games)
            return {
                "status": "success",
                "records": total_games,
                "dates": dates,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_game_log"], "mlb_game_log", _run)

    async def _ingest_mlb_box_stats(self) -> dict:
        """
        Daily MLB player box stats ingestion (lock 100_017).

        Runs at 2 AM ET -- 1 hour after game-log (1 AM) so mlb_game_log rows
        are present when we write the FK game_id into mlb_player_stats.

        For each stat row:
          - Validates via MLBPlayerStats Pydantic V2 contract (validation failures
            are logged at WARNING and skipped -- never kills the job)
          - Upserts into mlb_player_stats on (bdl_player_id, game_id)
          - Dual-write: raw_payload = stat.model_dump()

        Anomaly: if BDL returns 0 stat rows for dates that had scheduled games,
        logs WARNING (network blip / off-day -- does not raise).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("mlb_box_stats: BDL init failed -- %s", exc)
                self._record_job_run("mlb_box_stats", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            today = today_et()
            yesterday = today - timedelta(days=1)
            date_strs = [yesterday.isoformat(), today.isoformat()]

            # Fetch game IDs from our log for these dates. BDL /stats dates filter
            # is unreliable (returns historical data), so we filter by specific game IDs.
            db = SessionLocal()
            try:
                game_ids = [
                    r[0] for r in db.query(MLBGameLog.game_id).filter(
                        MLBGameLog.game_date.in_([yesterday, today])
                    ).all()
                ]
            finally:
                db.close()

            if not game_ids:
                logger.info("mlb_box_stats: no games in log for %s -- skipping stats", date_strs)
                self._record_job_run("mlb_box_stats", "success", 0)
                elapsed = int((time.monotonic() - t0) * 1000)
                return {"status": "success", "records": 0, "dates": date_strs, "elapsed_ms": elapsed}

            # Fetch stats for these specific games
            stats = await asyncio.to_thread(bdl.get_mlb_stats, game_ids=game_ids)

            if not stats:
                logger.warning(
                    "mlb_box_stats: 0 stat rows returned for %d games on %s",
                    len(game_ids), date_strs,
                )
                self._record_job_run("mlb_box_stats", "success", 0)
                elapsed = int((time.monotonic() - t0) * 1000)
                return {"status": "success", "records": 0, "dates": date_strs, "elapsed_ms": elapsed}

            now = now_et()
            rows_upserted = 0
            db = SessionLocal()
            try:
                # Group stats by game for date lookup
                game_date_map = {
                    g.game_id: g.game_date 
                    for g in db.query(MLBGameLog).filter(MLBGameLog.game_id.in_(game_ids)).all()
                }

                for stat in stats:
                    # Skip rows with no player identity -- bdl_player_id is NOT NULL in schema.
                    if stat.bdl_player_id is None:
                        logger.warning("mlb_box_stats: skipping stat row with no player object")
                        continue

                    # Validate stat row before insertion
                    if not _validate_mlb_stats(stat):
                        continue  # Skip this row

                    game_date = game_date_map.get(stat.game_id, today)

                    # Compute OPS from OBP + SLG (BDL doesn't provide it)
                    computed_ops = None
                    if stat.obp is not None and stat.slg is not None:
                        computed_ops = stat.obp + stat.slg
                        logger.debug(
                            "mlb_box_stats: computed OPS for player %d in game %d: %.3f = %.3f + %.3f",
                            stat.bdl_player_id, stat.game_id, computed_ops, stat.obp, stat.slg
                        )

                    # Compute WHIP from (BB + H) / IP (BDL doesn't provide it)
                    computed_whip = None
                    if (stat.bb_allowed is not None and
                        stat.h_allowed is not None and
                        stat.ip is not None):
                        ip_decimal = _parse_innings_pitched(stat.ip)
                        if ip_decimal is not None and ip_decimal > 0:
                            computed_whip = (stat.bb_allowed + stat.h_allowed) / ip_decimal
                            logger.debug(
                                "mlb_box_stats: computed WHIP for player %d in game %d: %.3f = (%d + %d) / %.1f",
                                stat.bdl_player_id, stat.game_id, computed_whip,
                                stat.bb_allowed, stat.h_allowed, ip_decimal
                            )

                    # Default caught_stealing to 0 when BDL doesn't provide it
                    computed_cs = stat.cs if stat.cs is not None else 0

                    # Validate ERA is within reasonable range (0-100)
                    validated_era = stat.era
                    if validated_era is not None and (validated_era < 0 or validated_era > 100):
                        logger.warning(
                            "mlb_box_stats: Impossible ERA %s for player %s (ER=%s, IP=%s) - skipping ERA",
                            validated_era, stat.bdl_player_id, stat.er, stat.ip
                        )
                        validated_era = None  # Don't store impossible values

                    payload = stat.model_dump()
                    stmt = pg_insert(MLBPlayerStats.__table__).values(
                        bdl_stat_id=stat.id,
                        bdl_player_id=stat.bdl_player_id,
                        game_id=stat.game_id,
                        game_date=game_date,
                        season=stat.season if stat.season is not None else 2026,
                        # Batting
                        ab=stat.ab,
                        runs=stat.r,
                        hits=stat.h,
                        doubles=stat.double,
                        triples=stat.triple,
                        home_runs=stat.hr,
                        rbi=stat.rbi,
                        walks=stat.bb,
                        strikeouts_bat=stat.so,
                        stolen_bases=stat.sb,
                        caught_stealing=computed_cs,
                        avg=stat.avg,
                        obp=stat.obp,
                        slg=stat.slg,
                        ops=computed_ops,
                        # Pitching
                        innings_pitched=stat.ip,
                        hits_allowed=stat.h_allowed,
                        runs_allowed=stat.r_allowed,
                        earned_runs=stat.er,
                        walks_allowed=stat.bb_allowed,
                        strikeouts_pit=stat.k,
                        whip=computed_whip,
                        era=validated_era,
                        # Audit
                        raw_payload=payload,
                        ingested_at=now,
                    ).on_conflict_do_update(
                        constraint="_mps_player_game_uc",
                        set_=dict(
                            bdl_stat_id=stat.id,
                            season=stat.season if stat.season is not None else 2026,
                            ab=stat.ab,
                            runs=stat.r,
                            hits=stat.h,
                            doubles=stat.double,
                            triples=stat.triple,
                            home_runs=stat.hr,
                            rbi=stat.rbi,
                            walks=stat.bb,
                            strikeouts_bat=stat.so,
                            stolen_bases=stat.sb,
                            caught_stealing=computed_cs,
                            avg=stat.avg,
                            obp=stat.obp,
                            slg=stat.slg,
                            ops=computed_ops,
                            innings_pitched=stat.ip,
                            hits_allowed=stat.h_allowed,
                            runs_allowed=stat.r_allowed,
                            earned_runs=stat.er,
                            walks_allowed=stat.bb_allowed,
                            strikeouts_pit=stat.k,
                            whip=computed_whip,
                            era=validated_era,
                            raw_payload=payload,
                        ),
                    )
                    try:
                        with db.begin_nested():
                            db.execute(stmt)
                            rows_upserted += 1
                    except Exception as exc:
                        if "violates foreign key constraint" in str(exc).lower():
                            logger.warning(
                                "mlb_box_stats: skipping stat row for player_id=%d game_id=%s -- game not in log",
                                stat.bdl_player_id, stat.game_id
                            )
                        else:
                            logger.error(
                                "mlb_box_stats: failed to upsert row for player_id=%d game_id=%s: %s",
                                stat.bdl_player_id, stat.game_id, exc
                            )
                        continue

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("mlb_box_stats DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_box_stats", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "mlb_box_stats: %d rows upserted for dates=%s in %dms",
                rows_upserted, date_strs, elapsed,
            )
            self._record_job_run("mlb_box_stats", "success", rows_upserted)
            return {
                "status": "success",
                "records": rows_upserted,
                "dates": date_strs,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_box_stats"], "mlb_box_stats", _run)

    async def _supplement_statsapi_counting_stats(self) -> dict:
        """
        Supplement BDL per-game stats with counting stats from MLB Stats API
        (lock 100_026, 2:30 AM ET).

        BDL /mlb/v1/stats returns NULL for critical batting counting stats
        (ab, h, r, doubles, triples, so, sb, cs) while rate stats (avg, obp,
        slg) are populated. This job fills the gaps using statsapi.boxscore_data()
        which returns complete per-player box scores from the MLB Stats API.

        Matching strategy:
          1. Find mlb_player_stats rows for yesterday+today where ab IS NULL
          2. For each date, get MLB schedule (statsapi.schedule)
          3. For each game, get box score (statsapi.boxscore_data)
          4. Match players by full_name (BDL raw_payload) -> fullName (playerInfo)
          5. UPDATE counting stats where currently NULL

        The match is done within a single game_date, so name collisions are
        extremely unlikely (no two players with same full name play in the
        same MLB game).
        """
        t0 = time.monotonic()

        async def _run():
            try:
                import statsapi
            except ImportError:
                logger.error("statsapi_supplement: MLB-StatsAPI not installed")
                self._record_job_run("statsapi_supplement", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            today = today_et()
            yesterday = today - timedelta(days=1)
            target_dates = [yesterday, today]

            db = SessionLocal()
            try:
                # Find rows needing supplementation (ab IS NULL = BDL didn't provide counting stats)
                rows_needing_fill = (
                    db.query(MLBPlayerStats)
                    .filter(
                        MLBPlayerStats.game_date.in_(target_dates),
                        MLBPlayerStats.ab.is_(None),
                    )
                    .all()
                )

                if not rows_needing_fill:
                    elapsed = int((time.monotonic() - t0) * 1000)
                    logger.info("statsapi_supplement: no rows needing fill for %s", target_dates)
                    self._record_job_run("statsapi_supplement", "success", 0)
                    return {"status": "success", "records": 0, "elapsed_ms": elapsed}

                # Build lookup: (game_date, normalized_name) -> list of DB rows
                # Strip diacriticals so "José Ramírez" (statsapi) matches "Jose Ramirez" (BDL)
                def _norm(name: str) -> str:
                    s = unicodedata.normalize("NFD", name)
                    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip().lower()

                name_lookup: dict[tuple, list] = {}
                for row in rows_needing_fill:
                    payload = row.raw_payload if isinstance(row.raw_payload, dict) else {}
                    player_obj = payload.get("player", {})
                    full_name = _norm(player_obj.get("full_name") or "")
                    if full_name:
                        key = (row.game_date, full_name)
                        name_lookup.setdefault(key, []).append(row)

                rows_patched = 0

                for target_date in target_dates:
                    date_str = target_date.strftime("%m/%d/%Y")
                    try:
                        games = await asyncio.to_thread(
                            statsapi.schedule, sportId=1, date=date_str
                        )
                    except Exception as exc:
                        logger.warning("statsapi_supplement: schedule fetch failed for %s: %s", date_str, exc)
                        continue

                    for game in games:
                        game_pk = game.get("game_id")
                        if not game_pk:
                            continue
                        game_status = game.get("status", "")
                        if game_status not in ("Final", "Game Over", "Completed Early"):
                            continue

                        try:
                            box = await asyncio.to_thread(statsapi.boxscore_data, game_pk)
                        except Exception as exc:
                            logger.warning(
                                "statsapi_supplement: boxscore_data(%s) failed: %s", game_pk, exc
                            )
                            continue

                        player_info = box.get("playerInfo", {})

                        # Process batters and pitchers from both sides
                        for side in ("away", "home"):
                            batter_list = box.get(f"{side}Batters", [])
                            pitcher_list = box.get(f"{side}Pitchers", [])

                            for batter in batter_list:
                                person_id = batter.get("personId")
                                if not person_id:
                                    continue
                                info = player_info.get(f"ID{person_id}", {})
                                statsapi_name = _norm(info.get("fullName") or "")
                                if not statsapi_name:
                                    continue

                                key = (target_date, statsapi_name)
                                matching_rows = name_lookup.get(key, [])
                                for db_row in matching_rows:
                                    patched = self._patch_counting_stats_batter(db_row, batter)
                                    if patched:
                                        rows_patched += 1

                            for pitcher in pitcher_list:
                                person_id = pitcher.get("personId")
                                if not person_id:
                                    continue
                                info = player_info.get(f"ID{person_id}", {})
                                statsapi_name = _norm(info.get("fullName") or "")
                                if not statsapi_name:
                                    continue

                                key = (target_date, statsapi_name)
                                matching_rows = name_lookup.get(key, [])
                                for db_row in matching_rows:
                                    patched = self._patch_counting_stats_pitcher(db_row, pitcher)
                                    if patched:
                                        rows_patched += 1

                        # Rate limit: be polite to MLB Stats API
                        await asyncio.sleep(0.15)

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("statsapi_supplement DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("statsapi_supplement", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "statsapi_supplement: %d rows patched for dates=%s in %dms",
                rows_patched, [d.isoformat() for d in target_dates], elapsed,
            )
            self._record_job_run("statsapi_supplement", "success", rows_patched)
            return {
                "status": "success",
                "records": rows_patched,
                "dates": [d.isoformat() for d in target_dates],
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["statsapi_supplement"], "statsapi_supplement", _run)

    @staticmethod
    def _patch_counting_stats_batter(db_row, batter: dict) -> bool:
        """Patch NULL batting counting stats from statsapi box score.
        Returns True if any field was updated."""
        patched = False
        field_map = {
            "ab": "ab",
            "runs": "r",
            "hits": "h",
            "doubles": "doubles",
            "triples": "triples",
            "home_runs": "hr",
            "rbi": "rbi",
            "walks": "bb",
            "strikeouts_bat": "k",
            "stolen_bases": "sb",
            "caught_stealing": "cs",
        }
        for db_col, box_key in field_map.items():
            if getattr(db_row, db_col) is None:
                raw_val = batter.get(box_key)
                if raw_val is not None:
                    try:
                        setattr(db_row, db_col, int(raw_val))
                        patched = True
                    except (ValueError, TypeError):
                        pass
        return patched

    @staticmethod
    def _patch_counting_stats_pitcher(db_row, pitcher: dict) -> bool:
        """Patch NULL pitching counting stats from statsapi box score.
        Returns True if any field was updated."""
        patched = False
        field_map = {
            "hits_allowed": "h",
            "earned_runs": "er",
            "walks_allowed": "bb",
            "strikeouts_pit": "k",
            "runs_allowed": "r",
        }
        for db_col, box_key in field_map.items():
            if getattr(db_row, db_col) is None:
                raw_val = pitcher.get(box_key)
                if raw_val is not None:
                    try:
                        setattr(db_row, db_col, int(raw_val))
                        patched = True
                    except (ValueError, TypeError):
                        pass
        # IP special handling — statsapi returns "5.2" as string
        if db_row.innings_pitched is None:
            ip_val = pitcher.get("ip")
            if ip_val is not None:
                db_row.innings_pitched = str(ip_val)
                patched = True
        return patched

    async def _compute_rolling_windows(self) -> dict:
        """
        Daily rolling window computation (lock 100_018, 3 AM ET).

        Runs after _ingest_mlb_box_stats (2 AM) so today's box stats are present.

        Algorithm:
          1. Query all mlb_player_stats rows for the past 30 days (max window)
          2. Compute 7/14/30-day decay-weighted windows for every player with data
          3. Upsert to player_rolling_stats on (bdl_player_id, as_of_date, window_days)

        Anomaly: logs WARNING if 0 players processed (likely off-day or box stats missing).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.rolling_window_engine import compute_all_rolling_windows

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            lookback_start = as_of_date - timedelta(days=30)

            db = SessionLocal()
            try:
                rows = (
                    db.query(MLBPlayerStats)
                    .filter(
                        MLBPlayerStats.game_date >= lookback_start,
                        MLBPlayerStats.game_date <= as_of_date,
                    )
                    .all()
                )
            except Exception as exc:
                db.close()
                logger.error("rolling_windows: DB query failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            if not rows:
                logger.warning(
                    "rolling_windows: 0 stat rows found for window %s..%s -- off-day or box stats missing",
                    lookback_start, as_of_date,
                )
                db.close()
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "success", 0)
                return {
                    "status": "success",
                    "as_of_date": str(as_of_date),
                    "players_processed": 0,
                    "rows_upserted": 0,
                    "elapsed_ms": elapsed,
                }

            results = compute_all_rolling_windows(
                rows,
                as_of_date=as_of_date,
                window_sizes=[7, 14, 30],
            )

            players_processed = len({r.bdl_player_id for r in results})

            if players_processed == 0:
                logger.warning(
                    "rolling_windows: compute_all_rolling_windows returned 0 results for as_of_date=%s",
                    as_of_date,
                )

            now = datetime.now(ZoneInfo("America/New_York"))
            rows_upserted = 0
            try:
                for res in results:
                    stmt = pg_insert(PlayerRollingStats.__table__).values(
                        bdl_player_id=res.bdl_player_id,
                        as_of_date=res.as_of_date,
                        window_days=res.window_days,
                        games_in_window=res.games_in_window,
                        w_games=res.w_games,
                        w_ab=res.w_ab,
                        w_hits=res.w_hits,
                        w_doubles=res.w_doubles,
                        w_triples=res.w_triples,
                        w_home_runs=res.w_home_runs,
                        w_rbi=res.w_rbi,
                        w_walks=res.w_walks,
                        w_strikeouts_bat=res.w_strikeouts_bat,
                        w_stolen_bases=res.w_stolen_bases,
                        w_caught_stealing=res.w_caught_stealing,
                        w_net_stolen_bases=res.w_net_stolen_bases,
                        w_avg=res.w_avg,
                        w_obp=res.w_obp,
                        w_slg=res.w_slg,
                        w_ops=res.w_ops,
                        w_ip=res.w_ip,
                        w_earned_runs=res.w_earned_runs,
                        w_hits_allowed=res.w_hits_allowed,
                        w_walks_allowed=res.w_walks_allowed,
                        w_strikeouts_pit=res.w_strikeouts_pit,
                        w_era=res.w_era,
                        w_whip=res.w_whip,
                        w_k_per_9=res.w_k_per_9,
                        computed_at=now,
                    ).on_conflict_do_update(
                        constraint="_prs_player_date_window_uc",
                        set_=dict(
                            games_in_window=res.games_in_window,
                            w_games=res.w_games,
                            w_ab=res.w_ab,
                            w_hits=res.w_hits,
                            w_doubles=res.w_doubles,
                            w_triples=res.w_triples,
                            w_home_runs=res.w_home_runs,
                            w_rbi=res.w_rbi,
                            w_walks=res.w_walks,
                            w_strikeouts_bat=res.w_strikeouts_bat,
                            w_stolen_bases=res.w_stolen_bases,
                            w_caught_stealing=res.w_caught_stealing,
                            w_net_stolen_bases=res.w_net_stolen_bases,
                            w_avg=res.w_avg,
                            w_obp=res.w_obp,
                            w_slg=res.w_slg,
                            w_ops=res.w_ops,
                            w_ip=res.w_ip,
                            w_earned_runs=res.w_earned_runs,
                            w_hits_allowed=res.w_hits_allowed,
                            w_walks_allowed=res.w_walks_allowed,
                            w_strikeouts_pit=res.w_strikeouts_pit,
                            w_era=res.w_era,
                            w_whip=res.w_whip,
                            w_k_per_9=res.w_k_per_9,
                            computed_at=now,
                        ),
                    )
                    db.execute(stmt)
                    rows_upserted += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("rolling_windows: DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "rolling_windows: %d rows upserted for %d players, as_of_date=%s in %dms",
                rows_upserted, players_processed, as_of_date, elapsed,
            )
            self._record_job_run("rolling_windows", "success", rows_upserted)
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "players_processed": players_processed,
                "rows_upserted": rows_upserted,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["rolling_windows"], "rolling_windows", _run)

    async def _compute_player_scores(self) -> dict:
        """
        Daily Z-score scoring computation (lock 100_019, 4 AM ET).

        Runs after _compute_rolling_windows (3 AM) so player_rolling_stats is current.

        Algorithm:
          1. For each window_days in [7, 14, 30]:
             a. Query all player_rolling_stats WHERE as_of_date = yesterday AND window_days = N
             b. Call compute_league_zscores(rows, yesterday, N)
             c. Upsert each PlayerScoreResult to player_scores table
          2. Anomaly: WARN if 0 players scored for any window
          3. Return scored counts per window

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.scoring_engine import compute_league_zscores, compute_league_params

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

            scored_7d = 0
            scored_14d = 0
            scored_30d = 0

            db = SessionLocal()
            try:
                now = datetime.now(ZoneInfo("America/New_York"))

                for window_days in [7, 14, 30]:
                    try:
                        rows = (
                            db.query(PlayerRollingStats)
                            .filter(
                                PlayerRollingStats.as_of_date == as_of_date,
                                PlayerRollingStats.window_days == window_days,
                            )
                            .all()
                        )
                    except Exception as exc:
                        logger.error(
                            "player_scores: DB query failed for window_days=%d: %s",
                            window_days, exc,
                        )
                        elapsed = int((time.monotonic() - t0) * 1000)
                        self._record_job_run("player_scores", "failed")
                        return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                    if not rows:
                        logger.warning(
                            "player_scores: 0 rolling_stats rows found for as_of_date=%s "
                            "window_days=%d -- off-day or rolling windows missing",
                            as_of_date, window_days,
                        )

                    score_results = compute_league_zscores(rows, as_of_date, window_days)

                    # H2 fix: capture league-level means/stds from the 14d window
                    # for downstream simulation composite risk metrics.
                    if window_days == 14 and rows:
                        self._league_means, self._league_stds = compute_league_params(rows)

                    if not score_results:
                        logger.warning(
                            "player_scores: 0 players scored for as_of_date=%s window_days=%d",
                            as_of_date, window_days,
                        )

                    try:
                        for res in score_results:
                            stmt = pg_insert(PlayerScore.__table__).values(
                                bdl_player_id=res.bdl_player_id,
                                as_of_date=res.as_of_date,
                                window_days=res.window_days,
                                player_type=res.player_type,
                                games_in_window=res.games_in_window,
                                z_hr=res.z_hr,
                                z_rbi=res.z_rbi,
                                z_sb=res.z_sb,
                                z_nsb=res.z_nsb,
                                z_avg=res.z_avg,
                                z_obp=res.z_obp,
                                z_era=res.z_era,
                                z_whip=res.z_whip,
                                z_k_per_9=res.z_k_per_9,
                                composite_z=res.composite_z,
                                score_0_100=res.score_0_100,
                                confidence=res.confidence,
                                computed_at=now,
                            ).on_conflict_do_update(
                                constraint="_ps_player_date_window_uc",
                                set_=dict(
                                    player_type=res.player_type,
                                    games_in_window=res.games_in_window,
                                    z_hr=res.z_hr,
                                    z_rbi=res.z_rbi,
                                    z_sb=res.z_sb,
                                    z_nsb=res.z_nsb,
                                    z_avg=res.z_avg,
                                    z_obp=res.z_obp,
                                    z_era=res.z_era,
                                    z_whip=res.z_whip,
                                    z_k_per_9=res.z_k_per_9,
                                    composite_z=res.composite_z,
                                    score_0_100=res.score_0_100,
                                    confidence=res.confidence,
                                    computed_at=now,
                                ),
                            )
                            db.execute(stmt)

                        db.commit()
                    except Exception as exc:
                        db.rollback()
                        logger.error(
                            "player_scores: DB write failed for window_days=%d: %s",
                            window_days, exc,
                        )
                        elapsed = int((time.monotonic() - t0) * 1000)
                        self._record_job_run("player_scores", "failed")
                        return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                    count = len(score_results)
                    if window_days == 7:
                        scored_7d = count
                    elif window_days == 14:
                        scored_14d = count
                    else:
                        scored_30d = count

                    logger.info(
                        "player_scores: %d players scored for as_of_date=%s window_days=%d",
                        count, as_of_date, window_days,
                    )

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            total = scored_7d + scored_14d + scored_30d
            self._record_job_run("player_scores", "success", total)
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "scored_7d": scored_7d,
                "scored_14d": scored_14d,
                "scored_30d": scored_30d,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["player_scores"], "player_scores", _run)

    async def _compute_vorp(self) -> dict:
        """
        Daily VORP computation (lock 100_030, 4:30 AM ET).

        Runs after _compute_player_scores (4 AM) so player_scores rows are current.

        Algorithm:
          1. Query player_scores for 7d and 30d windows (yesterday's date)
          2. Query position_eligibility to build bdl_player_id -> positions map
          3. Compute VORP = composite_z - replacement_z(scarcest_position)
          4. Upsert vorp_7d/vorp_30d into player_daily_metrics
        """
        logger.info("SYNC JOB ENTRY: _compute_vorp - Starting VORP computation")
        t0 = time.monotonic()

        async def _run():
            from backend.services.vorp_engine import (
                compute_vorp_batch,
                get_eligible_positions,
            )

            as_of_date = today_et() - timedelta(days=1)
            db = SessionLocal()
            vorp_count = 0

            try:
                # Build position map: bdl_player_id -> [positions]
                # and name map: bdl_player_id -> player_name
                pos_rows = db.query(PositionEligibility).filter(
                    PositionEligibility.bdl_player_id.isnot(None)
                ).all()
                position_map: dict[int, list[str]] = {}
                name_map: dict[int, str] = {}
                for pr in pos_rows:
                    position_map[pr.bdl_player_id] = get_eligible_positions(pr)
                    if pr.player_name:
                        name_map[pr.bdl_player_id] = pr.player_name

                logger.info("_compute_vorp: %d players with position data", len(position_map))

                if not position_map:
                    logger.warning("_compute_vorp: No position data found, skipping")
                    self._record_job_run("vorp", "skipped")
                    return {"status": "skipped", "reason": "no_position_data", "elapsed_ms": 0}

                # Process 7d and 30d windows
                for window_days, vorp_col in [(7, "vorp_7d"), (30, "vorp_30d")]:
                    scores = db.query(PlayerScore).filter(
                        PlayerScore.as_of_date == as_of_date,
                        PlayerScore.window_days == window_days,
                    ).all()

                    if not scores:
                        logger.warning(
                            "_compute_vorp: No player_scores for %s window=%dd",
                            as_of_date, window_days,
                        )
                        continue

                    vorp_results = compute_vorp_batch(scores, position_map)
                    logger.info(
                        "_compute_vorp: Computed %d VORP values for %dd window",
                        len(vorp_results), window_days,
                    )

                    # Upsert into player_daily_metrics
                    for bdl_id, vorp_val in vorp_results.items():
                        player_name = name_map.get(bdl_id, f"player_{bdl_id}")

                        stmt = pg_insert(PlayerDailyMetric).values(
                            player_id=str(bdl_id),
                            player_name=player_name,
                            metric_date=as_of_date,
                            sport="mlb",
                            **{vorp_col: vorp_val},
                        )
                        stmt = stmt.on_conflict_do_update(
                            constraint="_pdm_player_date_sport_uc",
                            set_={vorp_col: vorp_val},
                        )
                        db.execute(stmt)
                        vorp_count += 1

                db.commit()
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info(
                    "SYNC JOB SUCCESS: _compute_vorp - %d VORP values in %d ms",
                    vorp_count, elapsed,
                )
                self._record_job_run("vorp", "success", vorp_count)
                return {"status": "success", "records": vorp_count, "elapsed_ms": elapsed}

            except Exception as exc:
                db.rollback()
                logger.error("_compute_vorp: Failed (%s)", exc)
                self._record_job_run("vorp", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}
            finally:
                db.close()

        try:
            return await _with_advisory_lock(LOCK_IDS["vorp"], "vorp", _run)
        except Exception as exc:
            logger.error("_compute_vorp: Job failed (%s)", exc)
            self._record_job_run("vorp", "error", 0)
            return {"status": "error", "records": 0, "elapsed_ms": 0}

    async def _compute_player_momentum(self) -> dict:
        """
        Daily momentum signal computation (lock 100_020, 5 AM ET).

        Runs after _compute_player_scores (4 AM) so player_scores is current.

        Algorithm:
          1. Query player_scores WHERE as_of_date = yesterday AND window_days = 14
          2. Query player_scores WHERE as_of_date = yesterday AND window_days = 30
          3. compute_all_momentum(scores_14d, scores_30d)
          4. Upsert each MomentumResult to player_momentum ON CONFLICT (_pm_player_date_uc)
          5. WARN if 0 results (off-day or scoring pipeline missing)
          6. WARN if any single signal exceeds 60% of total (indicates potential data issue)
          7. Return {"as_of_date": str(yesterday), "total": n, "signals": {signal: count}}

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            from collections import Counter
            from backend.services.momentum_engine import compute_all_momentum

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

            db = SessionLocal()
            results = []
            try:
                now = datetime.now(ZoneInfo("America/New_York"))

                try:
                    scores_14d = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        )
                        .all()
                    )
                    scores_30d = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 30,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "player_momentum: DB query failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("player_momentum", "failed")
                    return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                results = compute_all_momentum(scores_14d, scores_30d)

                if not results:
                    logger.warning(
                        "player_momentum: 0 players computed for as_of_date=%s -- "
                        "off-day or player_scores pipeline missing",
                        as_of_date,
                    )

                # Anomaly: warn if any single signal dominates (>60% of total)
                if results:
                    signal_counts: dict = Counter(r.signal for r in results)
                    total_check = len(results)
                    for sig, cnt in signal_counts.items():
                        if cnt / total_check > 0.60:
                            logger.warning(
                                "player_momentum: signal '%s' is %.1f%% of total (%d/%d) "
                                "for as_of_date=%s -- possible data issue",
                                sig, cnt / total_check * 100, cnt, total_check, as_of_date,
                            )

                try:
                    for res in results:
                        stmt = pg_insert(PlayerMomentum.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            player_type=res.player_type,
                            delta_z=res.delta_z,
                            signal=res.signal,
                            composite_z_14d=res.composite_z_14d,
                            composite_z_30d=res.composite_z_30d,
                            score_14d=res.score_14d,
                            score_30d=res.score_30d,
                            confidence_14d=res.confidence_14d,
                            confidence_30d=res.confidence_30d,
                            confidence=res.confidence,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_pm_player_date_uc",
                            set_=dict(
                                player_type=res.player_type,
                                delta_z=res.delta_z,
                                signal=res.signal,
                                composite_z_14d=res.composite_z_14d,
                                composite_z_30d=res.composite_z_30d,
                                score_14d=res.score_14d,
                                score_30d=res.score_30d,
                                confidence_14d=res.confidence_14d,
                                confidence_30d=res.confidence_30d,
                                confidence=res.confidence,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "player_momentum: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("player_momentum", "failed")
                    return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            total = len(results)
            signal_counts = Counter(r.signal for r in results)
            self._record_job_run("player_momentum", "success", total)
            logger.info(
                "player_momentum: %d players computed for as_of_date=%s signals=%s",
                total, as_of_date, dict(signal_counts),
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "total": total,
                "signals": dict(signal_counts),
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["player_momentum"], "player_momentum", _run)

    async def _run_ros_simulation(self) -> dict:
        """
        Daily Rest-of-Season Monte Carlo simulation (lock 100_021, 6 AM ET).

        Runs after _compute_player_momentum (5 AM) so momentum layer is current.

        Algorithm:
          1. Query player_rolling_stats WHERE as_of_date = yesterday AND window_days = 14
          2. simulate_all_players(rows, remaining_games=REMAINING_GAMES_DEFAULT)
             -> list[SimulationResult dataclass]
          3. Upsert each result to simulation_results ON CONFLICT (_sr_player_date_uc)
          4. WARN if 0 players simulated (off-day or rolling_windows pipeline missing)
          5. Return {"as_of_date": str(yesterday), "players_simulated": n}

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch 14d rolling window rows
                try:
                    rolling_rows = (
                        db.query(PlayerRollingStats)
                        .filter(
                            PlayerRollingStats.as_of_date == as_of_date,
                            PlayerRollingStats.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "ros_simulation: DB query failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("ros_simulation", "failed")
                    return {"status": "failed", "players_simulated": 0, "elapsed_ms": elapsed}

                # Step 2: run simulations (CPU-bound -- offload to thread pool)
                sim_results = await asyncio.to_thread(
                    simulate_all_players,
                    rolling_rows,
                    REMAINING_GAMES_DEFAULT,
                    1000,
                    self._league_means,
                    self._league_stds,
                )

                if not sim_results:
                    logger.warning(
                        "ros_simulation: 0 players simulated for as_of_date=%s -- "
                        "off-day or rolling_windows pipeline missing",
                        as_of_date,
                    )

                # Step 3: upsert results
                try:
                    for res in sim_results:
                        stmt = pg_insert(SimulationResultORM.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            window_days=res.window_days,
                            remaining_games=res.remaining_games,
                            n_simulations=res.n_simulations,
                            player_type=res.player_type,
                            proj_hr_p10=res.proj_hr_p10,
                            proj_hr_p25=res.proj_hr_p25,
                            proj_hr_p50=res.proj_hr_p50,
                            proj_hr_p75=res.proj_hr_p75,
                            proj_hr_p90=res.proj_hr_p90,
                            proj_rbi_p10=res.proj_rbi_p10,
                            proj_rbi_p25=res.proj_rbi_p25,
                            proj_rbi_p50=res.proj_rbi_p50,
                            proj_rbi_p75=res.proj_rbi_p75,
                            proj_rbi_p90=res.proj_rbi_p90,
                            proj_sb_p10=res.proj_sb_p10,
                            proj_sb_p25=res.proj_sb_p25,
                            proj_sb_p50=res.proj_sb_p50,
                            proj_sb_p75=res.proj_sb_p75,
                            proj_sb_p90=res.proj_sb_p90,
                            proj_avg_p10=res.proj_avg_p10,
                            proj_avg_p25=res.proj_avg_p25,
                            proj_avg_p50=res.proj_avg_p50,
                            proj_avg_p75=res.proj_avg_p75,
                            proj_avg_p90=res.proj_avg_p90,
                            proj_k_p10=res.proj_k_p10,
                            proj_k_p25=res.proj_k_p25,
                            proj_k_p50=res.proj_k_p50,
                            proj_k_p75=res.proj_k_p75,
                            proj_k_p90=res.proj_k_p90,
                            proj_era_p10=res.proj_era_p10,
                            proj_era_p25=res.proj_era_p25,
                            proj_era_p50=res.proj_era_p50,
                            proj_era_p75=res.proj_era_p75,
                            proj_era_p90=res.proj_era_p90,
                            proj_whip_p10=res.proj_whip_p10,
                            proj_whip_p25=res.proj_whip_p25,
                            proj_whip_p50=res.proj_whip_p50,
                            proj_whip_p75=res.proj_whip_p75,
                            proj_whip_p90=res.proj_whip_p90,
                            composite_variance=res.composite_variance,
                            downside_p25=res.downside_p25,
                            upside_p75=res.upside_p75,
                            prob_above_median=res.prob_above_median,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_sr_player_date_uc",
                            set_=dict(
                                window_days=res.window_days,
                                remaining_games=res.remaining_games,
                                n_simulations=res.n_simulations,
                                player_type=res.player_type,
                                proj_hr_p10=res.proj_hr_p10,
                                proj_hr_p25=res.proj_hr_p25,
                                proj_hr_p50=res.proj_hr_p50,
                                proj_hr_p75=res.proj_hr_p75,
                                proj_hr_p90=res.proj_hr_p90,
                                proj_rbi_p10=res.proj_rbi_p10,
                                proj_rbi_p25=res.proj_rbi_p25,
                                proj_rbi_p50=res.proj_rbi_p50,
                                proj_rbi_p75=res.proj_rbi_p75,
                                proj_rbi_p90=res.proj_rbi_p90,
                                proj_sb_p10=res.proj_sb_p10,
                                proj_sb_p25=res.proj_sb_p25,
                                proj_sb_p50=res.proj_sb_p50,
                                proj_sb_p75=res.proj_sb_p75,
                                proj_sb_p90=res.proj_sb_p90,
                                proj_avg_p10=res.proj_avg_p10,
                                proj_avg_p25=res.proj_avg_p25,
                                proj_avg_p50=res.proj_avg_p50,
                                proj_avg_p75=res.proj_avg_p75,
                                proj_avg_p90=res.proj_avg_p90,
                                proj_k_p10=res.proj_k_p10,
                                proj_k_p25=res.proj_k_p25,
                                proj_k_p50=res.proj_k_p50,
                                proj_k_p75=res.proj_k_p75,
                                proj_k_p90=res.proj_k_p90,
                                proj_era_p10=res.proj_era_p10,
                                proj_era_p25=res.proj_era_p25,
                                proj_era_p50=res.proj_era_p50,
                                proj_era_p75=res.proj_era_p75,
                                proj_era_p90=res.proj_era_p90,
                                proj_whip_p10=res.proj_whip_p10,
                                proj_whip_p25=res.proj_whip_p25,
                                proj_whip_p50=res.proj_whip_p50,
                                proj_whip_p75=res.proj_whip_p75,
                                proj_whip_p90=res.proj_whip_p90,
                                composite_variance=res.composite_variance,
                                downside_p25=res.downside_p25,
                                upside_p75=res.upside_p75,
                                prob_above_median=res.prob_above_median,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "ros_simulation: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("ros_simulation", "failed")
                    return {"status": "failed", "players_simulated": 0, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n = len(sim_results)
            self._record_job_run("ros_simulation", "success", n)
            logger.info(
                "ros_simulation: %d players simulated for as_of_date=%s "
                "remaining_games=%d elapsed_ms=%d",
                n, as_of_date, REMAINING_GAMES_DEFAULT, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "players_simulated": n,
                "remaining_games": REMAINING_GAMES_DEFAULT,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["ros_simulation"], "ros_simulation", _run)

    async def _run_decision_optimization(self) -> dict:
        """
        Daily Decision Engine optimization (lock 100_022, 7 AM ET).

        Runs after _run_ros_simulation (6 AM) so simulation_results is current.

        Algorithm:
          1. Query player_scores WHERE as_of_date = yesterday AND window_days = 14
          2. Query player_momentum WHERE as_of_date = yesterday (join on bdl_player_id)
          3. Query simulation_results WHERE as_of_date = yesterday (join on bdl_player_id)
          4. Build PlayerDecisionInput list from joined data
          5. Call optimize_lineup(players) and optimize_waivers(players, waiver_pool=[])
          6. Upsert DecisionResult ORM rows ON CONFLICT _dr_date_type_player_uc DO UPDATE
          7. Return summary dict

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch player_scores (14d window)
                try:
                    score_rows = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: player_scores query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "failed")
                    return {"status": "failed", "lineup_decisions": 0, "waiver_decisions": 0, "elapsed_ms": elapsed}

                # Step 2: fetch player_momentum
                try:
                    momentum_rows = (
                        db.query(PlayerMomentum)
                        .filter(PlayerMomentum.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: player_momentum query failed for %s: %s",
                        as_of_date, exc,
                    )
                    momentum_rows = []

                # Step 3: fetch simulation_results
                try:
                    sim_rows = (
                        db.query(SimulationResultORM)
                        .filter(SimulationResultORM.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    sim_rows = []

                if not score_rows:
                    logger.warning(
                        "decision_optimization: 0 player_scores rows for %s -- "
                        "player_scores pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "lineup_decisions": 0,
                        "waiver_decisions": 0,
                        "elapsed_ms": elapsed,
                    }

                # Step 4: build lookup dicts for join
                momentum_by_id = {r.bdl_player_id: r for r in momentum_rows}
                sim_by_id      = {r.bdl_player_id: r for r in sim_rows}

                # H1 fix: fetch real position eligibility from Yahoo roster
                # via the PlayerIDMapping cross-reference table.
                yahoo_positions_by_bdl: dict[int, list[str]] = {}
                try:
                    from backend.fantasy_baseball.yahoo_client_resilient import (
                        YahooFantasyClient, YahooAuthError, YahooAPIError,
                    )
                    client = YahooFantasyClient()
                    roster = client.get_roster()

                    # Build yahoo_key -> positions map from roster
                    yahoo_key_to_pos = {
                        p["player_key"]: p.get("positions", [])
                        for p in roster if p.get("player_key")
                    }

                    # Resolve yahoo_key -> bdl_id via PlayerIDMapping
                    if yahoo_key_to_pos:
                        mappings = (
                            db.query(PlayerIDMapping.yahoo_key, PlayerIDMapping.bdl_id)
                            .filter(
                                PlayerIDMapping.yahoo_key.in_(list(yahoo_key_to_pos.keys())),
                                PlayerIDMapping.bdl_id.isnot(None),
                            )
                            .all()
                        )
                        for ykey, bdl_id in mappings:
                            positions = yahoo_key_to_pos.get(ykey, [])
                            if positions:
                                yahoo_positions_by_bdl[bdl_id] = positions

                    logger.info(
                        "decision_optimization: resolved %d/%d roster players to BDL IDs",
                        len(yahoo_positions_by_bdl), len(roster),
                    )
                except Exception as exc:
                    logger.warning(
                        "decision_optimization: Yahoo roster fetch failed (%s) — "
                        "falling back to player_type heuristic for positions",
                        exc,
                    )

                players = []
                for score in score_rows:
                    pid = score.bdl_player_id
                    mom  = momentum_by_id.get(pid)
                    sim  = sim_by_id.get(pid)

                    pt = (sim.player_type if sim else None) or score.player_type or "unknown"

                    # Use real Yahoo positions when available; fall back to type heuristic
                    eligible = yahoo_positions_by_bdl.get(pid)
                    if not eligible:
                        if pt == "hitter":
                            eligible = ["Util"]
                        elif pt == "pitcher":
                            eligible = ["P"]
                        elif pt == "two_way":
                            eligible = ["Util", "P"]
                        else:
                            eligible = []

                    players.append(PlayerDecisionInput(
                        bdl_player_id=pid,
                        name=getattr(score, "player_name", str(pid)),
                        player_type=pt,
                        eligible_positions=eligible,
                        score_0_100=score.score_0_100 or 0.0,
                        composite_z=score.composite_z or 0.0,
                        momentum_signal=mom.signal if mom else "STABLE",
                        delta_z=mom.delta_z if mom else 0.0,
                        proj_hr_p50=sim.proj_hr_p50    if sim else None,
                        proj_rbi_p50=sim.proj_rbi_p50  if sim else None,
                        proj_sb_p50=sim.proj_sb_p50    if sim else None,
                        proj_avg_p50=sim.proj_avg_p50  if sim else None,
                        proj_k_p50=sim.proj_k_p50      if sim else None,
                        proj_era_p50=sim.proj_era_p50  if sim else None,
                        proj_whip_p50=sim.proj_whip_p50 if sim else None,
                        downside_p25=sim.downside_p25  if sim else None,
                        upside_p75=sim.upside_p75      if sim else None,
                    ))

                # Step 5: run decision engine (CPU-bound -- offload to thread pool)
                lineup_decision, lineup_results = await asyncio.to_thread(
                    optimize_lineup, players, as_of_date
                )

                # M1 fix: fetch waiver pool from Yahoo free agents
                waiver_pool: list = []
                try:
                    if "client" not in dir():
                        from backend.fantasy_baseball.yahoo_client_resilient import (
                            YahooFantasyClient,
                        )
                        client = YahooFantasyClient()
                    free_agents = client.get_free_agents(count=25)

                    # Resolve yahoo_key -> bdl_id for free agents
                    fa_keys = [p["player_key"] for p in free_agents if p.get("player_key")]
                    fa_bdl_map: dict[str, int] = {}
                    if fa_keys:
                        fa_mappings = (
                            db.query(PlayerIDMapping.yahoo_key, PlayerIDMapping.bdl_id)
                            .filter(
                                PlayerIDMapping.yahoo_key.in_(fa_keys),
                                PlayerIDMapping.bdl_id.isnot(None),
                            )
                            .all()
                        )
                        fa_bdl_map = {ykey: bdl_id for ykey, bdl_id in fa_mappings}

                    # FA identity resolution fallback: yahoo_key is NULL for every
                    # free agent in player_id_mapping (only roster players are
                    # backfilled by scripts/backfill_yahoo_keys.py), so yahoo_key
                    # lookup always misses for FAs. Best-effort fuzzy-match FA
                    # names against player_id_mapping.normalized_name. Read-only:
                    # never writes to player_id_mapping from here.
                    def _norm_fa_name(name: str) -> str:
                        if not name:
                            return ""
                        n = unicodedata.normalize("NFKD", name).lower().strip()
                        for suffix in (" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"):
                            if n.endswith(suffix):
                                n = n[: -len(suffix)].strip()
                        n = n.replace(".", "")
                        while "  " in n:
                            n = n.replace("  ", " ")
                        return n.strip()

                    fa_name_map: dict[str, list[int]] = {}
                    unresolved_fas = [
                        fa for fa in free_agents
                        if fa_bdl_map.get(fa.get("player_key")) is None
                    ]
                    if unresolved_fas:
                        mapping_rows = (
                            db.query(PlayerIDMapping.normalized_name, PlayerIDMapping.bdl_id)
                            .filter(PlayerIDMapping.bdl_id.isnot(None))
                            .all()
                        )
                        for norm_name, bdl_id in mapping_rows:
                            key = _norm_fa_name(norm_name or "")
                            if not key:
                                continue
                            fa_name_map.setdefault(key, []).append(bdl_id)

                    fa_skipped = 0
                    fa_fuzzy_resolved = 0
                    for fa in free_agents:
                        fa_bdl_id = fa_bdl_map.get(fa.get("player_key"))
                        if fa_bdl_id is None:
                            fa_name = fa.get("name", "")
                            candidates = fa_name_map.get(_norm_fa_name(fa_name), [])
                            if len(candidates) == 1:
                                fa_bdl_id = candidates[0]
                                fa_fuzzy_resolved += 1
                                logger.info(
                                    "decision_optimization: resolved FA '%s' via "
                                    "name fallback (bdl_id=%d)",
                                    fa_name, fa_bdl_id,
                                )
                            else:
                                fa_skipped += 1
                                logger.debug(
                                    "decision_optimization: FA '%s' unresolved "
                                    "(yahoo_key miss, %d name candidates)",
                                    fa_name, len(candidates),
                                )
                                continue
                        fa_positions = fa.get("positions", [])
                        if not fa_positions:
                            fa_positions = ["Util"]
                        # Determine player type from positions
                        fa_type = "hitter"
                        pitcher_pos = {"SP", "RP", "P"}
                        if all(p in pitcher_pos for p in fa_positions):
                            fa_type = "pitcher"
                        elif any(p in pitcher_pos for p in fa_positions) and any(
                            p not in pitcher_pos for p in fa_positions
                        ):
                            fa_type = "two_way"
                        waiver_pool.append(PlayerDecisionInput(
                            bdl_player_id=fa_bdl_id,
                            name=fa.get("name", str(fa_bdl_id)),
                            player_type=fa_type,
                            eligible_positions=fa_positions,
                            score_0_100=0.0,
                            composite_z=0.0,
                            momentum_signal="STABLE",
                            delta_z=0.0,
                        ))
                    logger.info(
                        "decision_optimization: built waiver pool with %d candidates "
                        "(fuzzy_resolved=%d, skipped=%d, fetched=%d)",
                        len(waiver_pool), fa_fuzzy_resolved, fa_skipped, len(free_agents),
                    )
                    if fa_skipped > 0 and len(free_agents) > 0:
                        logger.warning(
                            "decision_optimization: %d/%d free agents unresolvable to "
                            "bdl_id (yahoo_key NULL in player_id_mapping and no unique "
                            "normalized_name match). Fuzzy resolved %d. See "
                            "reports/2026-04-15-decision-results-investigation.md.",
                            fa_skipped, len(free_agents), fa_fuzzy_resolved,
                        )
                except Exception as exc:
                    logger.warning(
                        "decision_optimization: waiver pool fetch failed (%s) — "
                        "proceeding with empty pool",
                        exc,
                    )

                _waiver_decision, waiver_results = await asyncio.to_thread(
                    optimize_waivers, players, waiver_pool, as_of_date
                )

                all_results = lineup_results + waiver_results

                # Step 6: upsert DecisionResult rows
                try:
                    for res in all_results:
                        stmt = pg_insert(DecisionResultORM.__table__).values(
                            as_of_date=res.as_of_date,
                            decision_type=res.decision_type,
                            bdl_player_id=res.bdl_player_id,
                            target_slot=res.target_slot,
                            drop_player_id=res.drop_player_id,
                            lineup_score=res.lineup_score,
                            value_gain=res.value_gain,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_dr_date_type_player_uc",
                            set_=dict(
                                target_slot=res.target_slot,
                                drop_player_id=res.drop_player_id,
                                lineup_score=res.lineup_score,
                                value_gain=res.value_gain,
                                confidence=res.confidence,
                                reasoning=res.reasoning,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "decision_optimization: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "failed")
                    return {
                        "status": "failed",
                        "lineup_decisions": 0,
                        "waiver_decisions": 0,
                        "elapsed_ms": elapsed,
                    }

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n_lineup = len(lineup_results)
            n_waiver = len(waiver_results)
            self._record_job_run("decision_optimization", "success", n_lineup + n_waiver)
            logger.info(
                "decision_optimization: %d lineup + %d waiver decisions for as_of_date=%s "
                "elapsed_ms=%d",
                n_lineup, n_waiver, as_of_date, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "lineup_decisions": n_lineup,
                "waiver_decisions": n_waiver,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["decision_optimization"], "decision_optimization", _run)

    async def _run_backtesting(self) -> dict:
        """
        Daily Backtesting Harness (lock 100_023, 8 AM ET).

        Runs after _run_decision_optimization (7 AM) so the full pipeline is current.

        Algorithm:
          1. Compute as_of_date = yesterday
          2. Query simulation_results WHERE as_of_date = yesterday AND window_days = 14
          3. For each sim_row, query mlb_player_stats for the 14-day actuals window
          4. Aggregate actuals: sum HR/RBI/SB/K, mean AVG, IP-weighted ERA/WHIP
          5. Build BacktestInput list and call evaluate_cohort via asyncio.to_thread
          6. Call summarize() with golden baseline loaded from BASELINE_PATH
          7. Save new golden baseline if no regression detected
          8. Upsert BacktestResultORM rows ON CONFLICT _br_player_date_uc DO UPDATE
          9. Return summary dict

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            window_start = as_of_date - timedelta(days=14)
            now = datetime.now(ZoneInfo("America/New_York"))

            results = []   # populated after evaluate_cohort; guard for finally path
            summary = None  # populated after summarize(); guard for finally path

            db = SessionLocal()
            try:
                # Step 1: fetch simulation_results for yesterday (14d window)
                try:
                    sim_rows = (
                        db.query(SimulationResultORM)
                        .filter(
                            SimulationResultORM.as_of_date == as_of_date,
                            SimulationResultORM.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "backtesting: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "failed")
                    return {"status": "failed", "n_players": 0, "elapsed_ms": elapsed}

                if not sim_rows:
                    logger.warning(
                        "backtesting: 0 simulation_results rows for as_of_date=%s -- "
                        "ros_simulation pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "n_players": 0,
                        "mean_composite_mae": None,
                        "regression_detected": False,
                        "elapsed_ms": elapsed,
                    }

                # Step 2: for each player, fetch actual stats from the 14-day window
                inputs = []
                for sim in sim_rows:
                    pid = sim.bdl_player_id
                    try:
                        stat_rows = (
                            db.query(MLBPlayerStats)
                            .filter(
                                MLBPlayerStats.bdl_player_id == pid,
                                MLBPlayerStats.game_date >= window_start,
                                MLBPlayerStats.game_date <= as_of_date,
                            )
                            .all()
                        )
                    except Exception as exc:
                        logger.warning(
                            "backtesting: stats query failed for player %d: %s",
                            pid, exc,
                        )
                        stat_rows = []

                    games_played = len(stat_rows)

                    # Aggregate batting totals
                    actual_hr  = None
                    actual_rbi = None
                    actual_sb  = None
                    actual_avg = None
                    actual_k   = None
                    actual_era = None
                    actual_whip = None

                    if stat_rows:
                        hr_vals  = [r.home_runs    for r in stat_rows if r.home_runs    is not None]
                        rbi_vals = [r.rbi           for r in stat_rows if r.rbi          is not None]
                        sb_vals  = [r.stolen_bases  for r in stat_rows if r.stolen_bases is not None]
                        k_vals   = [r.strikeouts_pit for r in stat_rows if r.strikeouts_pit is not None]

                        actual_hr  = float(sum(hr_vals))  if hr_vals  else None
                        actual_rbi = float(sum(rbi_vals)) if rbi_vals else None
                        actual_sb  = float(sum(sb_vals))  if sb_vals  else None
                        actual_k   = float(sum(k_vals))   if k_vals   else None

                        # M4 fix: compute AVG as total_hits/total_ab (AB-weighted),
                        # not arithmetic mean of per-game AVG.
                        hit_vals = [r.hits for r in stat_rows if r.hits is not None]
                        ab_vals  = [r.ab   for r in stat_rows if r.ab   is not None]
                        total_hits = sum(hit_vals)
                        total_ab   = sum(ab_vals)
                        actual_avg = total_hits / total_ab if total_ab > 0 else None

                        # IP-weighted ERA and WHIP aggregation
                        # innings_pitched stored as string e.g. "6.2" meaning 6 and 2/3
                        total_ip = 0.0
                        era_sum  = 0.0
                        whip_sum = 0.0
                        for r in stat_rows:
                            ip_str = r.innings_pitched
                            if ip_str is None:
                                continue
                            try:
                                parts = str(ip_str).split(".")
                                whole = int(parts[0])
                                frac  = int(parts[1]) if len(parts) > 1 else 0
                                ip_dec = whole + frac / 3.0
                            except (ValueError, IndexError):
                                ip_dec = 0.0
                            if ip_dec <= 0.0:
                                continue
                            total_ip += ip_dec
                            if r.era is not None:
                                era_sum += r.era * ip_dec
                            if r.whip is not None:
                                whip_sum += r.whip * ip_dec

                        if total_ip > 0.0:
                            actual_era  = era_sum  / total_ip
                            actual_whip = whip_sum / total_ip

                    # H3 fix: scale ROS projections (130-game totals) down to
                    # match the 14-day actual window. Without this, MAE compares
                    # season totals to ~10-game sums and produces garbage values.
                    remaining = sim.remaining_games or REMAINING_GAMES_DEFAULT
                    scale = games_played / remaining if remaining > 0 and games_played > 0 else 0.0

                    def _scale(val):
                        return val * scale if val is not None else None

                    inputs.append(BacktestInput(
                        bdl_player_id=pid,
                        as_of_date=as_of_date,
                        player_type=sim.player_type,
                        proj_hr_p50=_scale(sim.proj_hr_p50),
                        proj_rbi_p50=_scale(sim.proj_rbi_p50),
                        proj_sb_p50=_scale(sim.proj_sb_p50),
                        proj_avg_p50=sim.proj_avg_p50,       # AVG is a rate — don't scale
                        proj_k_p50=_scale(sim.proj_k_p50),
                        proj_era_p50=sim.proj_era_p50,       # ERA is a rate — don't scale
                        proj_whip_p50=sim.proj_whip_p50,     # WHIP is a rate — don't scale
                        actual_hr=actual_hr,
                        actual_rbi=actual_rbi,
                        actual_sb=actual_sb,
                        actual_avg=actual_avg,
                        actual_k=actual_k,
                        actual_era=actual_era,
                        actual_whip=actual_whip,
                        games_played=games_played,
                    ))

                # Step 3: evaluate cohort (CPU-bound -- offload to thread pool)
                results = await asyncio.to_thread(evaluate_cohort, inputs)

                # Step 4: summarize with golden baseline
                baseline_data = load_golden_baseline(BASELINE_PATH)
                baseline_mae = baseline_data.get("mean_composite_mae")
                summary = summarize(results, window_start, as_of_date, baseline_mae)

                # Step 5: persist new baseline if no regression
                if not summary.regression_detected:
                    try:
                        save_golden_baseline(summary, BASELINE_PATH)
                    except Exception as exc:
                        logger.warning(
                            "backtesting: could not save golden baseline: %s", exc
                        )

                if summary.regression_detected:
                    logger.warning(
                        "backtesting: REGRESSION DETECTED as_of_date=%s "
                        "mean_composite_mae=%.4f baseline=%.4f delta=%.4f",
                        as_of_date,
                        summary.mean_composite_mae or 0.0,
                        baseline_mae or 0.0,
                        summary.regression_delta or 0.0,
                    )

                # Step 6: upsert BacktestResultORM rows
                try:
                    for res in results:
                        stmt = pg_insert(BacktestResultORM.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            player_type=res.player_type,
                            games_played=res.games_played,
                            mae_hr=res.mae_hr,
                            rmse_hr=res.rmse_hr,
                            mae_rbi=res.mae_rbi,
                            rmse_rbi=res.rmse_rbi,
                            mae_sb=res.mae_sb,
                            rmse_sb=res.rmse_sb,
                            mae_avg=res.mae_avg,
                            rmse_avg=res.rmse_avg,
                            mae_k=res.mae_k,
                            rmse_k=res.rmse_k,
                            mae_era=res.mae_era,
                            rmse_era=res.rmse_era,
                            mae_whip=res.mae_whip,
                            rmse_whip=res.rmse_whip,
                            composite_mae=res.composite_mae,
                            direction_correct=res.direction_correct,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_br_player_date_uc",
                            set_=dict(
                                player_type=res.player_type,
                                games_played=res.games_played,
                                mae_hr=res.mae_hr,
                                rmse_hr=res.rmse_hr,
                                mae_rbi=res.mae_rbi,
                                rmse_rbi=res.rmse_rbi,
                                mae_sb=res.mae_sb,
                                rmse_sb=res.rmse_sb,
                                mae_avg=res.mae_avg,
                                rmse_avg=res.rmse_avg,
                                mae_k=res.mae_k,
                                rmse_k=res.rmse_k,
                                mae_era=res.mae_era,
                                rmse_era=res.rmse_era,
                                mae_whip=res.mae_whip,
                                rmse_whip=res.rmse_whip,
                                composite_mae=res.composite_mae,
                                direction_correct=res.direction_correct,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "backtesting: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "failed")
                    return {"status": "failed", "n_players": len(results), "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n = len(results)
            self._record_job_run("backtesting", "success", n)
            logger.info(
                "backtesting: %d players evaluated for as_of_date=%s "
                "mean_composite_mae=%s regression=%s elapsed_ms=%d",
                n, as_of_date, summary.mean_composite_mae,
                summary.regression_detected, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "n_players": n,
                "mean_composite_mae": summary.mean_composite_mae,
                "regression_detected": summary.regression_detected,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["backtesting"], "backtesting", _run)

    async def _run_explainability(self) -> dict:
        """
        Daily Explainability Engine (lock 100_024, 9 AM ET).

        Runs after _run_backtesting (8 AM) so all P14-P18 signals are current.

        Algorithm:
          1. Query decision_results WHERE as_of_date = yesterday
          2. For each decision, join player_scores (14d), player_momentum,
             simulation_results, backtest_results, and PlayerIDMapping for names
          3. Build ExplanationInput dataclasses; skip if player_scores row missing
          4. Call explain_batch(inputs) via asyncio.to_thread (CPU-bound)
          5. Upsert DecisionExplanationORM rows ON CONFLICT _de_decision_id_uc DO UPDATE
          6. Return summary dict with n_explained, n_skipped, elapsed_ms

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch all decision_results for yesterday
                try:
                    decision_rows = (
                        db.query(DecisionResultORM)
                        .filter(DecisionResultORM.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "explainability: decision_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "failed")
                    return {"status": "failed", "n_explained": 0, "n_skipped": 0, "elapsed_ms": elapsed}

                if not decision_rows:
                    logger.warning(
                        "explainability: 0 decision_results rows for as_of_date=%s -- "
                        "decision_optimization pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "n_explained": 0,
                        "n_skipped": 0,
                        "elapsed_ms": elapsed,
                    }

                # Step 2: bulk-fetch supporting tables into dicts keyed by bdl_player_id
                try:
                    score_map = {
                        row.bdl_player_id: row
                        for row in db.query(PlayerScore).filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        ).all()
                    }
                except Exception as exc:
                    logger.error(
                        "explainability: player_scores query failed for %s: %s",
                        as_of_date, exc,
                    )
                    score_map = {}

                try:
                    momentum_map = {
                        row.bdl_player_id: row
                        for row in db.query(PlayerMomentum).filter(
                            PlayerMomentum.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: player_momentum query failed for %s: %s",
                        as_of_date, exc,
                    )
                    momentum_map = {}

                try:
                    sim_map = {
                        row.bdl_player_id: row
                        for row in db.query(SimulationResultORM).filter(
                            SimulationResultORM.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    sim_map = {}

                try:
                    backtest_map = {
                        row.bdl_player_id: row
                        for row in db.query(BacktestResultORM).filter(
                            BacktestResultORM.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: backtest_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    backtest_map = {}

                # Build a name-lookup dict from PlayerIDMapping (bdl_id -> full_name)
                try:
                    all_pids = set(d.bdl_player_id for d in decision_rows)
                    if decision_rows:
                        # also include drop_player_ids
                        for d in decision_rows:
                            if d.drop_player_id is not None:
                                all_pids.add(d.drop_player_id)
                    name_map = {
                        row.bdl_id: row.full_name
                        for row in db.query(PlayerIDMapping).filter(
                            PlayerIDMapping.bdl_id.in_(list(all_pids)),
                        ).all()
                        if row.bdl_id is not None
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: PlayerIDMapping query failed: %s", exc,
                    )
                    name_map = {}

                # Step 3: build ExplanationInput list
                inputs = []
                n_skipped = 0
                for dec in decision_rows:
                    pid = dec.bdl_player_id
                    score_row = score_map.get(pid)
                    if score_row is None:
                        # Cannot explain without Z-scores
                        n_skipped += 1
                        continue

                    momentum_row = momentum_map.get(pid)
                    sim_row = sim_map.get(pid)
                    bt_row = backtest_map.get(pid)

                    player_name = name_map.get(pid, "Player {}".format(pid))
                    drop_name = None
                    if dec.drop_player_id is not None:
                        drop_name = name_map.get(dec.drop_player_id, "Player {}".format(dec.drop_player_id))

                    inputs.append(ExplanationInput(
                        decision_id=dec.id,
                        as_of_date=as_of_date,
                        decision_type=dec.decision_type,
                        bdl_player_id=pid,
                        player_name=player_name,
                        target_slot=dec.target_slot,
                        drop_player_id=dec.drop_player_id,
                        drop_player_name=drop_name,
                        lineup_score=dec.lineup_score,
                        value_gain=dec.value_gain,
                        decision_confidence=dec.confidence if dec.confidence is not None else 0.0,
                        player_type=score_row.player_type,
                        score_0_100=score_row.score_0_100 if score_row.score_0_100 is not None else 0.0,
                        composite_z=score_row.composite_z if score_row.composite_z is not None else 0.0,
                        z_hr=score_row.z_hr,
                        z_rbi=score_row.z_rbi,
                        z_sb=score_row.z_sb,
                        z_nsb=score_row.z_nsb,
                        z_avg=score_row.z_avg,
                        z_obp=score_row.z_obp,
                        z_era=score_row.z_era,
                        z_whip=score_row.z_whip,
                        z_k_per_9=score_row.z_k_per_9,
                        score_confidence=score_row.confidence if score_row.confidence is not None else 0.0,
                        games_in_window=score_row.games_in_window if score_row.games_in_window is not None else 0,
                        signal=momentum_row.signal if momentum_row else "STABLE",
                        delta_z=momentum_row.delta_z if momentum_row and momentum_row.delta_z is not None else 0.0,
                        proj_hr_p50=sim_row.proj_hr_p50 if sim_row else None,
                        proj_rbi_p50=sim_row.proj_rbi_p50 if sim_row else None,
                        proj_sb_p50=sim_row.proj_sb_p50 if sim_row else None,
                        proj_avg_p50=sim_row.proj_avg_p50 if sim_row else None,
                        proj_k_p50=sim_row.proj_k_p50 if sim_row else None,
                        proj_era_p50=sim_row.proj_era_p50 if sim_row else None,
                        proj_whip_p50=sim_row.proj_whip_p50 if sim_row else None,
                        prob_above_median=sim_row.prob_above_median if sim_row else None,
                        downside_p25=sim_row.downside_p25 if sim_row else None,
                        upside_p75=sim_row.upside_p75 if sim_row else None,
                        backtest_composite_mae=bt_row.composite_mae if bt_row else None,
                        backtest_games=bt_row.games_played if bt_row else None,
                    ))

                # Step 4: generate explanations (CPU-bound -- offload to thread pool)
                explanation_results = await asyncio.to_thread(explain_batch, inputs)

                if not explanation_results:
                    logger.warning(
                        "explainability: 0 explanations generated for as_of_date=%s "
                        "(inputs=%d, skipped=%d)",
                        as_of_date, len(inputs), n_skipped,
                    )

                # Step 5: upsert DecisionExplanationORM rows
                try:
                    for res in explanation_results:
                        factors_data = [
                            {
                                "name": f.name,
                                "value": f.value,
                                "label": f.label,
                                "weight": f.weight,
                                "narrative": f.narrative,
                            }
                            for f in res.factors
                        ]
                        stmt = pg_insert(DecisionExplanationORM.__table__).values(
                            decision_id=res.decision_id,
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            decision_type=res.decision_type,
                            summary=res.summary,
                            factors_json=factors_data,
                            confidence_narrative=res.confidence_narrative,
                            risk_narrative=res.risk_narrative,
                            track_record_narrative=res.track_record_narrative,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_de_decision_id_uc",
                            set_=dict(
                                bdl_player_id=res.bdl_player_id,
                                as_of_date=res.as_of_date,
                                decision_type=res.decision_type,
                                summary=res.summary,
                                factors_json=factors_data,
                                confidence_narrative=res.confidence_narrative,
                                risk_narrative=res.risk_narrative,
                                track_record_narrative=res.track_record_narrative,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "explainability: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "failed")
                    return {"status": "failed", "n_explained": 0, "n_skipped": n_skipped, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n_explained = len(explanation_results)
            self._record_job_run("explainability", "success", n_explained)
            logger.info(
                "explainability: %d decisions explained for as_of_date=%s "
                "skipped=%d elapsed_ms=%d",
                n_explained, as_of_date, n_skipped, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "n_explained": n_explained,
                "n_skipped": n_skipped,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["explainability"], "explainability", _run)

    async def _run_snapshot(self) -> dict:
        """
        Daily Snapshot Engine (lock 100_025, 10 AM ET).

        Runs after _run_explainability (9 AM) -- final stage of the daily pipeline.

        Algorithm:
          1. Compute as_of_date = yesterday
          2. Query count metrics from all 6 phase tables for that date
          3. Compute regression detection vs. historical average composite_mae
          4. Fetch top 5 lineup + top 3 waiver player IDs from decision_results
          5. Build SnapshotInput; call build_snapshot() via asyncio.to_thread
          6. Upsert DailySnapshotORM ON CONFLICT _ds_date_uc DO UPDATE all columns
          7. Return summary dict with health, counts, elapsed_ms

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 2: query count metrics from all phase tables
                n_players_scored = (
                    db.query(func.count(PlayerScore.id))
                    .filter(
                        PlayerScore.as_of_date == as_of_date,
                        PlayerScore.window_days == 14,
                    )
                    .scalar() or 0
                )

                n_momentum_records = (
                    db.query(func.count(PlayerMomentum.id))
                    .filter(PlayerMomentum.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_simulation_records = (
                    db.query(func.count(SimulationResultORM.id))
                    .filter(SimulationResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_decisions = (
                    db.query(func.count(DecisionResultORM.id))
                    .filter(DecisionResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_explanations = (
                    db.query(func.count(DecisionExplanationORM.id))
                    .filter(DecisionExplanationORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_backtest_records = (
                    db.query(func.count(BacktestResultORM.id))
                    .filter(BacktestResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                mean_mae = (
                    db.query(func.avg(BacktestResultORM.composite_mae))
                    .filter(BacktestResultORM.as_of_date == as_of_date)
                    .scalar()
                )  # may be None

                # Step 3: regression detection vs. historical baseline
                prev_avg = (
                    db.query(func.avg(BacktestResultORM.composite_mae))
                    .filter(
                        BacktestResultORM.as_of_date < as_of_date,
                        BacktestResultORM.composite_mae.isnot(None),
                    )
                    .scalar()
                )
                regression_detected = (
                    mean_mae is not None
                    and prev_avg is not None
                    and mean_mae > prev_avg * 1.20
                )

                # Step 4: top lineup and waiver player IDs
                top_lineup = [
                    r.bdl_player_id
                    for r in db.query(DecisionResultORM.bdl_player_id)
                    .filter(
                        DecisionResultORM.as_of_date == as_of_date,
                        DecisionResultORM.decision_type == "lineup",
                    )
                    .order_by(DecisionResultORM.lineup_score.desc())
                    .limit(5)
                    .all()
                ]

                top_waiver = [
                    r.bdl_player_id
                    for r in db.query(DecisionResultORM.bdl_player_id)
                    .filter(
                        DecisionResultORM.as_of_date == as_of_date,
                        DecisionResultORM.decision_type == "waiver",
                    )
                    .order_by(DecisionResultORM.value_gain.desc())
                    .limit(3)
                    .all()
                ]

                # Step 5: build SnapshotInput and compute result
                inp = SnapshotInput(
                    as_of_date=as_of_date,
                    n_players_scored=n_players_scored,
                    n_momentum_records=n_momentum_records,
                    n_simulation_records=n_simulation_records,
                    n_decisions=n_decisions,
                    n_explanations=n_explanations,
                    n_backtest_records=n_backtest_records,
                    mean_composite_mae=mean_mae,
                    regression_detected=regression_detected,
                    top_lineup_player_ids=top_lineup,
                    top_waiver_player_ids=top_waiver,
                    pipeline_jobs_run=[
                        job_id
                        for job_id, info in self._job_status.items()
                        if info.get("last_status") == "success"
                    ],
                )

                result = await asyncio.to_thread(build_snapshot, inp)

                # Step 6: upsert DailySnapshotORM ON CONFLICT _ds_date_uc DO UPDATE
                try:
                    stmt = pg_insert(DailySnapshotORM.__table__).values(
                        as_of_date=result.as_of_date,
                        n_players_scored=result.n_players_scored,
                        n_momentum_records=result.n_momentum_records,
                        n_simulation_records=result.n_simulation_records,
                        n_decisions=result.n_decisions,
                        n_explanations=result.n_explanations,
                        n_backtest_records=result.n_backtest_records,
                        mean_composite_mae=result.mean_composite_mae,
                        regression_detected=result.regression_detected,
                        top_lineup_player_ids=result.top_lineup_player_ids,
                        top_waiver_player_ids=result.top_waiver_player_ids,
                        pipeline_jobs_run=result.pipeline_jobs_run,
                        pipeline_health=result.pipeline_health,
                        health_reasons=result.health_reasons,
                        summary=result.summary,
                        computed_at=now,
                    ).on_conflict_do_update(
                        constraint="_ds_date_uc",
                        set_=dict(
                            n_players_scored=result.n_players_scored,
                            n_momentum_records=result.n_momentum_records,
                            n_simulation_records=result.n_simulation_records,
                            n_decisions=result.n_decisions,
                            n_explanations=result.n_explanations,
                            n_backtest_records=result.n_backtest_records,
                            mean_composite_mae=result.mean_composite_mae,
                            regression_detected=result.regression_detected,
                            top_lineup_player_ids=result.top_lineup_player_ids,
                            top_waiver_player_ids=result.top_waiver_player_ids,
                            pipeline_jobs_run=result.pipeline_jobs_run,
                            pipeline_health=result.pipeline_health,
                            health_reasons=result.health_reasons,
                            summary=result.summary,
                            computed_at=now,
                        ),
                    )
                    db.execute(stmt)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "snapshot: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("snapshot", "failed")
                    return {
                        "status": "failed",
                        "as_of_date": str(as_of_date),
                        "pipeline_health": "FAILED",
                        "n_players_scored": n_players_scored,
                        "n_decisions": n_decisions,
                        "elapsed_ms": elapsed,
                    }

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            self._record_job_run("snapshot", "success", n_players_scored)
            logger.info(
                "snapshot: pipeline_health=%s n_players_scored=%d n_decisions=%d "
                "as_of_date=%s elapsed_ms=%d",
                result.pipeline_health, n_players_scored, n_decisions, as_of_date, elapsed,
            )
            return {
                "as_of_date": str(as_of_date),
                "pipeline_health": result.pipeline_health,
                "n_players_scored": n_players_scored,
                "n_decisions": n_decisions,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["snapshot"], "snapshot", _run)

    async def _update_statcast(self) -> dict:
        """Daily Statcast enrichment — fetches yesterday's data and runs Bayesian projection updates."""      
        t0 = time.monotonic()
        async def _run():
            try:
                result = await asyncio.to_thread(run_daily_ingestion)
                elapsed = int((time.monotonic() - t0) * 1000)
                status = "success" if result.get("success") else "failed"

                if not result.get("success"):
                    logger.error(
                        "_update_statcast: ingestion reported failure -- %s",
                        result.get("error", "unknown error"),
                    )

                records = result.get("records_processed", 0) if isinstance(result, dict) else 0
                self._record_job_run("statcast", status)

                if isinstance(result, dict):
                    result["status"] = status
                    result["records"] = records
                    result["elapsed_ms"] = elapsed
                else:
                    result = {"status": status, "records": 0, "elapsed_ms": elapsed}
                return result
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.exception("_update_statcast: unhandled error -- %s", exc)
                self._record_job_run("statcast", "failed")
                return {"status": "failed", "records": 0, "error": str(exc), "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["statcast"], "statcast", _run)

    async def _calc_rolling_zscores(self) -> dict:
        """
        Compute 7-day and 30-day rolling z-scores for all MLB players who have
        sufficient history in player_daily_metrics.

        Spec: HANDOFF.md section 3.5
        - 7-day window: requires >= 7 rows  -> z_score_recent
        - 30-day window: requires >= 30 rows -> z_score_total
        Writes a ProjectionSnapshot with significant_changes count.
        """
        t0 = time.monotonic()

        async def _run():
            import statistics

            today = today_et()
            cutoff = today - timedelta(days=30)
            db = SessionLocal()
            records_updated = 0
            significant_changes = 0
            skipped_insufficient_data = 0
            try:
                rows = (
                    db.query(PlayerDailyMetric)
                    .filter(
                        PlayerDailyMetric.sport == "mlb",
                        PlayerDailyMetric.metric_date >= cutoff,
                    )
                    .order_by(PlayerDailyMetric.player_id, PlayerDailyMetric.metric_date)
                    .all()
                )

                # Group by player
                players: dict[str, list] = {}
                for row in rows:
                    players.setdefault(row.player_id, []).append(row)

                for player_id, player_rows in players.items():
                    # Sort ascending by date
                    player_rows.sort(key=lambda r: r.metric_date)

                    vorp_values = [
                        r.vorp_7d for r in player_rows if r.vorp_7d is not None
                    ]

                    new_z_recent: Optional[float] = None
                    new_z_total: Optional[float] = None

                    # 7-day window z-score
                    if len(vorp_values) >= 7:
                        window_7 = vorp_values[-7:]
                        mean_7 = statistics.mean(window_7)
                        try:
                            std_7 = statistics.stdev(window_7)
                        except statistics.StatisticsError:
                            std_7 = 0.0
                        if std_7 > 0 and vorp_values:
                            new_z_recent = (vorp_values[-1] - mean_7) / std_7
                        else:
                            new_z_recent = 0.0

                    # 30-day window z-score
                    if len(vorp_values) >= 30:
                        mean_30 = statistics.mean(vorp_values)
                        try:
                            std_30 = statistics.stdev(vorp_values)
                        except statistics.StatisticsError:
                            std_30 = 0.0
                        if std_30 > 0 and vorp_values:
                            new_z_total = (vorp_values[-1] - mean_30) / std_30
                        else:
                            new_z_total = 0.0

                    if new_z_recent is None and new_z_total is None:
                        skipped_insufficient_data += 1
                        continue

                    # Detect significant change vs stored values
                    latest_row = player_rows[-1]
                    old_z = latest_row.z_score_recent or 0.0
                    if new_z_recent is not None and abs(new_z_recent - old_z) > 0.5:
                        significant_changes += 1

                    # Upsert today's row
                    existing = (
                        db.query(PlayerDailyMetric)
                        .filter(
                            PlayerDailyMetric.player_id == player_id,
                            PlayerDailyMetric.metric_date == today,
                            PlayerDailyMetric.sport == "mlb",
                        )
                        .first()
                    )
                    if existing:
                        if new_z_recent is not None:
                            existing.z_score_recent = new_z_recent
                        if new_z_total is not None:
                            existing.z_score_total = new_z_total
                    else:
                        # Create a minimal row for today's z-scores
                        new_metric = PlayerDailyMetric(
                            player_id=player_id,
                            player_name=latest_row.player_name,
                            metric_date=today,
                            sport="mlb",
                            z_score_recent=new_z_recent,
                            z_score_total=new_z_total,
                            rolling_window={},
                            data_source="rolling_zscore_job",
                        )
                        db.add(new_metric)

                    records_updated += 1

                db.commit()

                # Write ProjectionSnapshot
                snapshot = ProjectionSnapshot(
                    snapshot_date=today,
                    sport="mlb",
                    player_changes={},
                    total_players=len(players),
                    significant_changes=significant_changes,
                )
                db.add(snapshot)
                db.commit()

            except Exception as exc:
                db.rollback()
                logger.error("_calc_rolling_zscores error: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_z", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "_calc_rolling_zscores: updated %d players, %d significant changes",
                records_updated, significant_changes,
            )
            # M-5: alert when most players lack sufficient history (first 7 days of season)
            total_seen = len(players)
            if total_seen > 0 and skipped_insufficient_data > 0:
                skip_pct = skipped_insufficient_data / total_seen * 100
                if skipped_insufficient_data == total_seen:
                    logger.warning(
                        "rolling_z: ALL %d players skipped — insufficient data "
                        "(need >= 7 days of vorp_7d). Season may be < 7 days old.",
                        total_seen,
                    )
                elif skip_pct > 50:
                    logger.warning(
                        "rolling_z: %d/%d players skipped (%.0f%%) — insufficient data. "
                        "z-scores will improve as season history accumulates.",
                        skipped_insufficient_data, total_seen, skip_pct,
                    )
                else:
                    logger.debug(
                        "rolling_z: %d players skipped for insufficient data (<7 rows)",
                        skipped_insufficient_data,
                    )
            self._record_job_run("rolling_z", "success", records_updated)
            return {"status": "success", "records": records_updated, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["rolling_z"], "rolling_z", _run)

    async def _poll_yahoo_adp_injury(self) -> dict:
        """Fetch Yahoo ADP + injury status snapshot every 4 hours (lock 100_013).

        Pulls up to 100 players sorted by ADP (sort=DA) and caches their
        injury status in PlayerDailyMetric.  This feed is the sole source for
        detecting new injuries and ADP rank movements without polling each
        player individually.

        Data written: player status, injury_note, percent_owned updated on today's row.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.yahoo_client_resilient import (
                YahooFantasyClient, YahooAuthError, YahooAPIError,
            )
            from backend.services.yahoo_ingestion import get_validated_adp_feed
            try:
                client = YahooFantasyClient()
            except YahooAuthError as exc:
                logger.error("yahoo_adp_injury: auth error — %s", exc)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0}

            try:
                players = get_validated_adp_feed(client)
            except (YahooAuthError, YahooAPIError) as exc:
                logger.error("yahoo_adp_injury: API error — %s", exc)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0}

            if not players:
                logger.warning("yahoo_adp_injury: no players returned")
                self._record_job_run("yahoo_adp_injury", "success", 0)
                return {"status": "success", "records": 0}

            today = today_et()
            db = SessionLocal()
            records_written = 0
            injury_flags = 0
            try:
                for p in players:
                    if p.is_injured:
                        injury_flags += 1

                    existing = (
                        db.query(PlayerDailyMetric)
                        .filter(
                            PlayerDailyMetric.player_id == p.player_key,
                            PlayerDailyMetric.metric_date == today,
                            PlayerDailyMetric.sport == "mlb",
                        )
                        .first()
                    )
                    if existing:
                        # Patch injury / ownership fields on existing row
                        existing.rolling_window = {
                            **(existing.rolling_window or {}),
                            "status": p.status,
                            "injury_note": p.injury_note,
                            "percent_owned": p.percent_owned,
                            "adp_updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        }
                    else:
                        db.add(PlayerDailyMetric(
                            player_id=p.player_key,
                            player_name=p.name,
                            metric_date=today,
                            sport="mlb",
                            rolling_window={
                                "status": p.status,
                                "injury_note": p.injury_note,
                                "percent_owned": p.percent_owned,
                                "adp_updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                            },
                            data_source="yahoo_adp_injury",
                        ))
                    records_written += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("yahoo_adp_injury DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "yahoo_adp_injury: wrote %d rows (%d injury flags) in %dms",
                records_written, injury_flags, elapsed,
            )
            self._record_job_run("yahoo_adp_injury", "success", records_written)
            return {
                "status": "success",
                "records": records_written,
                "injury_flags": injury_flags,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["yahoo_adp_injury"], "yahoo_adp_injury", _run)

    async def _fetch_fangraphs_ros(self) -> dict:
        """Fetch daily Rest-of-Season projections from FanGraphs (lock 100_012).

        Runs at 3 AM ET, before the ensemble blend job at 5 AM.
        Stores raw blend results in a module-level cache so _update_ensemble_blend
        can use them without re-fetching.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.fangraphs_loader import (
                fetch_all_ros, compute_ensemble_blend,
            )

            bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
            pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
            fetched_at = now_et()

            bat_count = sum(len(df) for df in bat_raw.values()) if bat_raw else 0
            pit_count = sum(len(df) for df in pit_raw.values()) if pit_raw else 0

            # Mirror in memory for same-process handoff and persist for cross-process durability.
            _ROS_CACHE["bat"] = bat_raw
            _ROS_CACHE["pit"] = pit_raw
            _ROS_CACHE["fetched_at"] = fetched_at

            try:
                _store_persisted_ros_cache(bat_raw, pit_raw, fetched_at)
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.error("fangraphs_ros: failed to persist durable cache: %s", exc)
                self._record_job_run("fangraphs_ros", "failed")
                return {
                    "status": "failed",
                    "bat_rows": bat_count,
                    "pit_rows": pit_count,
                    "elapsed_ms": elapsed,
                    "error": "durable cache persist failed",
                }

            elapsed = int((time.monotonic() - t0) * 1000)
            status = "ok" if (bat_raw or pit_raw) else "failed"
            logger.info(
                "fangraphs_ros: %d bat / %d pit rows from %d/%d systems; status=%s",
                bat_count, pit_count,
                len(bat_raw) + len(pit_raw), 8,  # 4 systems × 2 stat types
                status,
            )
            if status == "failed":
                logger.warning("fangraphs_ros: all FanGraphs fetches failed — cloudscraper or network issue")
            self._record_job_run("fangraphs_ros", status, bat_count + pit_count)
            return {"status": status, "bat_rows": bat_count, "pit_rows": pit_count, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["fangraphs_ros"], "fangraphs_ros", _run)

    async def _update_ensemble_blend(self) -> dict:
        """Compute weighted ensemble blend and persist to PlayerDailyMetric (lock 100_014).

        Runs at 5 AM ET, 2 hours after _fetch_fangraphs_ros.
        Uses the durable Fangraphs cache from the earlier fetch when available.
        Blend columns written: blend_hr, blend_rbi, blend_avg, blend_era, blend_whip.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.fangraphs_loader import (
                fetch_all_ros, compute_ensemble_blend,
            )

            # Use cached RoS data if fresh (< 4 hours); otherwise re-fetch
            bat_raw = None
            pit_raw = None
            cached_at = _ROS_CACHE.get("fetched_at")
            if cached_at:
                age_h = (now_et() - cached_at).total_seconds() / 3600
                if age_h < 4:
                    bat_raw = _ROS_CACHE.get("bat")
                    pit_raw = _ROS_CACHE.get("pit")

            if not bat_raw and not pit_raw:
                persisted_bat, persisted_pit, persisted_at = _load_persisted_ros_cache()
                if persisted_at is not None:
                    age_h = (now_et() - persisted_at).total_seconds() / 3600
                    if age_h < 4:
                        bat_raw = persisted_bat
                        pit_raw = persisted_pit
                        cached_at = persisted_at
                        _ROS_CACHE["bat"] = bat_raw
                        _ROS_CACHE["pit"] = pit_raw
                        _ROS_CACHE["fetched_at"] = persisted_at

            if not bat_raw and not pit_raw:
                logger.info("ensemble_update: cache miss — re-fetching FanGraphs RoS")
                bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
                pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
                if bat_raw or pit_raw:
                    fetched_at = now_et()
                    _ROS_CACHE["bat"] = bat_raw
                    _ROS_CACHE["pit"] = pit_raw
                    _ROS_CACHE["fetched_at"] = fetched_at
                    _store_persisted_ros_cache(bat_raw, pit_raw, fetched_at)

            if not bat_raw and not pit_raw:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.error("ensemble_update: no RoS data available — skipping blend")
                self._record_job_run("ensemble_update", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            bat_blend = compute_ensemble_blend(bat_raw or {}, stat_columns=["HR", "R", "RBI", "SB", "AVG"]) if bat_raw else None
            pit_blend = compute_ensemble_blend(pit_raw or {}, stat_columns=["ERA", "WHIP"]) if pit_raw else None

            today = today_et()
            db = SessionLocal()
            bat_rows, bat_skipped = _extract_blend_rows(
                bat_blend,
                {"HR": "blend_hr", "RBI": "blend_rbi", "AVG": "blend_avg"},
            )
            pit_rows, pit_skipped = _extract_blend_rows(
                pit_blend,
                {"ERA": "blend_era", "WHIP": "blend_whip"},
            )
            records_written = 0
            inserted = 0
            updated = 0
            skipped = bat_skipped + pit_skipped
            errors = 0
            try:
                all_rows = bat_rows + pit_rows
                existing_ids: set[str] = set()
                if all_rows:
                    existing_ids = {
                        player_id
                        for (player_id,) in (
                            db.query(PlayerDailyMetric.player_id)
                            .filter(
                                PlayerDailyMetric.metric_date == today,
                                PlayerDailyMetric.sport == "mlb",
                                PlayerDailyMetric.player_id.in_([row["player_id"] for row in all_rows]),
                            )
                            .all()
                        )
                    }

                for row in all_rows:
                    stmt = pg_insert(PlayerDailyMetric.__table__).values(
                        player_id=row["player_id"],
                        player_name=row["player_name"],
                        metric_date=today,
                        sport="mlb",
                        rolling_window={},
                        data_source="ensemble_blend",
                        fetched_at=now_et(),
                        blend_hr=row.get("blend_hr"),
                        blend_rbi=row.get("blend_rbi"),
                        blend_avg=row.get("blend_avg"),
                        blend_era=row.get("blend_era"),
                        blend_whip=row.get("blend_whip"),
                    ).on_conflict_do_update(
                        index_elements=["player_id", "metric_date", "sport"],
                        set_={
                            "player_name": row["player_name"],
                            "data_source": "ensemble_blend",
                            "fetched_at": now_et(),
                            "blend_hr": row.get("blend_hr"),
                            "blend_rbi": row.get("blend_rbi"),
                            "blend_avg": row.get("blend_avg"),
                            "blend_era": row.get("blend_era"),
                            "blend_whip": row.get("blend_whip"),
                        },
                    )
                    try:
                        with db.begin_nested():
                            db.execute(stmt)
                        if row["player_id"] in existing_ids:
                            updated += 1
                        else:
                            inserted += 1
                            existing_ids.add(row["player_id"])
                        records_written += 1
                    except Exception as row_exc:
                        errors += 1
                        logger.warning(
                            "ensemble_update: skip row %s -- %s",
                            row.get("player_id"),
                            row_exc,
                        )

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("ensemble_update DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("ensemble_update", "failed")
                return {
                    "status": "failed",
                    "records": 0,
                    "elapsed_ms": elapsed,
                    "inserted": inserted,
                    "updated": updated,
                    "skipped": skipped,
                    "errors": errors,
                }
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "ensemble_update: wrote %d player blend rows in %dms (inserted=%d updated=%d skipped=%d errors=%d)",
                records_written,
                elapsed,
                inserted,
                updated,
                skipped,
                errors,
            )
            self._record_job_run("ensemble_update", "success", records_written)
            return {
                "status": "success",
                "records": records_written,
                "elapsed_ms": elapsed,
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
                "errors": errors,
            }

        return await _with_advisory_lock(LOCK_IDS["ensemble_update"], "ensemble_update", _run)

    async def _check_projection_freshness(self) -> dict:
        """
        SLA gate: warn when projection data is stale but do NOT block anything.
        SLAs: ensemble_blend ≤ 12 h, statcast ≤ 6 h, Fangraphs RoS cache ≤ 12 h.
        Results stored in self._job_status["projection_freshness"] for /admin/ingestion/status.
        """
        t0 = time.monotonic()

        async def _run():
            from datetime import datetime
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo("America/New_York"))
            violations: list[str] = []
            report: dict = {"checked_at": now.isoformat(), "violations": violations}

            db = SessionLocal()
            try:
                # --- ensemble_blend SLA (12 hours) ---
                SLA_ENSEMBLE_H = 12
                result = db.execute(
                    text(
                        "SELECT MAX(metric_date) FROM player_daily_metrics "
                        "WHERE data_source = 'ensemble_blend'"
                    )
                )
                latest_ensemble = result.scalar()
                if latest_ensemble is None:
                    msg = "ensemble_blend: no rows found — pipeline may not have run yet"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)
                else:
                    if hasattr(latest_ensemble, "tzinfo") and latest_ensemble.tzinfo is None:
                        latest_ensemble = latest_ensemble.replace(tzinfo=ZoneInfo("America/New_York"))
                    age_h = (now - latest_ensemble).total_seconds() / 3600
                    report["ensemble_blend_age_h"] = round(age_h, 1)
                    if age_h > SLA_ENSEMBLE_H:
                        msg = f"ensemble_blend stale: {age_h:.1f}h > SLA {SLA_ENSEMBLE_H}h"
                        logger.warning("PROJECTION FRESHNESS: %s", msg)
                        violations.append(msg)

                # --- statcast SLA (6 hours) ---
                SLA_STATCAST_H = 6
                result = db.execute(
                    text(
                        "SELECT MAX(metric_date) FROM player_daily_metrics "
                        "WHERE data_source = 'statcast'"
                    )
                )
                latest_statcast = result.scalar()
                if latest_statcast is None:
                    msg = "statcast: no rows found — statcast ingestion may not have run yet"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)
                else:
                    if hasattr(latest_statcast, "tzinfo") and latest_statcast.tzinfo is None:
                        latest_statcast = latest_statcast.replace(tzinfo=ZoneInfo("America/New_York"))
                    age_h = (now - latest_statcast).total_seconds() / 3600
                    report["statcast_age_h"] = round(age_h, 1)
                    if age_h > SLA_STATCAST_H:
                        msg = f"statcast stale: {age_h:.1f}h > SLA {SLA_STATCAST_H}h"
                        logger.warning("PROJECTION FRESHNESS: %s", msg)
                        violations.append(msg)
            finally:
                db.close()

            # --- persisted Fangraphs RoS cache SLA (12 hours) ---
            SLA_ROS_H = 12
            _, _, ros_fetched_at = _load_persisted_ros_cache(include_payload=False)
            if ros_fetched_at is None:
                msg = "fangraphs_ros cache missing — durable RoS cache has not been persisted yet"
                logger.warning("PROJECTION FRESHNESS: %s", msg)
                violations.append(msg)
            else:
                if hasattr(ros_fetched_at, "tzinfo") and ros_fetched_at.tzinfo is None:
                    ros_fetched_at = ros_fetched_at.replace(tzinfo=ZoneInfo("America/New_York"))
                age_h = (now - ros_fetched_at).total_seconds() / 3600
                report["ros_cache_age_h"] = round(age_h, 1)
                if age_h > SLA_ROS_H:
                    msg = f"fangraphs_ros cache stale: {age_h:.1f}h > SLA {SLA_ROS_H}h"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)

            elapsed = round((time.monotonic() - t0) * 1000)
            report["elapsed_ms"] = elapsed
            report["violation_count"] = len(violations)

            if not violations:
                logger.debug("PROJECTION FRESHNESS: all SLAs met (checked in %d ms)", elapsed)

            self._job_status["projection_freshness"] = report
            return {"status": "success", **report}

        return await _with_advisory_lock(LOCK_IDS["projection_freshness"], "projection_freshness", _run)

    async def _compute_clv(self) -> dict:
        """
        Run nightly CLV attribution and persist a ProjectionSnapshot summary.
        Delegates computation to compute_daily_clv_attribution() in clv.py.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.clv import compute_daily_clv_attribution, CLVAttributionError

            try:
                result = await compute_daily_clv_attribution()
            except CLVAttributionError as exc:
                logger.error("_compute_clv CLVAttributionError: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("clv", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            # Persist summary to ProjectionSnapshot
            yesterday = today_et() - timedelta(days=1)
            db = SessionLocal()
            try:
                snapshot = ProjectionSnapshot(
                    snapshot_date=yesterday,
                    sport="cbb",
                    player_changes={
                        "clv_summary": {
                            "clv_positive": result.get("clv_positive", 0),
                            "clv_negative": result.get("clv_negative", 0),
                            "avg_clv_points": result.get("avg_clv_points", 0.0),
                            "favorable_rate": result.get("favorable_rate", 0.0),
                        }
                    },
                    total_players=result.get("games_processed", 0),
                    significant_changes=result.get("clv_negative", 0),
                )
                db.add(snapshot)
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_compute_clv snapshot write error: %s", exc)
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            records = result.get("games_processed", 0)
            self._record_job_run("clv", "success", records)
            result["elapsed_ms"] = elapsed
            return result

        return await _with_advisory_lock(LOCK_IDS["clv"], "clv", _run)

    async def _cleanup_old_metrics(self) -> dict:
        """
        Delete player_daily_metrics rows older than 90 days to keep the table lean.
        """
        t0 = time.monotonic()

        async def _run():
            cutoff = today_et() - timedelta(days=90)
            db = SessionLocal()
            try:
                result = db.execute(
                    text(
                        "DELETE FROM player_daily_metrics WHERE metric_date < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                deleted = result.rowcount
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_cleanup_old_metrics error: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("cleanup", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("_cleanup_old_metrics: deleted %d rows older than %s", deleted, cutoff)
            self._record_job_run("cleanup", "success", deleted)
            return {"status": "success", "records": deleted, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["cleanup"], "cleanup", _run)

    async def _refresh_valuation_cache(self) -> None:
        """
        Refresh player valuation cache for all configured leagues.
        Advisory lock: 100_011.
        """
        from backend.fantasy_baseball.valuation_worker import run_valuation_worker

        league_str = os.getenv("FANTASY_LEAGUES", "")
        if not league_str:
            logger.info("valuation_cache: FANTASY_LEAGUES not set -- skipping")
            return

        leagues = [lk.strip() for lk in league_str.split(",") if lk.strip()]

        async def _run():
            results = []
            for lk in leagues:
                try:
                    result = await run_valuation_worker(lk)
                    results.append(result)
                except Exception as exc:
                    logger.error("valuation_cache: failed for league=%s (%s)", lk, exc)
            return results

        self._job_status["valuation_cache"] = {
            "name": "valuation_cache",
            "enabled": True,
            "last_run": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "last_status": "running",
            "next_run": self._get_next_run("valuation_cache"),
        }

        try:
            results = await _with_advisory_lock(LOCK_IDS["valuation_cache"], _run)
            self._job_status["valuation_cache"]["last_status"] = "ok"
            logger.info("valuation_cache: complete -- %s", results)
        except Exception as exc:
            self._job_status["valuation_cache"]["last_status"] = f"error: {exc}"
            logger.error("valuation_cache: job failed (%s)", exc)

    async def _sync_position_eligibility(self) -> dict:
        """
        Sync position eligibility from Yahoo Fantasy API (lock 100_027).

        Every sync (daily 8:00 AM ET):
          1. Fetch all rosters via get_league_rosters() — flat list of player dicts
          2. Build boolean position flags from positions list
          3. Upsert ONE ROW PER PLAYER keyed on yahoo_player_key

        Natural key: yahoo_player_key (unique constraint _pe_yahoo_uc).
        Critical for H2H One Win UI — CF scarcity calculations depend on this data.
        """
        logger.info("SYNC JOB ENTRY: _sync_position_eligibility - Starting position eligibility sync")
        t0 = time.monotonic()

        # Position scarcity priority for primary_position selection
        _POSITION_PRIORITY = ["C", "SS", "2B", "CF", "3B", "RF", "LF", "1B", "OF", "DH", "SP", "RP", "Util"]
        _BATTER_POS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH"}

        async def _run():
            from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
            try:
                logger.info("SYNC JOB PROGRESS: _sync_position_eligibility - Initializing Yahoo client")
                yahoo = YahooFantasyClient()
            except Exception as exc:
                logger.error("SYNC JOB ERROR: _sync_position_eligibility - Yahoo client init failed: %s", exc)
                self._record_job_run("position_eligibility", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            league_key = yahoo.league_key
            if not league_key:
                logger.warning("_sync_position_eligibility: Yahoo client league_key not set -- skipping")
                self._record_job_run("position_eligibility", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            try:
                all_players = await asyncio.to_thread(
                    yahoo.get_league_rosters,
                    league_key=league_key,
                    include_team_key=True
                )
            except Exception as exc:
                logger.error("_sync_position_eligibility: Failed to fetch rosters (%s)", exc)
                self._record_job_run("position_eligibility", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}

            db = SessionLocal()
            records_processed = 0
            seen_keys = set()

            try:
                now = now_et()
                for player_data in all_players:
                    try:
                        player_key = player_data.get("player_key")
                        if not player_key or player_key in seen_keys:
                            continue
                        seen_keys.add(player_key)

                        name = player_data.get("name", "Unknown")
                        positions = player_data.get("positions", [])
                        if not positions:
                            continue

                        # Build boolean flags from positions list
                        pos_set = {p.upper() for p in positions if p}
                        flags = {
                            "can_play_c": "C" in pos_set,
                            "can_play_1b": "1B" in pos_set,
                            "can_play_2b": "2B" in pos_set,
                            "can_play_3b": "3B" in pos_set,
                            "can_play_ss": "SS" in pos_set,
                            "can_play_lf": "LF" in pos_set,
                            "can_play_cf": "CF" in pos_set,
                            "can_play_rf": "RF" in pos_set,
                            "can_play_of": "OF" in pos_set or bool(pos_set & {"LF", "CF", "RF"}),
                            "can_play_dh": "DH" in pos_set,
                            "can_play_util": "UTIL" in pos_set or "Util" in {p for p in positions if p},
                            "can_play_sp": "SP" in pos_set,
                            "can_play_rp": "RP" in pos_set,
                        }

                        # Primary position by scarcity
                        pos_upper = [p.upper() for p in positions if p]
                        primary = next((pr for pr in _POSITION_PRIORITY if pr.upper() in pos_upper), positions[0] if positions else "DH")

                        # Player type classification
                        has_pitcher = bool(pos_set & {"SP", "RP", "P"})
                        has_batter = bool(pos_set & _BATTER_POS)
                        if has_pitcher and has_batter:
                            ptype = "two_way"
                        elif has_pitcher:
                            ptype = "pitcher"
                        else:
                            ptype = "batter"

                        # Multi-eligibility count (exclude Util)
                        multi_count = len([p for p in positions if p.upper() != "UTIL"])

                        # Upsert: ON CONFLICT (yahoo_player_key) DO UPDATE
                        stmt = pg_insert(PositionEligibility.__table__).values(
                            yahoo_player_key=player_key,
                            bdl_player_id=None,
                            player_name=name,
                            first_name="",
                            last_name="",
                            primary_position=primary,
                            player_type=ptype,
                            multi_eligibility_count=multi_count,
                            fetched_at=now,
                            updated_at=now,
                            **flags,
                        ).on_conflict_do_update(
                            constraint="_pe_yahoo_uc",
                            set_={
                                "player_name": name,
                                "primary_position": primary,
                                "player_type": ptype,
                                "multi_eligibility_count": multi_count,
                                "updated_at": now,
                                **flags,
                            },
                        )
                        db.execute(stmt)
                        records_processed += 1

                    except Exception as exc:
                        logger.error("_sync_position_eligibility: Failed to process player %s (%s)",
                                     player_data.get("player_key", "?"), exc)
                        continue

                db.commit()
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info("SYNC JOB SUCCESS: _sync_position_eligibility - Processed %d records in %d ms", records_processed, elapsed)
                logger.info("SYNC JOB EXIT: _sync_position_eligibility - Completed successfully")
                self._record_job_run("position_eligibility", "success", records_processed)
                return {"status": "success", "records": records_processed, "elapsed_ms": elapsed}

            except Exception as exc:
                db.rollback()
                logger.error("_sync_position_eligibility: Database error (%s)", exc)
                self._record_job_run("position_eligibility", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}
            finally:
                db.close()

        try:
            return await _with_advisory_lock(LOCK_IDS["position_eligibility"], "position_eligibility", _run)
        except Exception as exc:
            logger.error("_sync_position_eligibility: Job failed (%s)", exc)
            self._record_job_run("position_eligibility", "error", 0)
            return {"status": "error", "records": 0, "elapsed_ms": 0}

    async def _sync_probable_pitchers(self) -> dict:
        """
        Sync probable pitchers from MLB Stats API (lock 100_028).

        Every sync (daily 8:30 AM ET, 4:00 PM ET, 8:00 PM ET):
          1. Fetch schedule from MLB Stats API for next 7 days with probablePitcher hydration
          2. Extract pitcher name + MLBAM ID directly from API response
          3. Resolve BDL player ID via PlayerIDMapping (mlbam_id lookup)
          4. Upsert to probable_pitchers table on (game_date, team)

        Uses MLB Stats API because BDL does not expose probable pitcher data (K-37 confirmed).
        Pattern proven in daily_lineup_optimizer.py._fetch_probable_pitchers_for_date().

        Natural key: (game_date, team). One probable pitcher per team per date.
        """
        logger.info("SYNC JOB ENTRY: _sync_probable_pitchers - Starting probable pitchers sync")
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.ballpark_factors import get_park_factor

            # Team abbreviation aliases (MLB Stats API -> our standard)
            abbr_aliases = {
                "TBR": "TB", "KCR": "KC", "SFG": "SF", "SDP": "SD",
                "WSN": "WSH", "AZ": "ARI", "CHW": "CWS",
            }

            def _normalize_abbr(abbr: str) -> str:
                if not abbr:
                    return ""
                up = abbr.upper()
                return abbr_aliases.get(up, up)

            today = today_et()
            records_processed = 0
            api_errors = 0
            inferred_records = 0
            official_records = 0

            db = SessionLocal()
            try:
                # Pre-load MLBAM -> BDL ID mapping for fast lookups
                mlbam_to_bdl = {}
                mappings = db.query(
                    PlayerIDMapping.mlbam_id, PlayerIDMapping.bdl_id
                ).filter(
                    PlayerIDMapping.mlbam_id.isnot(None),
                    PlayerIDMapping.bdl_id.isnot(None),
                ).all()
                for m in mappings:
                    mlbam_to_bdl[m.mlbam_id] = m.bdl_id
                logger.info("_sync_probable_pitchers: Loaded %d MLBAM->BDL mappings", len(mlbam_to_bdl))

                recent_starter_candidates = build_recent_starter_candidates(db, today)
                logger.info(
                    "_sync_probable_pitchers: Built fallback starter candidates for %d teams",
                    len(recent_starter_candidates),
                )

                # Fetch schedule for next 7 days from MLB Stats API
                for days_ahead in range(7):
                    target_date = today + timedelta(days=days_ahead)
                    date_str = target_date.strftime("%Y-%m-%d")

                    try:
                        resp = await asyncio.to_thread(
                            requests.get,
                            "https://statsapi.mlb.com/api/v1/schedule",
                            params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as exc:
                        logger.error("_sync_probable_pitchers: MLB Stats API failed for %s (%s)", date_str, exc)
                        api_errors += 1
                        continue

                    for date_info in data.get("dates", []):
                        for game in date_info.get("games", []):
                            teams_data = game.get("teams", {})

                            # Extract game time -- gameDate is ISO8601 UTC
                            game_datetime_str = game.get("gameDate", "")
                            game_time_et_str = None
                            if game_datetime_str:
                                try:
                                    utc_dt = datetime.fromisoformat(game_datetime_str.replace("Z", "+00:00"))
                                    et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
                                    game_time_et_str = et_dt.strftime("%-I:%M %p")
                                except (ValueError, OSError):
                                    # Windows strftime doesn't support %-I
                                    try:
                                        utc_dt = datetime.fromisoformat(game_datetime_str.replace("Z", "+00:00"))
                                        et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
                                        game_time_et_str = et_dt.strftime("%I:%M %p").lstrip("0")
                                    except Exception:
                                        pass

                            # Process both sides (home and away)
                            for side, is_home in [("home", True), ("away", False)]:
                                opp_side = "away" if is_home else "home"
                                side_data = teams_data.get(side, {})
                                opp_data = teams_data.get(opp_side, {})

                                team_abbr = _normalize_abbr(
                                    side_data.get("team", {}).get("abbreviation", "")
                                )
                                opp_abbr = _normalize_abbr(
                                    opp_data.get("team", {}).get("abbreviation", "")
                                )

                                if not team_abbr:
                                    continue

                                pitcher_data = side_data.get("probablePitcher", {})
                                inferred_candidate = None
                                if pitcher_data:
                                    pitcher_name = pitcher_data.get("fullName", "")
                                    mlbam_id = pitcher_data.get("id")
                                    bdl_id = mlbam_to_bdl.get(mlbam_id) if mlbam_id else None
                                    official_records += 1
                                else:
                                    inferred_candidate = infer_probable_pitcher_for_team(
                                        recent_starter_candidates,
                                        team_abbr,
                                        target_date,
                                    )
                                    if inferred_candidate is None:
                                        continue
                                    pitcher_name = inferred_candidate.pitcher_name
                                    mlbam_id = inferred_candidate.mlbam_id
                                    bdl_id = inferred_candidate.bdl_player_id
                                    inferred_records += 1

                                # Resolve BDL ID via MLBAM mapping when only official MLBAM is known
                                if bdl_id is None and mlbam_id:
                                    bdl_id = mlbam_to_bdl.get(mlbam_id)

                                # Park factor: home team's park
                                home_abbr = team_abbr if is_home else opp_abbr
                                pf = get_park_factor(home_abbr, "era")

                                try:
                                    stmt = pg_insert(ProbablePitcherSnapshot).values(
                                        game_date=target_date,
                                        team=team_abbr,
                                        opponent=opp_abbr,
                                        is_home=is_home,
                                        pitcher_name=pitcher_name,
                                        bdl_player_id=bdl_id,
                                        mlbam_id=mlbam_id,
                                        is_confirmed=False,
                                        game_time_et=game_time_et_str,
                                        park_factor=pf,
                                        quality_score=None,
                                        fetched_at=now_et(),
                                        updated_at=now_et(),
                                    )
                                    stmt = stmt.on_conflict_do_update(
                                        constraint="_pp_date_team_uc",
                                        set_={
                                            "opponent": stmt.excluded.opponent,
                                            "is_home": stmt.excluded.is_home,
                                            "pitcher_name": stmt.excluded.pitcher_name,
                                            "bdl_player_id": stmt.excluded.bdl_player_id,
                                            "mlbam_id": stmt.excluded.mlbam_id,
                                            "game_time_et": stmt.excluded.game_time_et,
                                            "park_factor": stmt.excluded.park_factor,
                                            "updated_at": stmt.excluded.updated_at,
                                        },
                                    )
                                    db.execute(stmt)
                                    records_processed += 1
                                except Exception as exc:
                                    logger.error(
                                        "_sync_probable_pitchers: Failed to upsert %s %s (%s)",
                                        team_abbr, date_str, exc,
                                    )
                                    continue

                db.commit()
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info(
                    "SYNC JOB SUCCESS: _sync_probable_pitchers - %d records (%d official, %d inferred), %d API errors, %d ms",
                    records_processed, official_records, inferred_records, api_errors, elapsed,
                )
                self._record_job_run("probable_pitchers", "success", records_processed)
                return {
                    "status": "success",
                    "records": records_processed,
                    "official_records": official_records,
                    "inferred_records": inferred_records,
                    "api_errors": api_errors,
                    "elapsed_ms": elapsed,
                }

            except Exception as exc:
                db.rollback()
                logger.error("_sync_probable_pitchers: Database error (%s)", exc)
                self._record_job_run("probable_pitchers", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}
            finally:
                db.close()

        try:
            return await _with_advisory_lock(LOCK_IDS["probable_pitchers"], "probable_pitchers", _run)
        except Exception as exc:
            logger.error("_sync_probable_pitchers: Job failed (%s)", exc)
            self._record_job_run("probable_pitchers", "error", 0)
            return {"status": "error", "records": 0, "elapsed_ms": 0}

    async def _sync_player_id_mapping(self) -> dict:
        """
        Sync player ID mappings from BDL + MLB Stats API cross-reference (lock 100_029).

        Every sync (daily 7:00 AM ET):
          1. Fetch all players from BDL (GOAT tier: 600 req/min)
          2. For each player: get MLBAM ID from MLB Stats API player endpoint
          3. Store cross-reference in player_id_mapping table

        Critical for data integration — connects BDL, MLB Stats, and Yahoo namespaces.
        Natural key: (source_system, source_id).
        """
        logger.info("SYNC JOB ENTRY: _sync_player_id_mapping - Starting player ID mapping sync")
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("SYNC JOB ERROR: _sync_player_id_mapping - BDL client initialization failed: %s", exc)
                self._record_job_run("player_id_mapping", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            try:
                # Fetch all MLB players from BDL
                players = await asyncio.to_thread(bdl.get_all_mlb_players)
            except Exception as exc:
                logger.error("_sync_player_id_mapping: Failed to fetch players from BDL (%s)", exc)
                self._record_job_run("player_id_mapping", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}

            db = SessionLocal()
            records_processed = 0

            try:
                for player in players:
                    try:
                        bdl_id = player.id
                        if not bdl_id:
                            continue

                        # Get MLBAM ID from player data
                        mlbam_id = getattr(player, 'mlbam_id', None)

                        # Get full name
                        full_name = player.full_name
                        normalized_name = full_name.lower()

                        # Create/update mapping record
                        # We'll store both BDL and Yahoo mappings, plus MLBAM
                        existing = (
                            db.query(PlayerIDMapping)
                            .filter(PlayerIDMapping.bdl_id == bdl_id)
                            .first()
                        )
                        if existing:
                            existing.mlbam_id = mlbam_id
                            existing.full_name = full_name
                            existing.normalized_name = normalized_name
                            existing.resolution_confidence = 1.0
                        else:
                            mapping = PlayerIDMapping(
                                bdl_id=bdl_id,
                                yahoo_key=None,  # Will be populated by Yahoo sync
                                mlbam_id=mlbam_id,
                                full_name=full_name,
                                normalized_name=normalized_name,
                                source='api',
                                resolution_confidence=1.0,  # Direct from BDL API
                            )
                            db.add(mapping)
                        records_processed += 1

                    except Exception as exc:
                        logger.error("_sync_player_id_mapping: Failed to process player %s (%s)", player, exc)
                        continue

                db.commit()
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info("SYNC JOB SUCCESS: _sync_player_id_mapping - Processed %d records in %d ms", records_processed, elapsed)
                logger.info("SYNC JOB EXIT: _sync_player_id_mapping - Completed successfully")
                self._record_job_run("player_id_mapping", "success", records_processed)
                return {"status": "success", "records": records_processed, "elapsed_ms": elapsed}

            except Exception as exc:
                db.rollback()
                logger.error("_sync_player_id_mapping: Database error (%s)", exc)
                self._record_job_run("player_id_mapping", "error", 0)
                return {"status": "error", "records": 0, "elapsed_ms": 0}
            finally:
                db.close()

        try:
            return await _with_advisory_lock(LOCK_IDS["player_id_mapping"], "player_id_mapping", _run)
        except Exception as exc:
            logger.error("_sync_player_id_mapping: Job failed (%s)", exc)
            self._record_job_run("player_id_mapping", "error", 0)
            return {"status": "error", "records": 0, "elapsed_ms": 0}

    def _start_openclaw_monitoring(self) -> None:
        """
        Initialize OpenClaw Phase 1 monitoring (Performance Monitor + Pattern Detector).
        
        This is read-only monitoring that does NOT violate the Guardian freeze.
        Self-improvement features (Phase 4) remain disabled until Apr 7, 2026.
        """
        try:
            from backend.services.openclaw.scheduler import OpenClawScheduler
            
            self._openclaw = OpenClawScheduler(
                scheduler=self._scheduler,
                sport='cbb',  # Primary focus during tournament season
                discord_hook=self._send_discord_alert if os.getenv('DISCORD_ALERTS_ENABLED') else None
            )
            self._openclaw.start_monitoring()
            
            logger.info("OpenClaw Phase 1 monitoring started (Performance Monitor + Pattern Detector)")
        except Exception as exc:
            logger.warning("OpenClaw monitoring not started: %s", exc)
    
    def _send_discord_alert(self, embed: dict) -> None:
        """Send Discord alert via webhook."""
        webhook_url = os.getenv('DISCORD_ALERTS_WEBHOOK')
        if not webhook_url:
            return
        
        try:
            requests.post(
                webhook_url,
                json={"embeds": [embed]},
                timeout=5
            )
        except Exception as exc:
            logger.warning("Discord alert failed: %s", exc)
