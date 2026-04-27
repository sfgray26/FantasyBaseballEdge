"""Pipeline freshness validator -- checks table health for critical fantasy tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo


@dataclass
class TableHealth:
    table_name: str
    row_count: int
    expected_min_rows: int
    latest_date: Optional[date]
    max_staleness_days: Optional[int]
    is_healthy: bool
    issues: list = field(default_factory=list)


# (model_attr_for_date, max_staleness_days, min_rows)
_TABLE_SPECS: list[tuple[str, str, Optional[str], Optional[int], int]] = [
    # (model_class_name, table_name, date_column_attr, max_staleness_days, min_rows)
    ("PlayerRollingStats",      "player_rolling_stats",   "as_of_date",  2,  1000),
    ("PlayerScore",             "player_scores",          "as_of_date",  2,  1000),
    ("StatcastPerformance",     "statcast_performances",  "game_date",   3,  5000),
    ("ProbablePitcherSnapshot", "probable_pitchers",      "game_date",   1,  1),
    ("SimulationResult",        "simulation_results",     "as_of_date",  2,  100),
    ("MLBPlayerStats",          "mlb_player_stats",       "game_date",   None, 2000),
    ("DataIngestionLog",        "data_ingestion_logs",    "target_date", 1,  1),
]


def check_table_health(
    db: Session,
    today: Optional[date] = None,
) -> list[TableHealth]:
    """Check freshness and row counts for all critical fantasy tables."""
    from backend.models import (
        MLBPlayerStats,
        PlayerRollingStats,
        PlayerScore,
        ProbablePitcherSnapshot,
        SimulationResult,
        StatcastPerformance,
        DataIngestionLog,
    )

    model_map = {
        "PlayerRollingStats": PlayerRollingStats,
        "PlayerScore": PlayerScore,
        "StatcastPerformance": StatcastPerformance,
        "ProbablePitcherSnapshot": ProbablePitcherSnapshot,
        "SimulationResult": SimulationResult,
        "MLBPlayerStats": MLBPlayerStats,
        "DataIngestionLog": DataIngestionLog,
    }

    if today is None:
        today = datetime.now(ZoneInfo("America/New_York")).date()

    results: list[TableHealth] = []

    for class_name, table_name, date_col, max_stale, min_rows in _TABLE_SPECS:
        model_cls = model_map[class_name]
        issues: list[str] = []

        # Row count
        row_count: int = db.query(func.count(model_cls.id)).scalar() or 0

        if row_count < min_rows:
            issues.append(
                f"Row count {row_count} below minimum {min_rows}"
            )

        # Freshness (if date column exists)
        latest_date: Optional[date] = None
        if date_col is not None:
            col = getattr(model_cls, date_col)
            latest_date = db.query(func.max(col)).scalar()

            if max_stale is not None and latest_date is not None:
                stale_days = (today - latest_date).days
                if stale_days > max_stale:
                    issues.append(
                        f"Stale by {stale_days} days (latest: {latest_date})"
                    )

        is_healthy = len(issues) == 0

        results.append(
            TableHealth(
                table_name=table_name,
                row_count=row_count,
                expected_min_rows=min_rows,
                latest_date=latest_date,
                max_staleness_days=max_stale,
                is_healthy=is_healthy,
                issues=issues,
            )
        )

    return results


def pipeline_health_summary(checks: list[TableHealth]) -> dict:
    """Summarize into JSON-serializable dict with overall_healthy flag."""
    healthy = [c for c in checks if c.is_healthy]
    unhealthy = [c for c in checks if not c.is_healthy]

    return {
        "overall_healthy": len(unhealthy) == 0,
        "healthy_count": len(healthy),
        "unhealthy_count": len(unhealthy),
        "tables": [
            {
                "name": c.table_name,
                "healthy": c.is_healthy,
                "row_count": c.row_count,
                "latest_date": c.latest_date.isoformat() if c.latest_date else None,
                "issues": c.issues,
            }
            for c in checks
        ],
    }
