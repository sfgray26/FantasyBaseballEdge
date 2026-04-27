"""Shared helpers for resilient probable-pitcher lookups.

This module provides a conservative fallback when official probable starters
are missing from the MLB Stats API. It never attempts to outsmart the rotation;
it only infers a starter when a recent pitcher appearance lines up cleanly with
an exact 5-day rotation cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import MLBPlayerStats, PlayerIDMapping, ProbablePitcherSnapshot


TEAM_ALIASES = {
    "TBR": "TB",
    "KCR": "KC",
    "SFG": "SF",
    "SDP": "SD",
    "WSN": "WSH",
    "AZ": "ARI",
    "CHW": "CWS",
}


@dataclass
class RecentStarterCandidate:
    team: str
    bdl_player_id: Optional[int]
    mlbam_id: Optional[int]
    pitcher_name: str
    last_start_date: date
    typical_ip: float


def normalize_team_abbr(abbr: Optional[str]) -> str:
    if not abbr:
        return ""
    team = abbr.upper()
    return TEAM_ALIASES.get(team, team)


def parse_innings_pitched(ip: Optional[object]) -> Optional[float]:
    """Convert BDL innings-pitched notation into decimal innings."""
    if ip is None:
        return None
    if isinstance(ip, (int, float)):
        return float(ip)
    if isinstance(ip, str):
        parts = ip.split(".")
        try:
            whole = int(parts[0])
            outs = int(parts[1]) if len(parts) > 1 else 0
            return whole + (outs / 3.0)
        except (ValueError, IndexError):
            return None
    return None


def load_probable_pitchers_from_snapshot(db: Session, game_date: date) -> dict[str, str]:
    """Load persisted probable pitchers for a date, keyed by team abbreviation."""
    rows = (
        db.query(ProbablePitcherSnapshot.team, ProbablePitcherSnapshot.pitcher_name)
        .filter(
            ProbablePitcherSnapshot.game_date == game_date,
            ProbablePitcherSnapshot.pitcher_name.isnot(None),
        )
        .all()
    )
    result: dict[str, str] = {}
    for team, pitcher_name in rows:
        if team and pitcher_name:
            result[normalize_team_abbr(team)] = pitcher_name.lower()
    return result


def build_recent_starter_candidates(
    db: Session,
    as_of_date: date,
    lookback_days: int = 14,
    min_starter_ip: float = 4.0,
) -> dict[str, list[RecentStarterCandidate]]:
    """Build a conservative recent-starter map from per-game pitching stats."""
    window_start = as_of_date - timedelta(days=lookback_days)

    name_rows = (
        db.query(PlayerIDMapping.bdl_id, PlayerIDMapping.mlbam_id, PlayerIDMapping.full_name)
        .filter(PlayerIDMapping.bdl_id.isnot(None))
        .all()
    )
    id_map = {
        row.bdl_id: {"mlbam_id": row.mlbam_id, "full_name": row.full_name}
        for row in name_rows
        if row.bdl_id is not None
    }

    stat_rows = (
        db.query(MLBPlayerStats)
        .filter(
            MLBPlayerStats.game_date >= window_start,
            MLBPlayerStats.game_date < as_of_date,
            MLBPlayerStats.innings_pitched.isnot(None),
        )
        .all()
    )

    latest_by_team_player: dict[tuple[str, int], RecentStarterCandidate] = {}

    for row in stat_rows:
        ip_decimal = parse_innings_pitched(row.innings_pitched)
        if ip_decimal is None or ip_decimal < min_starter_ip:
            continue

        payload = row.raw_payload if isinstance(row.raw_payload, dict) else {}
        team_payload = payload.get("team") if isinstance(payload.get("team"), dict) else {}
        player_payload = payload.get("player") if isinstance(payload.get("player"), dict) else {}

        team = normalize_team_abbr(team_payload.get("abbreviation"))
        if not team:
            continue

        bdl_player_id = row.bdl_player_id
        mapping = id_map.get(bdl_player_id, {})
        pitcher_name = (
            player_payload.get("full_name")
            or player_payload.get("name")
            or mapping.get("full_name")
            or ""
        )
        if not pitcher_name:
            continue

        candidate = RecentStarterCandidate(
            team=team,
            bdl_player_id=bdl_player_id,
            mlbam_id=mapping.get("mlbam_id"),
            pitcher_name=pitcher_name,
            last_start_date=row.game_date,
            typical_ip=ip_decimal,
        )

        key = (team, bdl_player_id)
        existing = latest_by_team_player.get(key)
        if existing is None or candidate.last_start_date > existing.last_start_date:
            latest_by_team_player[key] = candidate

    grouped: dict[str, list[RecentStarterCandidate]] = {}
    for candidate in latest_by_team_player.values():
        grouped.setdefault(candidate.team, []).append(candidate)

    for team_candidates in grouped.values():
        team_candidates.sort(
            key=lambda item: (item.last_start_date.toordinal(), item.typical_ip),
            reverse=True,
        )

    return grouped


def infer_probable_pitcher_for_team(
    candidates_by_team: dict[str, list[RecentStarterCandidate]],
    team: str,
    target_date: date,
) -> Optional[RecentStarterCandidate]:
    """Infer a likely starter only on an exact 5-day cadence from a recent start."""
    team_key = normalize_team_abbr(team)
    candidates = candidates_by_team.get(team_key, [])
    exact_cycle = [
        candidate
        for candidate in candidates
        if (target_date - candidate.last_start_date).days >= 5
        and (target_date - candidate.last_start_date).days % 5 == 0
    ]
    if not exact_cycle:
        return None

    exact_cycle.sort(
        key=lambda item: ((target_date - item.last_start_date).days, -item.typical_ip),
    )
    return exact_cycle[0]


def infer_probable_pitcher_map(db: Session, target_date: date) -> dict[str, RecentStarterCandidate]:
    """Infer a map of team -> starter candidate for a target date."""
    candidates = build_recent_starter_candidates(db, target_date)
    inferred: dict[str, RecentStarterCandidate] = {}
    for team in candidates.keys():
        candidate = infer_probable_pitcher_for_team(candidates, team, target_date)
        if candidate is not None:
            inferred[team] = candidate
    return inferred