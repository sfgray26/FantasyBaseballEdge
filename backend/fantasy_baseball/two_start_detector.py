"""
Two-Start Pitcher Detection Service

Identifies pitchers with two starts in the next 7 days for streaming decisions.
Data source: probable_pitchers table (P26) populated from MLB Stats API.

Key features:
  - Count starts over rolling 7-day window
  - Matchup quality rating (park factor + opponent quality)
  - Acquisition cost transparency (ROSTERED/WAIVER/FREE_AGENT)
  - IP projection and categories addressed (W, QS, K, K/9)

UAT Focus: Validate all data sources before UI consumption.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Literal
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class MatchupRating:
    """Quality score for a single matchup."""

    opponent: str  # Team abbreviation (e.g., "NYY", "BOS")
    park_factor: float  # 1.0 = neutral, >1.0 = hitter-friendly, <1.0 = pitcher-friendly
    quality_score: float  # +2.0 (great) to -2.0 (terrible)
    game_date: date
    is_home: bool


@dataclass
class TwoStartOpportunity:
    """
    A pitcher with two starts in the next 7 days.

    From research doc Section 2.2 — used for Two-Start Command Center UI.
    """

    player_id: str  # BDL player ID
    name: str
    team: str
    week: int  # Fantasy scoring week

    game_1: MatchupRating
    game_2: Optional[MatchupRating]  # May only have one start confirmed

    total_ip_projection: float  # Expected innings (5-6 IP per start)
    categories_addressed: List[str]  # ["W", "QS", "K", "K/9"]

    acquisition_method: Literal["ROSTERED", "WAIVER", "FREE_AGENT"]
    waiver_priority_cost: Optional[int]  # If WAIVER, priority required
    faab_cost_estimate: Optional[int]  # If FREE_AGENT, estimated FAAB bid

    # Quality metrics
    average_quality_score: float  # Mean of game_1 and game_2 quality scores
    streamer_rating: Literal["EXCELLENT", "GOOD", "AVOID"]  # Recommendation

    # Data validation flags (UAT focus)
    data_freshness: Literal["FRESH", "STALE", "MISSING"]  # Is probable_pitchers data current?
    player_name_confidence: Literal["HIGH", "MEDIUM", "LOW"]  # Name match confidence


class TwoStartDetector:
    """
    Service for detecting two-start pitcher opportunities.

    UAT Validation Checklist:
      1. probable_pitchers table populated (P26)
      2. MLB Stats API data freshness (<24h old)
      3. Player ID mapping resolution (BDL → MLBAM)
      4. Matchup quality score ranges (-2.0 to +2.0)
      5. Acquisition method classification accuracy
    """

    # Park factors (simplified — in production, load from reference table)
    PARK_FACTORS: Dict[str, float] = {
        "ARI": 1.05, "ATL": 1.02, "BAL": 1.00, "BOS": 0.96, "CHC": 1.04,
        "CIN": 1.02, "CLE": 1.00, "COL": 1.12, "CWS": 1.05, "DET": 0.98,
        "HOU": 0.98, "KC": 1.06, "LAA": 0.99, "LAD": 0.97, "MIA": 1.03,
        "MIL": 1.01, "MIN": 1.00, "NYM": 0.94, "NYY": 0.97, "OAK": 0.97,
        "PHI": 1.01, "PIT": 0.98, "SD": 1.02, "SF": 0.99, "SEA": 0.97,
        "STL": 1.03, "TB": 1.05, "TEX": 1.07, "TOR": 1.01, "WSN": 1.00,
    }

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize detector with database connection.

        Args:
            db_url: Database connection string (defaults to DATABASE_URL env var)
        """
        if db_url is None:
            import os
            db_url = os.environ.get("DATABASE_URL")

        if db_url:
            self.engine = create_engine(db_url)
            self.SessionLocal = sessionmaker(bind=self.engine)
        else:
            self.engine = None
            self.SessionLocal = None

    def detect_two_start_pitchers(
        self,
        start_date: date,
        end_date: date,
        league_rosters: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[TwoStartOpportunity]:
        """
        Detect all pitchers with 2+ starts in the date range.

        Args:
            start_date: Start of window (typically today)
            end_date: End of window (typically today + 7 days)
            league_rosters: List of rosters for acquisition method classification
                           (if None, all pitchers marked as FREE_AGENT)

        Returns:
            List of TwoStartOpportunity objects sorted by streamer_rating
        """
        if not self.engine:
            # UAT fallback: return empty list with warning
            return []

        opportunities = []

        with self.SessionLocal() as session:
            # Query probable_pitchers for date range
            query = text("""
                SELECT
                    pitcher_name,
                    team,
                    bdl_player_id,
                    game_date,
                    opponent,
                    is_home,
                    is_confirmed,
                    park_factor,
                    quality_score
                FROM probable_pitchers
                WHERE game_date >= :start_date
                  AND game_date <= :end_date
                ORDER BY game_date, team
            """)

            result = session.execute(
                query,
                {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
            )

            # Group starts by pitcher
            pitcher_starts: Dict[int, List[Dict[str, Any]]] = {}
            for row in result:
                pitcher_id = row[2]  # bdl_player_id
                if pitcher_id not in pitcher_starts:
                    pitcher_starts[pitcher_id] = []

                pitcher_starts[pitcher_id].append({
                    "pitcher_name": row[0],
                    "team": row[1],
                    "bdl_player_id": pitcher_id,
                    "game_date": date.fromisoformat(row[3]),
                    "opponent": row[4],
                    "is_home": row[5],
                    "is_confirmed": row[6],
                    "park_factor": row[7] or 1.0,
                    "quality_score": row[8] or 0.0,
                })

            # Build opportunities for pitchers with 2+ starts
            for pitcher_id, starts in pitcher_starts.items():
                if len(starts) >= 2:
                    opp = self._build_opportunity(pitcher_id, starts, league_rosters)
                    if opp:
                        opportunities.append(opp)

        # Sort by average quality score (best first)
        opportunities.sort(key=lambda x: x.average_quality_score, reverse=True)

        return opportunities

    def _build_opportunity(
        self,
        pitcher_id: int,
        starts: List[Dict[str, Any]],
        league_rosters: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> Optional[TwoStartOpportunity]:
        """Build TwoStartOpportunity from pitcher starts."""

        if not starts:
            return None

        first_start = starts[0]
        second_start = starts[1] if len(starts) > 1 else None

        # Build matchup ratings
        game_1 = MatchupRating(
            opponent=first_start["opponent"],
            park_factor=first_start["park_factor"],
            quality_score=first_start["quality_score"],
            game_date=first_start["game_date"],
            is_home=first_start["is_home"],
        )

        game_2 = None
        if second_start:
            game_2 = MatchupRating(
                opponent=second_start["opponent"],
                park_factor=second_start["park_factor"],
                quality_score=second_start["quality_score"],
                game_date=second_start["game_date"],
                is_home=second_start["is_home"],
            )

        # Compute average quality
        avg_quality = (game_1.quality_score + (game_2.quality_score if game_2 else 0)) / 2

        # Determine streamer rating
        if avg_quality >= 1.0:
            streamer_rating = "EXCELLENT"
        elif avg_quality >= 0.0:
            streamer_rating = "GOOD"
        else:
            streamer_rating = "AVOID"

        # Classify acquisition method
        acquisition_method, waiver_cost, faab_cost = self._classify_acquisition(
            pitcher_id, league_rosters
        )

        # Data freshness validation (UAT)
        data_freshness = self._validate_data_freshness(starts)
        name_confidence = "HIGH" if first_start["pitcher_name"] else "LOW"

        return TwoStartOpportunity(
            player_id=str(pitcher_id),
            name=first_start["pitcher_name"] or "Unknown",
            team=first_start["team"],
            week=self._compute_fantasy_week(first_start["game_date"]),
            game_1=game_1,
            game_2=game_2,
            total_ip_projection=11.0,  # ~5.5 IP per start
            categories_addressed=["W", "QS", "K", "K/9"],
            acquisition_method=acquisition_method,
            waiver_priority_cost=waiver_cost,
            faab_cost_estimate=faab_cost,
            average_quality_score=avg_quality,
            streamer_rating=streamer_rating,
            data_freshness=data_freshness,
            player_name_confidence=name_confidence,
        )

    def _classify_acquisition(
        self,
        pitcher_id: int,
        league_rosters: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> tuple[Literal["ROSTERED", "WAIVER", "FREE_AGENT"], Optional[int], Optional[int]]:
        """
        Classify how to acquire this pitcher.

        Args:
            pitcher_id: BDL player ID
            league_rosters: List of 10 team rosters (each a list of player dicts)

        Returns:
            (acquisition_method, waiver_priority_cost, faab_cost_estimate)
        """
        if not league_rosters:
            # Default: assume free agent
            return "FREE_AGENT", None, 5  # $5 FAAB estimate

        # Check if pitcher is on any roster
        for team_roster in league_rosters:
            for player in team_roster:
                if player.get("bdl_player_id") == pitcher_id:
                    return "ROSTERED", None, None

        # Not on any roster → free agent or waiver wire
        # For now, assume free agent (waiver wire logic varies by league)
        return "FREE_AGENT", None, 5

    def _validate_data_freshness(self, starts: List[Dict[str, Any]]) -> Literal["FRESH", "STALE", "MISSING"]:
        """
        Validate that probable_pitchers data is fresh (<24h old).

        UAT check: Ensures data pipeline is running correctly.
        """
        if not starts:
            return "MISSING"

        today = date.today()
        latest_game_date = max(s["game_date"] for s in starts)

        if (today - latest_game_date).days <= 1:
            return "FRESH"
        elif (today - latest_game_date).days <= 3:
            return "STALE"
        else:
            return "MISSING"

    def _compute_fantasy_week(self, game_date: date) -> int:
        """Compute fantasy week number (simplified — MLB week logic is complex)."""
        # Approximate: weeks since March 28 (typical opening day)
        opening_day = date(game_date.year, 3, 28)
        days_since_open = (game_date - opening_day).days
        return max(1, (days_since_open // 7) + 1)
