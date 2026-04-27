"""
Dashboard Service — Phase B Foundation

Aggregates all dashboard data for the fantasy baseball home screen.

Usage:
    from backend.services.dashboard_service import DashboardService
    service = DashboardService()
    dashboard = await service.get_dashboard(user_id, team_key)
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from sqlalchemy.orm import Session

from backend.models import UserPreferences, SessionLocal, PlayerDailyMetric
from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer
from backend.services.waiver_edge_detector import WaiverEdgeDetector
from backend.services.data_reliability_engine import (
    get_reliability_engine,
    DataQualityTier,
    DataSource,
)
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient, YahooAuthError

logger = logging.getLogger(__name__)


@dataclass
class LineupGap:
    """Identifies an unfilled lineup slot."""
    position: str
    severity: str  # "critical", "warning", "info"
    message: str
    suggested_add: Optional[str] = None


@dataclass
class StreakPlayer:
    """Player with hot/cold streak information."""
    player_id: str
    name: str
    team: str
    positions: List[str]
    trend: str  # "hot", "cold", "neutral"
    trend_score: float  # z-score
    last_7_avg: float
    last_14_avg: float
    last_30_avg: float
    reason: str


@dataclass
class WaiverTarget:
    """Prioritized waiver wire target."""
    player_id: str
    name: str
    team: str
    positions: List[str]
    percent_owned: float
    priority_score: float
    tier: str  # "must_add", "strong_add", "streamer"
    reason: str


@dataclass
class InjuryFlag:
    """Injury alert for a rostered player."""
    player_id: str
    name: str
    status: str  # "IL", "IL10", "IL60", "DTD", "OUT"
    injury_note: Optional[str]
    severity: str  # "critical", "warning", "info"
    estimated_return: Optional[str]
    action_needed: str


@dataclass
class MatchupPreview:
    """This week's matchup outlook."""
    week_number: int
    opponent_team_name: str
    opponent_record: str
    my_projected_categories: Dict[str, float]
    opponent_projected_categories: Dict[str, float]
    win_probability: float
    category_advantages: List[str]
    category_disadvantages: List[str]


@dataclass
class ProbablePitcherInfo:
    """Pitcher start information."""
    name: str
    team: str
    opponent: str
    game_date: str
    is_two_start: bool
    matchup_quality: str  # "favorable", "neutral", "unfavorable"
    stream_score: float
    reason: str


@dataclass
class DashboardData:
    """Complete dashboard payload."""
    timestamp: str
    user_id: str
    
    # B1.1: Lineup Gaps
    lineup_gaps: List[LineupGap]
    lineup_filled_count: int
    lineup_total_count: int
    
    # B1.2: Hot/Cold Streaks
    hot_streaks: List[StreakPlayer]
    cold_streaks: List[StreakPlayer]
    
    # B1.3: Waiver Targets
    waiver_targets: List[WaiverTarget]
    
    # B1.4: Injury Flags
    injury_flags: List[InjuryFlag]
    healthy_count: int
    injured_count: int
    
    # B1.5: Matchup Preview
    matchup_preview: Optional[MatchupPreview]
    
    # B1.6: Probable Pitchers
    probable_pitchers: List[ProbablePitcherInfo]
    two_start_pitchers: List[ProbablePitcherInfo]
    
    # Settings
    preferences: Dict[str, Any]


class DashboardService:
    """
    Aggregates dashboard data from multiple sources with reliability validation.
    
    This service coordinates:
    - Yahoo API for roster/matchup data (with fallback handling)
    - Statcast for streak analysis (with freshness validation)
    - WaiverEdgeDetector for FA recommendations
    - DailyLineupOptimizer for lineup gaps
    - DataReliabilityEngine for quality scoring
    """
    
    def __init__(self):
        self.lineup_optimizer = DailyLineupOptimizer()
        self.waiver_detector = WaiverEdgeDetector()
        self.reliability_engine = get_reliability_engine()
        self._yahoo_client: Optional[YahooFantasyClient] = None
    
    def _get_yahoo_client(self) -> Optional[YahooFantasyClient]:
        """Get Yahoo client with lazy initialization and error handling."""
        if self._yahoo_client is None:
            try:
                self._yahoo_client = YahooFantasyClient()
                self.reliability_engine.record_source_success(DataSource.YAHOO_API)
            except YahooAuthError as e:
                logger.warning(f"Yahoo auth not available: {e}")
                self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
                return None
        return self._yahoo_client
    
    async def get_dashboard(
        self,
        user_id: str,
        team_key: Optional[str] = None,
        db: Optional[Session] = None
    ) -> DashboardData:
        """
        Build complete dashboard data for a user.
        
        Args:
            user_id: Unique user identifier
            team_key: Yahoo team key (optional, will try to detect)
            db: Database session (optional, will create if not provided)
        
        Returns:
            DashboardData with all panels populated
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            # Load user preferences
            prefs = self._get_or_create_preferences(db, user_id)
            
            # Gather all dashboard components in parallel
            (
                (lineup_gaps, filled, total),
                (hot_streaks, cold_streaks),
                waiver_targets,
                (injury_flags, healthy, injured),
                matchup,
                (pitchers, two_starts),
            ) = await asyncio.gather(
                self._get_lineup_gaps(user_id, team_key),
                self._get_streaks(user_id),
                self._get_waiver_targets(user_id, prefs),
                self._get_injury_flags(user_id),
                self._get_matchup_preview(user_id, team_key),
                self._get_probable_pitchers(user_id),
            )
            
            return DashboardData(
                timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
                user_id=user_id,
                lineup_gaps=lineup_gaps,
                lineup_filled_count=filled,
                lineup_total_count=total,
                hot_streaks=hot_streaks[:5],  # Top 5 hot
                cold_streaks=cold_streaks[:5],  # Top 5 cold
                waiver_targets=waiver_targets[:5],  # Top 5 targets
                injury_flags=injury_flags,
                healthy_count=healthy,
                injured_count=injured,
                matchup_preview=matchup,
                probable_pitchers=pitchers,
                two_start_pitchers=two_starts,
                preferences=self._prefs_to_dict(prefs)
            )
        
        finally:
            if close_db:
                db.close()
    
    def _get_or_create_preferences(self, db: Session, user_id: str) -> UserPreferences:
        """Get existing preferences or create defaults."""
        prefs = db.query(UserPreferences).filter_by(user_id=user_id).first()
        if prefs is None:
            prefs = UserPreferences(user_id=user_id)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
        return prefs
    
    def _prefs_to_dict(self, prefs: UserPreferences) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        return {
            "notifications": prefs.notifications,
            "dashboard_layout": prefs.dashboard_layout,
            "streak_settings": prefs.streak_settings,
            "waiver_preferences": prefs.waiver_preferences,
        }
    
    async def _get_lineup_gaps(
        self,
        user_id: str,
        team_key: Optional[str]
    ) -> tuple[List[LineupGap], int, int]:
        """
        B1.1: Detect unfilled lineup positions using Yahoo API.
        
        Returns:
            (gaps list, filled count, total slots)
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot detect lineup gaps")
            return [], 0, 9
        
        try:
            # Get roster from Yahoo
            roster = client.get_roster(team_key) if team_key else client.get_roster()
            
            # Validate roster data
            validation = self.reliability_engine.validate_yahoo_roster(
                roster, timestamp=datetime.utcnow()
            )
            
            if not validation.is_valid:
                logger.warning(f"Roster validation failed: {validation.errors}")
            
            # Define required positions for Yahoo H2H
            required_positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "Util"]
            
            # Get active players (not on IL)
            active_players = [
                p for p in roster 
                if p.get("selected_position") not in ("IL", "IL10", "IL60")
            ]
            
            # Map players to positions they can fill
            gaps = []
            filled_count = 0
            
            for req_pos in required_positions:
                # Find a player eligible for this position
                eligible = [
                    p for p in active_players 
                    if req_pos in p.get("positions", []) or 
                    (req_pos == "Util" and any(pos in ["C", "1B", "2B", "3B", "SS", "OF"] for pos in p.get("positions", [])))
                ]
                
                if eligible:
                    filled_count += 1
                else:
                    # Gap detected
                    severity = "critical" if req_pos in ("C", "SS") else "warning"
                    gaps.append(LineupGap(
                        position=req_pos,
                        severity=severity,
                        message=f"No eligible player for {req_pos} slot",
                        suggested_add=None  # Would need waiver wire analysis
                    ))
            
            return gaps, filled_count, len(required_positions)
            
        except Exception as e:
            logger.error(f"Failed to get lineup gaps: {e}")
            self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
            return [], 0, 9
    
    async def _get_streaks(
        self, 
        user_id: str, 
        db: Optional[Session] = None
    ) -> tuple[List[StreakPlayer], List[StreakPlayer]]:
        """
        B1.2: Calculate hot/cold streaks from Statcast data.
        
        Uses 7/14/30 day rolling windows from player_daily_metrics.
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            # Get player's rostered players first
            client = self._get_yahoo_client()
            roster = []
            if client:
                try:
                    roster = client.get_roster()
                except Exception as e:
                    logger.warning(f"Could not fetch roster for streaks: {e}")
            
            roster_scoped = bool(roster)

            # Query recent metrics from database
            recent_date = datetime.utcnow().date() - timedelta(days=1)
            metrics_q = db.query(PlayerDailyMetric).filter(
                PlayerDailyMetric.metric_date >= recent_date - timedelta(days=30),
                PlayerDailyMetric.sport == "mlb"
            )
            if roster_scoped:
                roster_names = {p.get("name", "").lower() for p in roster}
                metrics_q = metrics_q.filter(
                    PlayerDailyMetric.player_name.in_(roster_names)
                )
            metrics = metrics_q.all()

            if not metrics:
                return [], []
            
            hot = []
            cold = []

            # Build roster lookup for team/positions enrichment (empty when Yahoo unavailable)
            roster_lookup: dict = {}
            if roster_scoped:
                for p in roster:
                    roster_lookup[p.get("name", "").lower()] = p

            # Dedupe: keep most-recent row per player
            latest_per_player: dict = {}
            for m in metrics:
                key = m.player_id
                if key not in latest_per_player or m.metric_date > latest_per_player[key].metric_date:
                    latest_per_player[key] = m

            for latest in latest_per_player.values():
                # Validate data quality
                validation = self.reliability_engine.validate_statcast_data(
                    latest.player_id,
                    {
                        "player_id": latest.player_id,
                        "player_name": latest.player_name,
                        "game_date": latest.metric_date.isoformat(),
                        "exit_velocity_avg": latest.bat_speed or 0,
                    },
                    timestamp=datetime.combine(latest.metric_date, datetime.min.time())
                )

                # Only use data if quality is acceptable
                if validation.quality_tier in (DataQualityTier.TIER_4_STALE, DataQualityTier.TIER_5_UNAVAILABLE):
                    logger.debug(f"Skipping stale data for {latest.player_name}")
                    continue

                # Calculate trend
                z_score = latest.z_score_recent or 0

                # Get rolling averages
                rolling = latest.rolling_window or {}
                last_7 = rolling.get("7d", {}).get("avg", 0)
                last_14 = rolling.get("14d", {}).get("avg", 0)
                last_30 = rolling.get("30d", {}).get("avg", 0)

                roster_entry = roster_lookup.get(latest.player_name.lower(), {})
                streak_player = StreakPlayer(
                    player_id=latest.player_id,
                    name=latest.player_name,
                    team=roster_entry.get("team", ""),
                    positions=roster_entry.get("positions", []),
                    trend="hot" if z_score > 0.5 else "cold" if z_score < -0.5 else "neutral",
                    trend_score=z_score,
                    last_7_avg=last_7,
                    last_14_avg=last_14,
                    last_30_avg=last_30,
                    reason=f"z-score: {z_score:.2f} (data quality: {validation.quality_tier.value})"
                )

                if z_score > 0.5:
                    hot.append(streak_player)
                elif z_score < -0.5:
                    cold.append(streak_player)
            
            # Sort by trend score
            hot.sort(key=lambda x: x.trend_score, reverse=True)
            cold.sort(key=lambda x: x.trend_score)
            
            return hot, cold
            
        finally:
            if close_db:
                db.close()
    
    async def _get_waiver_targets(
        self,
        user_id: str,
        prefs: UserPreferences
    ) -> List[WaiverTarget]:
        """
        B1.3: Get prioritized waiver wire recommendations via WaiverEdgeDetector.
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot get waiver targets")
            return []

        try:
            my_roster = client.get_roster()

            # Attempt to get opponent roster to improve category deficit scoring
            opponent_roster: List[dict] = []
            try:
                scoreboard = client.get_scoreboard()
                my_team_key = client.get_my_team_key()
                for matchup in scoreboard:
                    teams_raw = matchup.get("teams", {})
                    opp_key = self._extract_opponent_key(teams_raw, my_team_key)
                    if opp_key:
                        opponent_roster = client.get_roster(opp_key)
                        break
            except Exception as e:
                logger.debug(f"Opponent roster unavailable for waiver scoring: {e}")

            moves = self.waiver_detector.get_top_moves(
                my_roster, opponent_roster, n_candidates=10
            )

            targets = []
            for move in moves:
                fa = move.get("add_player") or {}
                if not fa:
                    continue

                need_score = float(move.get("need_score", 0.0))
                win_gain = float(move.get("win_prob_gain", 0.0))
                priority_score = need_score + win_gain * 100

                if priority_score > 2.0:
                    tier = "must_add"
                elif priority_score > 1.0:
                    tier = "strong_add"
                else:
                    tier = "streamer"

                drop_name = move.get("drop_player_name", "")
                reason_parts = [f"Need score: {need_score:.2f}"]
                if drop_name:
                    reason_parts.append(f"Drop: {drop_name}")
                if win_gain:
                    reason_parts.append(f"Win gain: {win_gain:+.1%}")
                reason = " | ".join(reason_parts)

                targets.append(WaiverTarget(
                    player_id=str(fa.get("player_id") or fa.get("player_key", "")),
                    name=fa.get("name", "Unknown"),
                    team=fa.get("team", ""),
                    positions=fa.get("positions", []),
                    percent_owned=float(fa.get("percent_owned", 0.0)),
                    priority_score=priority_score,
                    tier=tier,
                    reason=reason,
                ))

            return targets

        except Exception as e:
            logger.error(f"Failed to get waiver targets: {e}")
            return []
    
    async def _get_injury_flags(self, user_id: str) -> tuple[List[InjuryFlag], int, int]:
        """
        B1.4: Detect injured players on roster using Yahoo API.
        
        Returns:
            (injury flags, healthy count, injured count)
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot detect injuries")
            return [], 0, 0
        
        try:
            roster = client.get_roster()
            
            # Validate roster data
            validation = self.reliability_engine.validate_yahoo_roster(roster)
            if not validation.is_valid:
                logger.warning(f"Roster validation failed for injury check: {validation.errors}")
            
            flags = []
            healthy = 0
            injured = 0
            
            # Status mappings
            injury_statuses = {"IL", "IL10", "IL60", "DTD", "OUT", "NA"}
            
            for player in roster:
                status = player.get("status", "")
                selected_pos = player.get("selected_position", "")
                
                # Check if player is injured
                is_injured = status in injury_statuses or selected_pos in ("IL", "IL10", "IL60")
                
                if is_injured:
                    injured += 1
                    
                    # Determine severity
                    if status in ("IL", "IL60") or selected_pos in ("IL", "IL60"):
                        severity = "critical"
                        action = "Move to IL slot immediately"
                    elif status == "IL10" or selected_pos == "IL10":
                        severity = "warning"
                        action = "Consider moving to IL slot"
                    elif status == "DTD":
                        severity = "warning"
                        action = "Check lineup status before lock"
                    else:
                        severity = "info"
                        action = "Monitor status"
                    
                    flags.append(InjuryFlag(
                        player_id=player.get("player_id", ""),
                        name=player.get("name", "Unknown"),
                        status=status or selected_pos or "OUT",
                        injury_note=player.get("injury_note"),
                        severity=severity,
                        estimated_return=None,  # Would need additional data source
                        action_needed=action
                    ))
                else:
                    healthy += 1
            
            return flags, healthy, injured
            
        except Exception as e:
            logger.error(f"Failed to get injury flags: {e}")
            self.reliability_engine.record_source_failure(DataSource.YAHOO_API, str(e))
            return [], 0, 0
    
    async def _get_matchup_preview(
        self,
        user_id: str,
        team_key: Optional[str],
        db: Optional[Session] = None,
    ) -> Optional[MatchupPreview]:
        """
        B1.5: Get this week's matchup analysis from Yahoo scoreboard.

        Enrichments:
        - opponent_record: pulled from Yahoo standings (W-L-T format)
        - my_projected_categories / opponent_projected_categories: 7-day rolling
          averages from PlayerDailyMetric for rostered players on each side.
        - win_probability: MCMC-simulated using cat_scores from player board
          or PlayerDailyMetric z-scores (B5 — calibrated March 30, 2026)
        """
        client = self._get_yahoo_client()
        if not client:
            return None

        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            my_team_key = team_key or client.get_my_team_key()
            scoreboard = client.get_scoreboard()

            if not scoreboard:
                return None

            # Find our matchup in the scoreboard
            opponent_key = ""
            week_number = 1
            for matchup in scoreboard:
                teams_raw = matchup.get("teams", {})
                opp_key = self._extract_opponent_key(teams_raw, my_team_key)
                if opp_key is not None:
                    opponent_key = opp_key
                    # Extract week number — Yahoo nests this differently per version
                    week_raw = matchup.get("week") or matchup.get("week_number", 1)
                    try:
                        week_number = int(week_raw)
                    except (TypeError, ValueError):
                        week_number = 1
                    break
            else:
                return None  # Our team not found in scoreboard

            # ── Resolve opponent team name ────────────────────────────────
            opponent_name = "Opponent"
            if opponent_key:
                try:
                    all_teams = client.get_all_teams()
                    opp = next((t for t in all_teams if t.get("team_key") == opponent_key), None)
                    if opp:
                        opponent_name = opp.get("name", "Opponent")
                except Exception:
                    pass

            # ── Populate opponent_record from standings ───────────────────
            opponent_record = ""
            if opponent_key:
                try:
                    opponent_record = self._fetch_team_record(client, opponent_key)
                except Exception as exc:
                    logger.debug("Could not fetch opponent record: %s", exc)

            # ── Populate projected categories from PlayerDailyMetric ──────
            my_projected_categories: Dict[str, float] = {}
            opponent_projected_categories: Dict[str, float] = {}
            try:
                my_roster = client.get_roster(team_key=my_team_key) if my_team_key else []
                opp_roster = client.get_roster(team_key=opponent_key) if opponent_key else []

                my_projected_categories = self._project_categories_from_db(
                    db, [p.get("name", "") for p in my_roster if p.get("name")]
                )
                opponent_projected_categories = self._project_categories_from_db(
                    db, [p.get("name", "") for p in opp_roster if p.get("name")]
                )
            except Exception as exc:
                logger.debug("Could not compute projected categories: %s", exc)

            # ── Calculate win probability via MCMC simulation ─────────────
            win_prob = 0.5
            cat_advantages: List[str] = []
            cat_disadvantages: List[str] = []
            try:
                from backend.fantasy_baseball.mcmc_calibration import (
                    calculate_matchup_win_probability,
                )
                mcmc_result = calculate_matchup_win_probability(
                    my_roster, opp_roster, db=db, n_sims=1000, seed=42
                )
                win_prob = mcmc_result.get("win_prob", 0.5)
                
                # Derive category advantages/disadvantages from simulation
                cat_win_probs = mcmc_result.get("category_win_probs", {})
                for cat, prob in cat_win_probs.items():
                    if prob > 0.6:
                        cat_advantages.append(cat.upper())
                    elif prob < 0.4:
                        cat_disadvantages.append(cat.upper())
            except Exception as exc:
                logger.debug("MCMC win probability calculation failed: %s", exc)
                win_prob = 0.5  # Fallback to 50/50

            return MatchupPreview(
                week_number=week_number,
                opponent_team_name=opponent_name,
                opponent_record=opponent_record,
                my_projected_categories=my_projected_categories,
                opponent_projected_categories=opponent_projected_categories,
                win_probability=win_prob,
                category_advantages=cat_advantages,
                category_disadvantages=cat_disadvantages,
            )

        except Exception as e:
            logger.error(f"Failed to get matchup preview: {e}")
            return None
        finally:
            if close_db:
                db.close()

    @staticmethod
    def _extract_team_standings(team_list: Any) -> tuple:
        """
        Recursively flatten a Yahoo team standings entry and extract
        team_key and outcome_totals.

        Returns (team_key: str, outcome: dict).
        """
        found_key: List[str] = []
        outcome: Dict = {}

        def _walk(obj: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(obj, list):
                for item in obj:
                    _walk(item, depth + 1)
            elif isinstance(obj, dict):
                if "team_key" in obj and not found_key:
                    found_key.append(str(obj["team_key"]))
                if "outcome_totals" in obj:
                    outcome.update(obj["outcome_totals"])
                for v in obj.values():
                    if isinstance(v, (list, dict)):
                        _walk(v, depth + 1)

        _walk(team_list)
        return (found_key[0] if found_key else ""), outcome

    @staticmethod
    def _fetch_team_record(client: Any, team_key: str) -> str:
        """
        Query Yahoo standings and extract the W-L-T record for a given team_key.

        Returns a string like "12-3-0" or "" if the data cannot be resolved.
        The standings endpoint includes team_standings.outcome_totals which Yahoo
        always populates (even at the start of the season with 0-0-0).
        """
        try:
            raw = client._get(f"league/{client.league_key}/standings")
            sec = client._league_section(raw, 1)
            teams_raw = sec.get("standings", [{}])[0].get("teams", {})
            count = int(teams_raw.get("count", 0))
            for i in range(count):
                entry = teams_raw.get(str(i), {})
                team_list = entry.get("team", [])
                tk, outcome = DashboardService._extract_team_standings(team_list)
                if tk == team_key and outcome:
                    w = outcome.get("wins", 0)
                    l = outcome.get("losses", 0)
                    t = outcome.get("ties", 0)
                    return f"{w}-{l}-{t}"
        except Exception as exc:
            logger.debug("_fetch_team_record error: %s", exc)
        return ""

    @staticmethod
    def _project_categories_from_db(
        db: Session,
        player_names: List[str],
    ) -> Dict[str, float]:
        """Aggregate 7-day rolling average stats from PlayerDailyMetric."""
        if not player_names:
            return {}

        try:
            recent_cutoff = date.today() - timedelta(days=2)
            metrics = (
                db.query(PlayerDailyMetric)
                .filter(
                    PlayerDailyMetric.player_name.in_(player_names),
                    PlayerDailyMetric.sport == "mlb",
                    PlayerDailyMetric.metric_date >= recent_cutoff,
                )
                .all()
            )

            if not metrics:
                return {}

            # Keep most-recent row per player
            latest: Dict[str, PlayerDailyMetric] = {}
            for m in metrics:
                existing = latest.get(m.player_name)
                if existing is None or m.metric_date > existing.metric_date:
                    latest[m.player_name] = m

            totals: Dict[str, float] = {}
            for m in latest.values():
                window = m.rolling_window or {}
                seven_day = window.get("7d", {})
                avg_block = seven_day.get("avg", {})
                if isinstance(avg_block, dict):
                    for cat, val in avg_block.items():
                        try:
                            totals[cat] = totals.get(cat, 0.0) + float(val)
                        except (TypeError, ValueError):
                            pass
                elif isinstance(avg_block, (int, float)):
                    # Scalar avg — use z_score_recent as a fallback signal
                    z = m.z_score_recent
                    if z is not None:
                        totals["z_score"] = totals.get("z_score", 0.0) + float(z)

            return totals
        except Exception as exc:
            logger.debug("_project_categories_from_db error: %s", exc)
            return {}
    
    async def _get_probable_pitchers(
        self,
        user_id: str
    ) -> tuple[List[ProbablePitcherInfo], List[ProbablePitcherInfo]]:
        """
        B1.6: Get probable pitchers and two-start SPs via DailyLineupOptimizer.

        Two-start detection: counts starts across the next 7 calendar days.
        """
        client = self._get_yahoo_client()
        if not client:
            logger.warning("Yahoo client unavailable - cannot get probable pitchers")
            return [], []

        try:
            roster = client.get_roster()
            today = datetime.now(ZoneInfo("America/New_York"))
            today_str = today.strftime("%Y-%m-%d")

            # Flag today's starters
            pitcher_data = self.lineup_optimizer.flag_pitcher_starts(roster, game_date=today_str)

            # Count starts over rolling 7-day window for two-start detection
            start_counts: Dict[str, int] = {}
            for day_offset in range(7):
                check_date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                try:
                    week_pitchers = self.lineup_optimizer.flag_pitcher_starts(
                        roster, game_date=check_date
                    )
                    for p in week_pitchers:
                        if p.get("has_start") and p.get("pitcher_slot") == "SP":
                            start_counts[p.get("name", "")] = (
                                start_counts.get(p.get("name", ""), 0) + 1
                            )
                except Exception:
                    pass  # Best-effort; a missing day doesn't break today's display

            pitchers: List[ProbablePitcherInfo] = []
            two_starts: List[ProbablePitcherInfo] = []

            for p in pitcher_data:
                if not p.get("has_start") or p.get("pitcher_slot") != "SP":
                    continue

                name = p.get("name", "Unknown")
                team = p.get("team", "")
                is_two_start = start_counts.get(name, 1) >= 2

                info = ProbablePitcherInfo(
                    name=name,
                    team=team,
                    opponent="",  # Cross-ref with odds map deferred to next pass
                    game_date=today_str,
                    is_two_start=is_two_start,
                    matchup_quality="neutral",  # Pitcher quality scoring deferred
                    stream_score=float(start_counts.get(name, 1)),
                    reason="Two-start week" if is_two_start else "Starting today",
                )
                pitchers.append(info)
                if is_two_start:
                    two_starts.append(info)

            return pitchers, two_starts

        except Exception as e:
            logger.error(f"Failed to get probable pitchers: {e}")
            return [], []
    
    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_opponent_key(teams_raw: Any, my_team_key: str) -> Optional[str]:
        """
        Given a Yahoo scoreboard 'teams' block (dict or list), return the
        team_key of the opponent, or None if my_team_key is not in this matchup.

        Returns None  -> our team is not in this matchup (skip it).
        Returns ""    -> our team is here but opponent key could not be resolved.
        """
        team_keys: List[str] = []

        if isinstance(teams_raw, dict):
            count = int(teams_raw.get("count", 0))
            for i in range(count):
                entry = teams_raw.get(str(i), {})
                team_list = entry.get("team", [])
                meta: Dict = {}
                items = team_list if isinstance(team_list, list) else [team_list]
                for item in items:
                    if isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict):
                                meta.update(sub)
                    elif isinstance(item, dict):
                        meta.update(item)
                key = meta.get("team_key", "")
                if key:
                    team_keys.append(key)
        elif isinstance(teams_raw, list):
            for entry in teams_raw:
                if isinstance(entry, dict):
                    key = entry.get("team_key", "")
                    if key:
                        team_keys.append(key)

        if my_team_key not in team_keys:
            return None  # Signal: wrong matchup

        opp_keys = [k for k in team_keys if k != my_team_key]
        return opp_keys[0] if opp_keys else ""

    # ---------------------------------------------------------------------
    # User Preferences CRUD
    # ---------------------------------------------------------------------
    
    def get_preferences(self, user_id: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """Get user preferences."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            prefs = self._get_or_create_preferences(db, user_id)
            return self._prefs_to_dict(prefs)
        finally:
            if close_db:
                db.close()
    
    def update_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any],
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            updates: Dict with keys matching preference fields
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            prefs = self._get_or_create_preferences(db, user_id)
            
            # Update allowed fields
            if "notifications" in updates:
                prefs.notifications = {**prefs.notifications, **updates["notifications"]}
            if "dashboard_layout" in updates:
                prefs.dashboard_layout = {**prefs.dashboard_layout, **updates["dashboard_layout"]}
            if "projection_weights" in updates:
                prefs.projection_weights = {**prefs.projection_weights, **updates["projection_weights"]}
            if "streak_settings" in updates:
                prefs.streak_settings = {**prefs.streak_settings, **updates["streak_settings"]}
            if "waiver_preferences" in updates:
                prefs.waiver_preferences = {**prefs.waiver_preferences, **updates["waiver_preferences"]}
            
            db.commit()
            db.refresh(prefs)
            
            return self._prefs_to_dict(prefs)
        finally:
            if close_db:
                db.close()


# Singleton instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service() -> DashboardService:
    """Get singleton dashboard service."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
