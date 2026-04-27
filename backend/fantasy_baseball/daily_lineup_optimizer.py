"""
Daily Fantasy Baseball Lineup Optimizer

Uses sportsbook odds from The Odds API to compute implied team run totals,
then ranks batters and pitchers for daily lineup decisions.

Key logic:
  - Game total + spread -> implied runs per team
  - High implied runs -> stack batters from that team
  - Low opponent implied runs + high K/9 -> stream SP
  - Injury filter -> skip IL/DTD players
  - Park factor adjustment (via ballpark_factors.py)

Usage:
    from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer
    opt = DailyLineupOptimizer()
    report = opt.build_daily_report("2026-04-01")
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from backend.models import SessionLocal
from backend.services.probable_pitcher_fallback import (
    infer_probable_pitcher_map,
    load_probable_pitchers_from_snapshot,
)
from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The Odds API — MLB games
# ---------------------------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_SPORT = "baseball_mlb"

# MLB team abbreviation -> full name normalization map (Odds API uses full names)
_TEAM_ABBREV = {
    "NYY": "New York Yankees", "BOS": "Boston Red Sox", "TOR": "Toronto Blue Jays",
    "BAL": "Baltimore Orioles", "TB": "Tampa Bay Rays",
    "CLE": "Cleveland Guardians", "CWS": "Chicago White Sox", "DET": "Detroit Tigers",
    "KC": "Kansas City Royals", "MIN": "Minnesota Twins",
    "HOU": "Houston Astros", "TEX": "Texas Rangers", "SEA": "Seattle Mariners",
    "OAK": "Oakland Athletics", "LAA": "Los Angeles Angels",
    "NYM": "New York Mets", "PHI": "Philadelphia Phillies", "ATL": "Atlanta Braves",
    "MIA": "Miami Marlins", "WSH": "Washington Nationals",
    "MIL": "Milwaukee Brewers", "CHC": "Chicago Cubs", "STL": "St. Louis Cardinals",
    "CIN": "Cincinnati Reds", "PIT": "Pittsburgh Pirates",
    "LAD": "Los Angeles Dodgers", "SF": "San Francisco Giants",
    "SD": "San Diego Padres", "COL": "Colorado Rockies",
    "ARI": "Arizona Diamondbacks",
}
# Reverse: full name -> abbreviation (use Yahoo's preferred abbreviations)
_FULL_TO_ABBREV: Dict[str, str] = {v: k for k, v in _TEAM_ABBREV.items()}

# Additional aliases for common alternate abbreviations
_TEAM_ALIASES = {
    "TBR": "TB",  # Tampa Bay Rays (ESPN/Odds API style)
    "KCR": "KC",  # Kansas City Royals (ESPN/Odds API style)
    "SFG": "SF",  # San Francisco Giants (ESPN/Odds API style)
    "SDP": "SD",  # San Diego Padres (ESPN/Odds API style)
    "WSN": "WSH", # Washington Nationals (ESPN style)
    "AZ": "ARI",  # Arizona Diamondbacks (Yahoo style)
    "CHW": "CWS", # Chicago White Sox (Yahoo/ESPN style -> standard)
}


def normalize_team_abbr(abbr: str) -> str:
    """Normalize team abbreviation to Yahoo standard."""
    if not abbr:
        return ""
    abbr_upper = abbr.upper()
    # First check if it's an alias
    if abbr_upper in _TEAM_ALIASES:
        return _TEAM_ALIASES[abbr_upper]
    # Otherwise return as-is (already in standard form)
    return abbr_upper


# Park run factors (1.0 = neutral; > 1.0 = hitter-friendly)
_PARK_FACTORS: Dict[str, float] = {
    "COL": 1.25, "CIN": 1.10, "TEX": 1.08, "PHI": 1.07, "ARI": 1.06,
    "MIL": 1.05, "NYY": 1.04, "BOS": 1.03, "CHC": 1.02, "TOR": 1.02,
    "STL": 1.00, "ATL": 1.00, "LAD": 1.00, "NYM": 0.99, "DET": 0.99,
    "KC":  0.98, "BAL": 0.98, "WSH": 0.97, "MIN": 0.97, "CWS": 0.97,
    "SEA": 0.96, "CLE": 0.96, "HOU": 0.95, "MIA": 0.95, "OAK": 0.94,
    "TB":  0.93, "SF":  0.92, "SD":  0.92, "PIT": 0.91, "LAA": 0.99,
}


@dataclass
class MLBGameOdds:
    """Parsed odds for one MLB game."""
    game_id: str
    commence_time: str
    home_team: str          # full name
    away_team: str          # full name
    home_abbrev: str
    away_abbrev: str
    spread_home: Optional[float] = None     # negative = home favored
    total: Optional[float] = None
    moneyline_home: Optional[float] = None
    moneyline_away: Optional[float] = None
    # Derived
    implied_home_runs: Optional[float] = None
    implied_away_runs: Optional[float] = None
    park_factor: float = 1.0


@dataclass
class BatterRanking:
    """Daily batter ranking for lineup decisions."""
    name: str
    team: str                   # abbreviation
    positions: List[str]
    implied_team_runs: float    # team's expected runs today
    park_factor: float
    projected_r: float = 0.0
    projected_hr: float = 0.0
    projected_rbi: float = 0.0
    projected_avg: float = 0.0
    is_home: bool = False
    status: Optional[str] = None
    lineup_score: float = 0.0   # composite daily score
    reason: str = ""
    has_game: bool = False      # Whether team plays today


@dataclass
class PitcherRanking:
    """Daily SP streaming ranking."""
    name: str
    team: str
    opponent: str
    implied_opp_runs: float     # lower = better for pitcher
    park_factor: float
    projected_k: float = 0.0
    projected_era: float = 0.0
    projected_ip: float = 0.0
    is_home: bool = False
    status: Optional[str] = None
    stream_score: float = 0.0
    reason: str = ""


@dataclass
class LineupSlotResult:
    """One filled lineup slot from the constraint solver."""
    slot: str               # "C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "BN"
    player_name: str
    player_team: str
    positions: List[str]
    lineup_score: float
    implied_runs: float
    park_factor: float
    has_game: bool          # True if team plays today
    status: Optional[str]  # injury status from Yahoo
    reason: str             # human-readable explanation


# ---------------------------------------------------------------------------
# Yahoo H2H standard slot config: fill scarcest positions first so that
# multi-eligible players (e.g. Castro 2B/3B) cover whatever gap remains.
# ---------------------------------------------------------------------------
_DEFAULT_BATTER_SLOTS: List[Tuple[str, List[str]]] = [
    ("C",    ["C"]),
    ("1B",   ["1B"]),
    ("2B",   ["2B"]),
    ("3B",   ["3B"]),
    ("SS",   ["SS"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("OF",   ["OF", "LF", "CF", "RF"]),
    ("Util", ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"]),
]

# Statuses that mean "occupying an IL slot, not an active roster spot"
_INACTIVE_STATUSES = frozenset({"IL", "IL10", "IL60", "NA", "OUT"})


class DailyLineupOptimizer:
    """
    Combines sportsbook odds with projection data to rank
    batters and pitchers for daily fantasy lineup decisions.
    """

    def __init__(self):
        self._api_key = os.getenv("THE_ODDS_API_KEY", "")
        self._odds_cache: Dict[str, List[MLBGameOdds]] = {}

    # ------------------------------------------------------------------
    # Odds fetching
    # ------------------------------------------------------------------

    def fetch_mlb_odds(self, game_date: Optional[str] = None) -> List[MLBGameOdds]:
        """
        Fetch today's MLB game odds from The Odds API.

        Returns list of MLBGameOdds with implied run totals computed.
        Falls back to empty list if API key missing or request fails.
        """
        if not self._api_key:
            logger.warning("THE_ODDS_API_KEY not set — lineup optimizer running without odds data")
            return []

        cache_key = game_date or "today"
        if cache_key in self._odds_cache:
            return self._odds_cache[cache_key]

        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{MLB_SPORT}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": "spreads,totals,h2h",
                    "oddsFormat": "american",
                    "commenceTimeTo": f"{game_date}T23:59:59Z" if game_date else None,
                    "commenceTimeFrom": f"{game_date}T00:00:00Z" if game_date else None,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Odds API returned %d for MLB odds", resp.status_code)
                return []

            games_raw = resp.json()
            games = []
            for g in games_raw:
                game = self._parse_game_odds(g)
                if game:
                    games.append(game)

            self._odds_cache[cache_key] = games
            matchup_str = ", ".join(f"{g.away_abbrev}@{g.home_abbrev}" for g in games)
            logger.info("Odds API [%s]: %d games — %s", game_date or "today", len(games), matchup_str or "NONE")
            if not games:
                logger.warning("Odds API returned 0 games for %s — check API key / coverage", game_date or "today")
            return games

        except Exception as exc:
            logger.warning("Failed to fetch MLB odds: %s", exc)
            return []

    def _parse_game_odds(self, raw: dict) -> Optional[MLBGameOdds]:
        """Parse raw Odds API game dict into MLBGameOdds."""
        home_name = raw.get("home_team", "")
        away_name = raw.get("away_team", "")
        home_abbrev = _FULL_TO_ABBREV.get(home_name, home_name[:3].upper())
        away_abbrev = _FULL_TO_ABBREV.get(away_name, away_name[:3].upper())
        
        logger.debug(f"[ODDS_PARSE] {away_name} @ {home_name} -> {away_abbrev} @ {home_abbrev}")

        game = MLBGameOdds(
            game_id=raw.get("id", ""),
            commence_time=raw.get("commence_time", ""),
            home_team=home_name,
            away_team=away_name,
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            park_factor=_PARK_FACTORS.get(home_abbrev, 1.0),
        )

        # Parse bookmaker odds — prefer DraftKings > FanDuel > first available
        bookmakers = raw.get("bookmakers", [])
        preferred_order = ["draftkings", "fanduel", "bovada"]
        bm_data = None
        for pref in preferred_order:
            bm_data = next((b for b in bookmakers if b.get("key") == pref), None)
            if bm_data:
                break
        if not bm_data and bookmakers:
            bm_data = bookmakers[0]

        if bm_data:
            for market in bm_data.get("markets", []):
                mtype = market.get("key")
                outcomes = market.get("outcomes", [])
                if mtype == "totals" and outcomes:
                    # Find "Over" — total is the point value
                    for o in outcomes:
                        if o.get("name") == "Over":
                            game.total = float(o.get("point", 0))
                            break
                elif mtype == "spreads":
                    for o in outcomes:
                        if o.get("name") == home_name:
                            game.spread_home = float(o.get("point", 0))
                            break
                elif mtype == "h2h":
                    for o in outcomes:
                        if o.get("name") == home_name:
                            game.moneyline_home = float(o.get("price", 0))
                        elif o.get("name") == away_name:
                            game.moneyline_away = float(o.get("price", 0))

        # Compute implied team runs
        if game.total is not None:
            game.implied_home_runs, game.implied_away_runs = self._implied_runs(
                game.total, game.spread_home or 0.0
            )

        return game

    @staticmethod
    def _implied_runs(total: float, spread_home: float) -> Tuple[float, float]:
        """
        Convert game total + spread to per-team implied runs.

        Spread reflects run differential, so:
          home_runs = (total - spread_home) / 2 + spread_home
                    = (total + spread_home) / 2
          away_runs = total - home_runs

        spread_home is negative when home team is favored (e.g., -1.5).
        """
        home_runs = (total + spread_home) / 2.0
        away_runs = total - home_runs
        # Clamp to realistic range
        home_runs = max(1.0, min(12.0, home_runs))
        away_runs = max(1.0, min(12.0, away_runs))
        return round(home_runs, 2), round(away_runs, 2)

    # ------------------------------------------------------------------
    # Batter ranking
    # ------------------------------------------------------------------

    def rank_batters(
        self,
        roster: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
    ) -> List[BatterRanking]:
        """
        Rank batters by daily lineup value.

        Args:
            roster: List of player dicts from YahooFantasyClient.get_roster()
            projections: List of player projection dicts from projections_loader
            game_date: YYYY-MM-DD (defaults to today)

        Returns:
            Sorted list of BatterRanking (best first).
        """
        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        proj_by_name = {p["name"].lower(): p for p in projections
                        if p.get("type") == "batter" or p.get("player_type") == "batter"}

        rankings = []
        for player in roster:
            positions = player.get("positions", [])
            # Skip pitchers - if ANY position is SP/RP/P, they're a pitcher
            # This handles two-way players (e.g., Shohei Ohtani with SP + Util)
            if any(p in ("SP", "RP", "P") for p in positions):
                continue
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue

            name = player.get("name", "")
            team_raw = player.get("team", "")
            team = normalize_team_abbr(team_raw)
            proj = proj_by_name.get(name.lower(), {})

            # Get team's implied runs from odds
            odds_data = team_odds.get(team, {})
            implied_runs = odds_data.get("implied_runs", 4.5)   # league avg fallback
            is_home = odds_data.get("is_home", False)
            park_factor = odds_data.get("park_factor", 1.0)
            has_game = team in team_odds  # True if team has a game today

            # Composite lineup score
            # Weights: implied_runs (environment) + projected stats
            base_score = implied_runs * park_factor
            # Use player's actual projected AVG (default to 0 if missing, not 0.250)
            # The 0.250 * 5.0 was adding 1.25 to every score, causing identical scores
            proj_avg = proj.get("avg", 0.0)
            stat_bonus = (
                proj.get("hr", 0) * 2.0
                + proj.get("r", 0) * 0.3
                + proj.get("rbi", 0) * 0.3
                + proj.get("nsb", 0) * 0.5
                + proj_avg * 5.0
            )
            lineup_score = base_score + stat_bonus * 0.1

            reason_parts = [f"team implied {implied_runs:.1f}R"]
            if park_factor > 1.05:
                reason_parts.append(f"hitter park ({park_factor:.2f}x)")
            if is_home:
                reason_parts.append("home")
            if status and status not in ("", "DTD"):
                reason_parts.append(f"status: {status}")

            rankings.append(BatterRanking(
                name=name,
                team=team,
                positions=positions,
                implied_team_runs=implied_runs,
                park_factor=park_factor,
                projected_r=proj.get("r", 0),
                projected_hr=proj.get("hr", 0),
                projected_rbi=proj.get("rbi", 0),
                projected_avg=proj.get("avg", 0.0),
                is_home=is_home,
                status=status,
                lineup_score=round(lineup_score, 3),
                reason=", ".join(reason_parts),
                has_game=has_game,
            ))

        rankings.sort(key=lambda x: x.lineup_score, reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # SP streaming
    # ------------------------------------------------------------------

    def rank_streamers(
        self,
        free_agents: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
        min_k9: float = 7.5,
        max_era: float = 4.50,
    ) -> List[PitcherRanking]:
        """
        Rank streaming SP candidates by daily matchup quality.

        Args:
            free_agents: Players from YahooFantasyClient.get_free_agents('SP')
            projections: Pitcher projections from projections_loader
            game_date: YYYY-MM-DD
            min_k9: Minimum K/9 for consideration
            max_era: Maximum projected ERA for consideration
        """
        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        proj_by_name = {p["name"].lower(): p for p in projections
                        if (p.get("type") or p.get("player_type", "")) == "pitcher"}

        rankings = []
        for player in free_agents:
            status = player.get("status")
            if status in ("IL", "IL60", "NA"):
                continue
            name = player.get("name", "")
            team_raw = player.get("team", "")
            team = normalize_team_abbr(team_raw)
            proj = proj_by_name.get(name.lower(), {})

            k9 = proj.get("k9", 0.0)
            era = proj.get("era", 5.0)
            if k9 < min_k9 or era > max_era:
                continue

            # Pitcher wants LOW opponent implied runs
            odds_data = team_odds.get(team, {})
            opp_team = odds_data.get("opponent", "")
            opp_odds = team_odds.get(opp_team, {})
            implied_opp_runs = opp_odds.get("implied_runs", 4.5)
            is_home = odds_data.get("is_home", False)
            park_factor = odds_data.get("park_factor", 1.0)

            # Stream score: lower opponent runs = better; higher K/9 = better
            # Normalize: 3.5 opp runs = best, 5.5 = worst
            env_score = max(0.0, (5.5 - implied_opp_runs) / 2.0) * 10  # 0-10
            k_score = min(10.0, k9 - 5.0)  # 0-10 for 5-15 K/9
            park_score = (2.0 - park_factor) * 5  # pitcher parks get bonus
            stream_score = env_score * 0.5 + k_score * 0.3 + park_score * 0.2

            reason_parts = [f"opp {implied_opp_runs:.1f}R", f"K/9 {k9:.1f}"]
            if is_home:
                reason_parts.append("home")
            if park_factor < 0.97:
                reason_parts.append(f"pitcher park ({park_factor:.2f}x)")

            rankings.append(PitcherRanking(
                name=name,
                team=team,
                opponent=opp_team,
                implied_opp_runs=implied_opp_runs,
                park_factor=park_factor,
                projected_k=proj.get("k", 0.0),
                projected_era=era,
                projected_ip=proj.get("ip", 0.0),
                is_home=is_home,
                status=status,
                stream_score=round(stream_score, 3),
                reason=", ".join(reason_parts),
            ))

        rankings.sort(key=lambda x: x.stream_score, reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Constraint-aware lineup solver
    # ------------------------------------------------------------------

    def solve_lineup(
        self,
        roster: List[dict],
        projections: List[dict],
        game_date: Optional[str] = None,
        slot_config: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> Tuple[List[LineupSlotResult], List[str]]:
        """
        Fill Yahoo lineup slots using greedy scarcity-first constraint solving.

        Slots are filled in order of scarcity (C → SS → 2B → 3B → 1B → OF×3 → Util)
        so that multi-eligible flex players (e.g. Castro 2B/3B) naturally cover
        whichever scarce position is left uncovered, rather than being wasted on OF.

        Off-day detection: when the Odds API returns data for 10+ teams (≥5 games),
        players whose team has no game are deprioritised — they fill slots only if
        no in-game player is available.

        Returns:
            (slot_results, warnings)
            slot_results — one LineupSlotResult per slot + BN entries for bench
            warnings     — human-readable alerts (empty slot, off-day start, etc.)
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        # Only apply off-day filtering when we have a credible slate (≥5 games worth
        # of teams).  Sparse/missing odds data must not bench healthy players.
        apply_offday_filter = len(team_odds) >= 10

        slots = slot_config if slot_config is not None else _DEFAULT_BATTER_SLOTS

        # Ranked by lineup_score descending; IL/pitcher rows already excluded by rank_batters
        ranked: List[BatterRanking] = self.rank_batters(roster, projections, game_date)

        # Belt-and-suspenders: also exclude any IL10/OUT rows rank_batters may have kept
        ranked = [b for b in ranked if b.status not in _INACTIVE_STATUSES]

        def _has_game(team: str) -> bool:
            return team in team_odds

        assigned: set = set()
        slot_results: List[LineupSlotResult] = []
        warnings: List[str] = []

        for slot_label, eligible_positions in slots:
            best: Optional[BatterRanking] = None

            # Pass 1: find best eligible player WITH a game today
            for b in ranked:
                if b.name in assigned:
                    continue
                if not any(pos in b.positions for pos in eligible_positions):
                    continue
                if apply_offday_filter and not _has_game(b.team):
                    continue
                best = b
                break

            # Pass 2: no in-game player found — fall back to any eligible player
            if best is None:
                for b in ranked:
                    if b.name in assigned:
                        continue
                    if not any(pos in b.positions for pos in eligible_positions):
                        continue
                    best = b
                    if apply_offday_filter:
                        warnings.append(
                            f"{slot_label}: {b.name} ({b.team}) has no game today — verify schedule"
                        )
                    break

            if best is None:
                warnings.append(f"No eligible active player found for {slot_label} slot")
                slot_results.append(LineupSlotResult(
                    slot=slot_label, player_name="EMPTY", player_team="",
                    positions=[], lineup_score=0.0, implied_runs=0.0,
                    park_factor=1.0, has_game=False, status=None,
                    reason=f"No eligible player for {slot_label}",
                ))
            else:
                assigned.add(best.name)
                slot_results.append(LineupSlotResult(
                    slot=slot_label,
                    player_name=best.name,
                    player_team=best.team,
                    positions=best.positions,
                    lineup_score=best.lineup_score,
                    implied_runs=best.implied_team_runs,
                    park_factor=best.park_factor,
                    has_game=_has_game(best.team),
                    status=best.status,
                    reason=best.reason,
                ))

        # All remaining eligible players → bench
        for b in ranked:
            if b.name not in assigned:
                slot_results.append(LineupSlotResult(
                    slot="BN",
                    player_name=b.name,
                    player_team=b.team,
                    positions=b.positions,
                    lineup_score=b.lineup_score,
                    implied_runs=b.implied_team_runs,
                    park_factor=b.park_factor,
                    has_game=_has_game(b.team),
                    status=b.status,
                    reason=b.reason,
                ))

        return slot_results, warnings

    # ------------------------------------------------------------------
    # SP off-day detection
    # ------------------------------------------------------------------

    def flag_pitcher_starts(
        self,
        roster: List[dict],
        game_date: Optional[str] = None,
    ) -> List[dict]:
        """
        Return each pitcher from roster annotated with has_start: bool.

        SP with has_start=False should sit (no start today).
        RP always has_start=True (they can pitch any day).
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        team_odds = self._build_team_odds_map(self.fetch_mlb_odds(game_date))
        has_slate = len(team_odds) >= 10
        
        # Fetch probable pitchers for accurate start detection
        probable_pitchers = self._fetch_probable_pitchers_for_date(game_date)

        result = []
        for p in roster:
            positions = p.get("positions", [])
            status = p.get("status")
            player_name = p.get("name", "")
            
            logger.debug(f"[PITCHER_DEBUG] {player_name}: positions={positions}, status={status}")
            
            if not any(pos in ("SP", "RP", "P") for pos in positions):
                continue
            if status in _INACTIVE_STATUSES:
                continue
            
            is_sp = "SP" in positions
            team_raw = p.get("team", "")
            team = normalize_team_abbr(team_raw)
            
            logger.debug(f"[PITCHER_DEBUG] {player_name}: is_sp={is_sp}, team={team}")
            
            # Check if this specific pitcher is the probable starter
            if is_sp:
                # Get expected opponent if team has a game
                has_game = team in team_odds if has_slate else True
                opponent = team_odds.get(team, {}).get("opponent", "") if has_slate else ""
                
                # Check if this player matches a probable starter
                is_probable = self._is_probable_starter(player_name, team, opponent, probable_pitchers)
                
                # FALLBACK: If no probable pitchers returned (spring training/offseason),
                # assume all SPs on roster with a game are potential starters
                if not probable_pitchers and has_game:
                    is_probable = True
                    logger.debug(f"No probable pitchers available, assuming {player_name} ({team}) is a starter")
                
                has_start = has_game and is_probable
            else:
                has_start = True  # RP can pitch any day
                
            result.append({
                **p,
                "has_start": has_start,
                "pitcher_slot": "SP" if is_sp else "RP",
            })
        return result
    
    def _fetch_probable_pitchers_for_date(self, game_date: str) -> dict:
        """
        Fetch probable pitchers for a date.

        Resolution order:
          1. persisted `probable_pitchers` snapshot table
          2. MLB Stats API live lookup
          3. conservative 5-day rotation inference from recent pitcher stats

        Returns dict mapping team abbrev to pitcher name.
        """
        try:
            from datetime import date as date_type

            parsed_date = date_type.fromisoformat(game_date)
        except ValueError:
            parsed_date = None

        if parsed_date is not None:
            db = SessionLocal()
            try:
                persisted = load_probable_pitchers_from_snapshot(db, parsed_date)
                if persisted:
                    return persisted
            except Exception as exc:
                logger.warning(f"Failed to load probable pitchers from snapshot: {exc}")
            finally:
                db.close()

        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
        }
        
        probable = {}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for date_info in data.get("dates", []):
                for game in date_info.get("games", []):
                    teams = game.get("teams", {})
                    
                    # Home pitcher - normalize team abbreviation to Yahoo standard
                    home_team_raw = teams.get("home", {}).get("team", {}).get("abbreviation", "")
                    home_team = normalize_team_abbr(home_team_raw)
                    home_pitcher = teams.get("home", {}).get("probablePitcher", {})
                    if home_pitcher and home_team:
                        probable[home_team] = home_pitcher.get("fullName", "").lower()
                    
                    # Away pitcher - normalize team abbreviation to Yahoo standard
                    away_team_raw = teams.get("away", {}).get("team", {}).get("abbreviation", "")
                    away_team = normalize_team_abbr(away_team_raw)
                    away_pitcher = teams.get("away", {}).get("probablePitcher", {})
                    if away_pitcher and away_team:
                        probable[away_team] = away_pitcher.get("fullName", "").lower()
                        
        except Exception as e:
            logger.warning(f"Failed to fetch probable pitchers: {e}")

        if probable or parsed_date is None:
            return probable

        db = SessionLocal()
        try:
            inferred = infer_probable_pitcher_map(db, parsed_date)
            return {team: candidate.pitcher_name.lower() for team, candidate in inferred.items()}
        except Exception as exc:
            logger.warning(f"Failed to infer probable pitchers from recent stats: {exc}")
            return probable
        finally:
            db.close()

    
    def _is_probable_starter(self, player_name: str, team: str, opponent: str, probable: dict) -> bool:
        """
        Check if a player is the probable starter.
        Uses fuzzy matching on names.
        """
        if not player_name:
            return False
            
        # Direct match
        player_lower = player_name.lower()
        if team in probable:
            if probable[team] == player_lower:
                return True
            # Partial match (e.g., "Shota Imanaga" matches "Shota Imanaga")
            if player_lower in probable[team] or probable[team] in player_lower:
                return True
                
        return False

    # ------------------------------------------------------------------
    # Full daily report
    # ------------------------------------------------------------------

    def build_daily_report(
        self,
        game_date: Optional[str] = None,
        roster: Optional[List[dict]] = None,
        projections: Optional[List[dict]] = None,
    ) -> dict:
        """
        Build a full daily report: game environment + batter/pitcher rankings.

        Returns dict with:
            - game_date
            - games: list of game odds with implied runs
            - batter_rankings: sorted list of BatterRanking dicts
            - pitcher_rankings: sorted list of PitcherRanking dicts
            - best_stacks: teams with highest implied runs (stack candidates)
            - avoid_pitchers: opponents of high-implied-run teams
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        games = self.fetch_mlb_odds(game_date)
        team_odds = self._build_team_odds_map(games)

        # Identify best stacks (teams with >5.0 implied runs)
        stack_candidates = sorted(
            [(team, data["implied_runs"]) for team, data in team_odds.items()
             if data.get("implied_runs", 0) >= 5.0],
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify environments to avoid for pitchers
        high_offense_teams = [t for t, _ in stack_candidates[:4]]

        batter_rankings = []
        pitcher_rankings = []
        if roster and projections:
            batter_rankings = [
                {
                    "name": b.name, "team": b.team, "positions": b.positions,
                    "implied_runs": b.implied_team_runs, "park_factor": b.park_factor,
                    "score": b.lineup_score, "reason": b.reason, "status": b.status,
                }
                for b in self.rank_batters(roster, projections, game_date)
            ]

        return {
            "game_date": game_date,
            "games": [
                {
                    "home": g.home_abbrev, "away": g.away_abbrev,
                    "total": g.total, "spread_home": g.spread_home,
                    "implied_home": g.implied_home_runs,
                    "implied_away": g.implied_away_runs,
                    "park_factor": g.park_factor,
                }
                for g in games
            ],
            "stack_candidates": [
                {"team": t, "implied_runs": round(r, 2)} for t, r in stack_candidates
            ],
            "avoid_pitcher_matchups": high_offense_teams,
            "batter_rankings": batter_rankings,
            "pitcher_rankings": pitcher_rankings,
            "games_found": len(games),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_team_odds_map(self, games: List[MLBGameOdds]) -> Dict[str, dict]:
        """
        Build a dict: team_abbrev -> {implied_runs, is_home, opponent, park_factor}
        """
        result: Dict[str, dict] = {}
        logger.debug(f"[BUILD_MAP] Building team odds map from {len(games)} games")
        for g in games:
            # Always add teams to the map, even without implied runs
            # This ensures has_game detection works
            # Normalize team abbreviations to Yahoo standard
            home_norm = normalize_team_abbr(g.home_abbrev)
            away_norm = normalize_team_abbr(g.away_abbrev)
            
            if home_norm and away_norm:
                result[home_norm] = {
                    "implied_runs": g.implied_home_runs if g.implied_home_runs is not None else 4.5,
                    "is_home": True,
                    "opponent": away_norm,
                    "park_factor": g.park_factor,
                }
                result[away_norm] = {
                    "implied_runs": g.implied_away_runs if g.implied_away_runs is not None else 4.5,
                    "is_home": False,
                    "opponent": home_norm,
                    "park_factor": g.park_factor,
                }
                logger.debug(f"[BUILD_MAP] Added {home_norm} vs {away_norm} (implied: {g.implied_home_runs}, {g.implied_away_runs})")
        logger.info(f"[BUILD_MAP] Final team_odds keys: {list(result.keys())}")
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_optimizer: Optional[DailyLineupOptimizer] = None


def get_lineup_optimizer() -> DailyLineupOptimizer:
    """Get singleton optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = DailyLineupOptimizer()
    return _optimizer
