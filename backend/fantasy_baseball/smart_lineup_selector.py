"""
Smart Lineup Selector - Advanced lineup optimization with platoon splits,
opposing pitcher analysis, and category awareness.

Integrates with:
- daily_lineup_optimizer (odds, implied runs)
- projections_loader (Steamer projections)
- lineup_validator (game-aware validation)
- platoon_fetcher (FanGraphs splits)
- pybaseball_loader (Statcast data)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import requests

from backend.fantasy_baseball.lineup_validator import (
    LineupValidator, OptimizedSlot, PlayerGameInfo, GameStatus
)
from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer, BatterRanking, normalize_team_abbr
from backend.fantasy_baseball.platoon_fetcher import PlatoonSplitFetcher, PlatoonSplits
from backend.fantasy_baseball.pitcher_deep_dive import PitcherDeepDiveFetcher, get_pitcher_fetcher
from backend.fantasy_baseball.elite_context import PitcherDeepDive, WeatherContext, RecentForm
from backend.fantasy_baseball.weather_fetcher import WeatherFetcher, GameWeather, get_weather_fetcher
from backend.fantasy_baseball.park_weather import ParkWeatherAnalyzer, get_park_analyzer

logger = logging.getLogger(__name__)


class Handedness(Enum):
    L = "L"
    R = "R"
    S = "S"  # Switch


@dataclass
class OpposingPitcher:
    """Pitcher a batter will face."""
    name: str
    team: str
    handedness: Handedness
    era: float = 4.50
    k9: float = 8.0
    whip: float = 1.30
    pitch_mix: Dict[str, float] = field(default_factory=dict)  # pitch type %
    recent_form: float = 0.0  # Last 3 starts ERA
    
    # Deep dive stats
    fip: float = 4.50
    xfip: float = 4.50
    sierra: float = 4.50
    gb_pct: float = 45.0
    hard_hit_pct: float = 35.0
    era_vs_lhb: float = 4.50
    era_vs_rhb: float = 4.50
    
    @property
    def is_ace(self) -> bool:
        """Is this an elite pitcher?"""
        return self.era < 3.00 and self.k9 > 9.0 and self.fip < 3.20
    
    @property
    def is_streamable(self) -> bool:
        """Is this a weak pitcher worth targeting?"""
        return self.era > 5.00 or (self.era > 4.50 and self.k9 < 7.0)
    
    @property
    def quality_score(self) -> float:
        """Composite pitcher quality (0-10, 10 = ace)."""
        era_score = max(0, min(10, (6.0 - self.era) * 2.5))
        fip_score = max(0, min(10, (6.0 - self.fip) * 2.5))
        k_score = min(10, self.k9)
        whip_score = max(0, min(10, (2.0 - self.whip) * 10))
        return (era_score * 0.3 + fip_score * 0.3 + k_score * 0.2 + whip_score * 0.2)
    
    @classmethod
    def from_deep_dive(cls, deep_dive: PitcherDeepDive) -> "OpposingPitcher":
        """Create from PitcherDeepDive."""
        return cls(
            name=deep_dive.name,
            team=deep_dive.team,
            handedness=Handedness.L if deep_dive.handedness == "L" else Handedness.R,
            era=deep_dive.era,
            k9=deep_dive.k9,
            whip=deep_dive.whip,
            fip=deep_dive.fip,
            xfip=deep_dive.xfip,
            sierra=deep_dive.sierra,
            gb_pct=deep_dive.gb_pct,
            hard_hit_pct=deep_dive.hard_hit_pct,
            era_vs_lhb=deep_dive.era_vs_lhb,
            era_vs_rhb=deep_dive.era_vs_rhb,
        )


@dataclass
class CategoryNeed:
    """Team's need in a specific category."""
    category: str  # "hr", "r", "rbi", "sb", "avg", etc.
    current: float
    projected_opponent: float
    needed: float  # Positive = need more, Negative = ahead
    
    @property
    def urgency(self) -> float:
        """How urgent is this need? (0-1 scale)"""
        if abs(self.projected_opponent) < 0.01:
            return 0.0
        gap_ratio = abs(self.needed) / max(abs(self.projected_opponent), 1.0)
        return min(1.0, gap_ratio)


@dataclass
class SmartBatterRanking:
    """Enhanced batter ranking with all factors."""
    name: str
    player_id: str
    team: str
    positions: List[str]
    
    # Base projections (from Steamer)
    proj_hr: float = 0.0
    proj_r: float = 0.0
    proj_rbi: float = 0.0
    proj_sb: float = 0.0
    proj_avg: float = 0.250
    proj_ops: float = 0.750
    
    # Game context (from odds/schedule)
    has_game: bool = False
    is_home: bool = False
    implied_team_runs: float = 4.5
    park_factor: float = 1.0
    game_time: Optional[datetime] = None
    weather: Optional[WeatherContext] = None
    hr_factor: float = 1.0  # Weather-adjusted HR factor
    
    # Platoon/matchup (from Statcast/pybaseball)
    platoon: Optional[PlatoonSplits] = None
    opposing_pitcher: Optional[OpposingPitcher] = None
    
    # Category fit
    category_fit: float = 0.0  # How well player fills team needs
    
    # Final score
    smart_score: float = 0.0
    
    def calculate_score(self, category_needs: List[CategoryNeed] = None):
        """Calculate composite smart score."""
        # Base projection score (standardized)
        base_score = (
            self.proj_hr * 2.0 +
            self.proj_r * 0.3 +
            self.proj_rbi * 0.3 +
            self.proj_sb * 0.5 +
            (self.proj_avg - 0.250) * 50 +
            (self.proj_ops - 0.750) * 30
        )
        
        # Game environment boost
        env_boost = (self.implied_team_runs - 4.5) * 2  # Higher runs = better
        park_boost = (self.park_factor - 1.0) * 5  # Hitter parks
        home_boost = 0.5 if self.is_home else 0.0
        
        # Weather boost (NEW)
        weather_boost = 0.0
        if self.weather:
            weather_boost = (self.weather.hitter_friendly_score - 5.0) * 0.5
        
        # HR factor boost for power hitters
        hr_weather_boost = 0.0
        if self.hr_factor != 1.0 and self.proj_hr > 0.2:  # Power hitter
            hr_weather_boost = (self.hr_factor - 1.0) * self.proj_hr * 10
        
        # Combine environment factors
        total_env_boost = env_boost + park_boost + home_boost + weather_boost + hr_weather_boost
        
        # Platoon advantage
        platoon_boost = 0.0
        if self.platoon and self.opposing_pitcher:
            opp_hand = self.opposing_pitcher.handedness
            if opp_hand == Handedness.L:
                platoon_boost = self.platoon.vs_lhp * 10  # wOBA scaled
            elif opp_hand == Handedness.R:
                platoon_boost = self.platoon.vs_rhp * 10
        
        # Opposing pitcher difficulty
        pitcher_penalty = 0.0
        if self.opposing_pitcher:
            # Facing an ace = penalty (negative), facing weak SP = bonus (positive)
            # quality_score: 0-10 (10 = ace), so ace gives negative, weak gives positive
            pitcher_penalty = (5.0 - self.opposing_pitcher.quality_score) * 0.5
        
        # Category need fit
        cat_boost = 0.0
        if category_needs:
            for need in category_needs:
                if need.needed > 0:  # Need more of this category
                    cat_contribution = self._category_contribution(need.category)
                    cat_boost += cat_contribution * need.urgency * 2
        
        self.smart_score = (
            base_score * 0.35 +
            total_env_boost * 0.20 +
            platoon_boost * 0.15 +
            pitcher_penalty * 0.10 +
            cat_boost * 0.20
        )
        
        return self.smart_score
    
    def _category_contribution(self, category: str) -> float:
        """How much does this player contribute to a category?"""
        contributions = {
            "hr": self.proj_hr,
            "r": self.proj_r,
            "rbi": self.proj_rbi,
            "sb": self.proj_sb,
            "avg": self.proj_avg * 100,  # Scale up
            "ops": self.proj_ops * 100,
        }
        return contributions.get(category, 0.0)


class SmartLineupSelector:
    """
    Intelligent lineup selection using:
    - Platoon splits (from FanGraphs)
    - Opposing pitcher quality
    - Sportsbook odds/implied totals
    - Category needs from matchup scoreboard
    - Park factors
    """
    
    def __init__(self):
        self.base_optimizer = DailyLineupOptimizer()
        self.lineup_validator = LineupValidator()
        self.platoon_fetcher = PlatoonSplitFetcher()
        self.pitcher_fetcher = PitcherDeepDiveFetcher()
        self.weather_fetcher = get_weather_fetcher()
        self.park_analyzer = get_park_analyzer()
        self._pitcher_cache: Dict[str, OpposingPitcher] = {}
        self._weather_cache: Dict[str, GameWeather] = {}
    
    def select_optimal_lineup(
        self,
        roster: List[Dict],
        projections: List[Dict],
        game_date: Optional[str] = None,
        category_needs: Optional[List[CategoryNeed]] = None,
        opponent_categories: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[SmartBatterRanking], List[str]]:
        """
        Select optimal lineup using all available data.
        
        Returns:
            (ranked_players, warnings)
            ranked_players: Sorted list of SmartBatterRanking (best first)
        """
        if game_date is None:
            game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        
        # Get base rankings from odds optimizer
        base_rankings = self.base_optimizer.rank_batters(roster, projections, game_date)
        games = self.base_optimizer.fetch_mlb_odds(game_date)
        
        # Build team->game lookup
        team_game_map = self._build_team_game_map(games)
        
        # Fetch probable pitchers for the day
        probable_pitchers = self._fetch_probable_pitchers(game_date)
        
        # Fetch weather for all games
        game_list = []
        for team, info in team_game_map.items():
            game_list.append({
                "venue": info.get("venue", ""),
                "game_time": datetime.strptime(f"{game_date} 19:05", "%Y-%m-%d %H:%M"),  # Default 7:05pm
                "home_team": team if info.get("is_home") else info.get("opponent"),
            })
        
        weather_map = self.weather_fetcher.get_weather_for_games(game_list)
        
        # Enhance each ranking with smart data
        smart_rankings = []
        warnings = []
        
        for base in base_rankings:
            # Get game context
            game_info = team_game_map.get(base.team, {})
            opponent = game_info.get("opponent")
            venue = game_info.get("venue", "")
            
            # Get opposing pitcher
            opp_pitcher = None
            if opponent and opponent in probable_pitchers:
                opp_pitcher = probable_pitchers[opponent]
                logger.debug(f"{base.name} ({base.team}) vs {opponent}: found pitcher {opp_pitcher.name}")
            elif opponent:
                logger.debug(f"{base.name} ({base.team}) vs {opponent}: NO pitcher data found")
                # Try to find by iterating through all pitchers
                for team, pitcher in probable_pitchers.items():
                    logger.debug(f"  Available: {team} -> {pitcher.name}")
            
            # Get platoon splits
            platoon = self._get_platoon_splits(base.name)
            
            # Get weather with park-specific analysis
            weather = None
            hr_factor = 1.0
            game_risk = "low"
            park_weather_summary = ""
            
            if venue and venue in weather_map:
                game_weather = weather_map[venue]
                
                # Use park analyzer for precise HR factor
                analysis = self.park_analyzer.analyze_game(venue, game_weather)
                hr_factor = analysis["total_hr_factor"]
                park_weather_summary = analysis["description"]
                
                weather = game_weather.to_context()
                weather.hitter_friendly_score = analysis["temp_factor"] * 10  # Adjust based on full analysis
                
                game_risk = game_weather.game_risk
                logger.debug(f"{base.name} ({base.team}): {park_weather_summary}")
            
            # Warn about high-risk games
            if game_risk in ["high", "postponement_risk"]:
                warnings.append(f"⚠️ {base.name} ({base.team}): Game at risk - {game_weather.summary if venue in weather_map else 'weather concern'}")
            
            # Build smart ranking
            smart = SmartBatterRanking(
                name=base.name,
                player_id=base.name.lower().replace(" ", "_"),  # Simple ID
                team=base.team,
                positions=base.positions,
                proj_hr=base.projected_hr,
                proj_r=base.projected_r,
                proj_rbi=base.projected_rbi,
                proj_sb=0.0,  # Not in base ranking, fetch from projections
                proj_avg=base.projected_avg,
                has_game=base.has_game,
                is_home=base.is_home,
                implied_team_runs=base.implied_team_runs,
                park_factor=base.park_factor,
                weather=weather,
                hr_factor=hr_factor,
                platoon=platoon,
                opposing_pitcher=opp_pitcher,
            )
            
            # Fill in SB from projections if available
            for proj in projections:
                if proj.get("name") == base.name:
                    smart.proj_sb = proj.get("nsb", 0.0)
                    break
            
            # Calculate smart score
            smart.calculate_score(category_needs)
            smart_rankings.append(smart)
        
        # Sort by smart score
        smart_rankings.sort(key=lambda x: x.smart_score, reverse=True)
        
        # Generate warnings for potential issues
        for rank in smart_rankings[:9]:  # Check starters
            if not rank.has_game:
                warnings.append(f"{rank.name}: Starting but no game today")
            if rank.opposing_pitcher and rank.opposing_pitcher.quality_score > 8:
                warnings.append(f"{rank.name}: Facing ace {rank.opposing_pitcher.name} ({rank.opposing_pitcher.era:.2f} ERA)")
            if rank.platoon and abs(rank.platoon.split_delta) > 0.050:
                hand = "LHP" if rank.platoon.split_delta > 0 else "RHP"
                warnings.append(f"{rank.name}: Strong platoon split vs {hand}")
        
        return smart_rankings, warnings
    
    def _build_team_game_map(self, games: List) -> Dict[str, Dict]:
        """Build lookup of team -> game info."""
        from backend.fantasy_baseball.daily_lineup_optimizer import normalize_team_abbr
        result = {}
        for game in games:
            home = normalize_team_abbr(getattr(game, 'home_abbrev', None))
            away = normalize_team_abbr(getattr(game, 'away_abbrev', None))
            
            if home:
                result[home] = {
                    "opponent": away,
                    "is_home": True,
                    "implied_runs": getattr(game, 'implied_home_runs', 4.5),
                    "park_factor": getattr(game, 'park_factor', 1.0),
                }
            if away:
                result[away] = {
                    "opponent": home,
                    "is_home": False,
                    "implied_runs": getattr(game, 'implied_away_runs', 4.5),
                    "park_factor": getattr(game, 'park_factor', 1.0),
                }
        return result
    
    def _fetch_probable_pitchers(self, game_date: str) -> Dict[str, OpposingPitcher]:
        """
        Fetch probable pitchers for the day with deep dive stats.
        Returns dict: team -> OpposingPitcher
        """
        # Use MLB Stats API for probable pitchers
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
        }
        
        result = {}
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
                        name = home_pitcher.get("fullName", "Unknown")
                        # Try to get deep dive stats
                        deep_dive = self.pitcher_fetcher.get_pitcher(name, home_team)
                        if deep_dive:
                            pitcher = OpposingPitcher.from_deep_dive(deep_dive)
                            logger.info(f"Pitcher deep dive for {home_team}: {pitcher.name} "
                                      f"(ERA: {pitcher.era:.2f}, FIP: {pitcher.fip:.2f}, "
                                      f"K/9: {pitcher.k9:.1f})")
                        else:
                            # Fallback to basic parsing
                            pitcher = self._parse_pitcher(home_pitcher)
                            pitcher.team = home_team
                            logger.debug(f"Basic pitcher data for {home_team}: {pitcher.name}")
                        result[home_team] = pitcher
                    
                    # Away pitcher - normalize team abbreviation to Yahoo standard
                    away_team_raw = teams.get("away", {}).get("team", {}).get("abbreviation", "")
                    away_team = normalize_team_abbr(away_team_raw)
                    away_pitcher = teams.get("away", {}).get("probablePitcher", {})
                    if away_pitcher and away_team:
                        name = away_pitcher.get("fullName", "Unknown")
                        deep_dive = self.pitcher_fetcher.get_pitcher(name, away_team)
                        if deep_dive:
                            pitcher = OpposingPitcher.from_deep_dive(deep_dive)
                            logger.info(f"Pitcher deep dive for {away_team}: {pitcher.name} "
                                      f"(ERA: {pitcher.era:.2f}, FIP: {pitcher.fip:.2f}, "
                                      f"K/9: {pitcher.k9:.1f})")
                        else:
                            pitcher = self._parse_pitcher(away_pitcher)
                            pitcher.team = away_team
                            logger.debug(f"Basic pitcher data for {away_team}: {pitcher.name}")
                        result[away_team] = pitcher
            
            logger.info(f"Fetched {len(result)} probable pitchers for {game_date}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch probable pitchers: {e}")
        
        return result
    
    def _parse_pitcher(self, data: Dict) -> OpposingPitcher:
        """Parse MLB API pitcher data."""
        name = data.get("fullName", "Unknown")
        
        # Get handedness from stats if available
        hand_str = data.get("pitchHand", {}).get("code", "R")
        handedness = Handedness.S if hand_str == "S" else Handedness.L if hand_str == "L" else Handedness.R
        
        return OpposingPitcher(
            name=name,
            team="",  # Will be set by caller
            handedness=handedness,
            era=4.50,  # Default, enhance with stats lookup
            k9=8.0,
        )
    
    def _get_platoon_splits(self, player_name: str, year: Optional[int] = None) -> Optional[PlatoonSplits]:
        """Get platoon splits for a player (using cached fetcher)."""
        return self.platoon_fetcher.get_splits(player_name, year)
    
    def solve_smart_lineup(
        self,
        roster: List[Dict],
        projections: List[Dict],
        game_date: Optional[str] = None,
        slot_config: Optional[List[Tuple[str, List[str]]]] = None,
        category_needs: Optional[List[CategoryNeed]] = None,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Solve lineup with smart rankings.
        
        Returns:
            (slot_assignments, warnings)
            slot_assignments: List of dicts with slot, player_id, player_name, reason
        """
        rankings, warnings = self.select_optimal_lineup(
            roster, projections, game_date, category_needs
        )
        
        # Default slot config if not provided
        if slot_config is None:
            from backend.fantasy_baseball.daily_lineup_optimizer import _DEFAULT_BATTER_SLOTS
            slot_config = _DEFAULT_BATTER_SLOTS
        
        # Filter to players with games
        rankings_with_games = [r for r in rankings if r.has_game]
        rankings_without_games = [r for r in rankings if not r.has_game]
        
        assigned = set()
        assignments = []
        
        # Fill slots in order
        for slot_label, eligible_positions in slot_config:
            # First try players with games
            selected = None
            for rank in rankings_with_games:
                if rank.name in assigned:
                    continue
                if any(pos in rank.positions for pos in eligible_positions):
                    selected = rank
                    break
            
            # Fall back to players without games if necessary
            if selected is None and rankings_without_games:
                for rank in rankings_without_games:
                    if rank.name in assigned:
                        continue
                    if any(pos in rank.positions for pos in eligible_positions):
                        selected = rank
                        warnings.append(f"{slot_label}: {rank.name} has no game today")
                        break
            
            if selected:
                assigned.add(selected.name)
                reason_parts = [f"score: {selected.smart_score:.1f}"]
                if selected.opposing_pitcher:
                    reason_parts.append(f"vs {selected.opposing_pitcher.name}")
                if selected.platoon:
                    reason_parts.append(f"platoon: {selected.platoon.split_delta:+.3f}")
                
                assignments.append({
                    "slot": slot_label,
                    "player_id": selected.player_id,
                    "player_name": selected.name,
                    "team": selected.team,
                    "smart_score": selected.smart_score,
                    "has_game": selected.has_game,
                    "park_factor": selected.park_factor,
                    "implied_runs": selected.implied_team_runs,
                    "reason": ", ".join(reason_parts),
                })
            else:
                assignments.append({
                    "slot": slot_label,
                    "player_id": None,
                    "player_name": "EMPTY",
                    "reason": "No eligible player",
                })
        
        # Add bench
        for rank in rankings:
            if rank.name not in assigned:
                assignments.append({
                    "slot": "BN",
                    "player_id": rank.player_id,
                    "player_name": rank.name,
                    "team": rank.team,
                    "smart_score": rank.smart_score,
                    "has_game": rank.has_game,
                    "park_factor": rank.park_factor,
                    "implied_runs": rank.implied_team_runs,
                    "reason": "Bench",
                })
        
        return assignments, warnings


def get_smart_selector() -> SmartLineupSelector:
    """Factory function for SmartLineupSelector."""
    return SmartLineupSelector()
