"""
Elite fantasy baseball manager's decision context.

What separates elite managers from casual players:
1. CONTEXT AWARENESS - Understanding the full game environment
2. SITUATIONAL STRATEGY - Adjusting for matchup dynamics  
3. RISK MANAGEMENT - Balancing floor vs ceiling
4. PROCESS OVER RESULTS - Making +EV decisions consistently
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Risk tolerance for lineup decisions."""
    FLOOR = "floor"      # Safe production (high contact, lineup security)
    BALANCED = "balanced"  # Default approach
    CEILING = "ceiling"   # Swing for upside (power, SB threat)


class MatchupStrategy(Enum):
    """Strategy based on H2H matchup state."""
    AGGRESSIVE = "aggressive"    # Need to make up ground - play ceiling
    STANDARD = "standard"        # Play best available
    PROTECTIVE = "protective"    # Protect leads - play floor
    PUNT = "punt"               # Intentionally sacrifice a category


@dataclass
class WeatherContext:
    """Weather conditions affecting the game."""
    temperature: int = 72  # Fahrenheit
    wind_speed: int = 0    # MPH
    wind_direction: str = ""  # "out", "in", "left", "right", "unknown"
    precipitation: str = "none"  # "none", "drizzle", "rain"
    roof_closed: bool = False  # For domed stadiums
    
    @property
    def hitter_friendly_score(self) -> float:
        """Score 0-10 on how hitter-friendly conditions are."""
        score = 5.0
        
        # Temperature (warmer = ball travels further)
        if self.temperature > 80:
            score += 1.0
        elif self.temperature < 60:
            score -= 1.0
        
        # Wind (out = big boost, in = big penalty)
        if "out" in self.wind_direction.lower():
            score += min(3.0, self.wind_speed / 5)
        elif "in" in self.wind_direction.lower():
            score -= min(3.0, self.wind_speed / 5)
        
        # Precipitation
        if self.precipitation != "none":
            score -= 1.5
        
        return max(0, min(10, score))


@dataclass
class RecentForm:
    """Player's recent performance (hot/cold streaks)."""
    last_7_games: Dict[str, float] = field(default_factory=dict)
    last_14_games: Dict[str, float] = field(default_factory=dict)
    last_30_games: Dict[str, float] = field(default_factory=dict)
    
    # Trend direction
    trend_7d: float = 0.0   # Positive = improving
    trend_14d: float = 0.0
    
    # Expected stats (xBA, xSLG, xwOBA) to identify luck
    xwoba_last_14: float = 0.0
    woba_last_14: float = 0.0
    
    @property
    def is_hot(self) -> bool:
        """Is player on a hot streak?"""
        return self.trend_7d > 0.5
    
    @property
    def is_cold(self) -> bool:
        """Is player in a slump?"""
        return self.trend_7d < -0.5
    
    @property
    def is_unlucky(self) -> bool:
        """Underperforming expected stats (positive regression candidate)."""
        if self.xwoba_last_14 > 0 and self.woba_last_14 > 0:
            return self.xwoba_last_14 - self.woba_last_14 > 0.03
        return False


@dataclass
class LineupSpot:
    """Where player hits in the batting order."""
    spot: int = 9  # 1-9
    role: str = "unknown"  # "leadoff", "cleanup", "table_setter", etc.
    platoon_status: str = "full_time"  # "full_time", "platoon_v_l", "platoon_v_r"
    
    @property
    def plate_appearance_multiplier(self) -> float:
        """Expected PA relative to lineup spot (leadoff gets ~4.5, #9 gets ~3.8)."""
        # Approximate based on baseball reference data
        pas = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2, 6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8}
        return pas.get(self.spot, 4.0) / 4.0  # Normalize to 1.0


@dataclass
class PitcherDeepDive:
    """Comprehensive pitcher analysis beyond basic ERA."""
    name: str
    team: str
    handedness: str = "R"
    
    # Traditional stats
    era: float = 4.50
    whip: float = 1.30
    k9: float = 8.0
    bb9: float = 3.0
    
    # Advanced stats
    fip: float = 4.50      # Fielding Independent Pitching
    xfip: float = 4.50     # Expected FIP
    sierra: float = 4.50   # SIERA
    
    # Batted ball profile
    gb_pct: float = 45.0   # Ground ball %
    fb_pct: float = 35.0   # Fly ball %
    hr_fb: float = 12.0    # HR/FB ratio
    hard_hit_pct: float = 35.0
    
    # Recent form
    last_3_starts_era: float = 4.50
    last_3_starts_k9: float = 8.0
    days_rest: int = 4     # Days since last start
    
    # Splits
    era_vs_lhb: float = 4.50
    era_vs_rhb: float = 4.50
    k9_vs_lhb: float = 8.0
    k9_vs_rhb: float = 8.0
    
    @property
    def is_ace(self) -> bool:
        """Is this an elite pitcher?"""
        return self.era < 3.00 and self.k9 > 9.0 and self.fip < 3.20
    
    @property
    def is_streamable(self) -> bool:
        """Is this a weak pitcher worth targeting?"""
        return self.era > 5.00 or (self.era > 4.50 and self.k9 < 7.0)
    
    @property
    def is_tired(self) -> bool:
        """Short rest or heavy workload recently."""
        return self.days_rest < 4
    
    @property
    def quality_score(self) -> float:
        """Composite 0-10 score."""
        era_score = max(0, min(10, (6.0 - self.era) * 2.5))
        fip_score = max(0, min(10, (6.0 - self.fip) * 2.5))
        k_score = min(10, self.k9)
        recent_score = max(0, min(10, (6.0 - self.last_3_starts_era) * 2.5))
        
        # Weight recent form heavily
        return (era_score * 0.3 + fip_score * 0.3 + k_score * 0.2 + recent_score * 0.2)
    
    def get_matchup_quality(self, batter_handedness: str) -> float:
        """How tough is this pitcher vs specific handedness?"""
        if batter_handedness == "L":
            return (6.0 - self.era_vs_lhb) / 6.0 * 10
        else:
            return (6.0 - self.era_vs_rhb) / 6.0 * 10


@dataclass
class PlayerDecisionContext:
    """Complete context for a single player decision."""
    player_name: str
    team: str
    opponent: str
    
    # Player profile
    season_projection: Dict[str, float] = field(default_factory=dict)
    recent_form: RecentForm = field(default_factory=RecentForm)
    platoon_splits: Dict[str, float] = field(default_factory=dict)  # vs_lhp, vs_rhp
    lineup_spot: LineupSpot = field(default_factory=LineupSpot)
    
    # Game context
    opposing_pitcher: Optional[PitcherDeepDive] = None
    park_factor: float = 1.0
    weather: WeatherContext = field(default_factory=WeatherContext)
    implied_team_runs: float = 4.5
    is_home: bool = False
    
    # Category context
    category_needs: Dict[str, float] = field(default_factory=dict)  # category -> needed
    
    # Decision metadata
    confidence_score: float = 0.0  # 0-100
    recommendation: str = ""  # "start", "bench", "monitor"
    reasoning: List[str] = field(default_factory=list)
    alternative: Optional[str] = None  # Player who would replace them
    
    def calculate_confidence(self) -> float:
        """Calculate confidence in recommendation based on data quality."""
        score = 50.0  # Base confidence
        
        # Data quality boosts
        if self.opposing_pitcher and not self.opposing_pitcher.is_ace:
            score += 10  # Clear matchup advantage
        if self.recent_form.last_7_games:
            score += 10  # Recent data available
        if self.platoon_splits:
            score += 10  # Splits available
        if self.weather.temperature != 72:  # Not just default
            score += 5
        
        # Red flags reduce confidence
        if self.lineup_spot.platoon_status.startswith("platoon"):
            score -= 10  # May be benched vs certain handedness
        if self.opposing_pitcher and self.opposing_pitcher.is_ace:
            score -= 15  # Tough matchup uncertainty
        if self.recent_form.is_cold:
            score -= 10
        
        return max(0, min(100, score))
    
    def generate_reasoning(self) -> List[str]:
        """Generate human-readable reasoning for the decision."""
        reasons = []
        
        # Game environment
        if self.implied_team_runs > 5.0:
            reasons.append(f"High run environment ({self.implied_team_runs:.1f} implied runs)")
        elif self.implied_team_runs < 3.5:
            reasons.append(f"Low run environment ({self.implied_team_runs:.1f} implied runs)")
        
        # Park factor
        if self.park_factor > 1.10:
            reasons.append(f"Great hitter's park ({self.park_factor:.2f}x)")
        elif self.park_factor < 0.95:
            reasons.append(f"Tough pitcher's park ({self.park_factor:.2f}x)")
        
        # Weather
        if self.weather.hitter_friendly_score > 7:
            reasons.append(f"Hitter-friendly weather ({self.weather.temperature}°, {self.weather.wind_direction} wind)")
        
        # Opposing pitcher
        if self.opposing_pitcher:
            if self.opposing_pitcher.is_ace:
                reasons.append(f"⚠️ Tough matchup vs {self.opposing_pitcher.name} ({self.opposing_pitcher.era:.2f} ERA)")
            elif self.opposing_pitcher.is_streamable:
                reasons.append(f"✓ Target matchup vs {self.opposing_pitcher.name} ({self.opposing_pitcher.era:.2f} ERA)")
            
            if self.opposing_pitcher.is_tired:
                reasons.append(f"Pitcher on short rest ({self.opposing_pitcher.days_rest} days)")
        
        # Recent form
        if self.recent_form.is_hot:
            reasons.append(f"🔥 Hot streak: {self.recent_form.trend_7d:+.0f}% vs baseline")
        elif self.recent_form.is_cold:
            reasons.append(f"❄️ Cold streak: {self.recent_form.trend_7d:+.0f}% vs baseline")
        
        if self.recent_form.is_unlucky:
            reasons.append("📈 Positive regression candidate (underperforming xwOBA)")
        
        # Platoon
        if self.platoon_splits:
            vs_hand = "vs_lhp" if self.opposing_pitcher and self.opposing_pitcher.handedness == "L" else "vs_rhp"
            split = self.platoon_splits.get(vs_hand, 0)
            if split > 0.050:
                reasons.append(f"Strong platoon advantage ({split:.3f} wOBA split)")
            elif split < -0.030:
                reasons.append(f"Platoon disadvantage ({split:.3f} wOBA split)")
        
        # Category needs
        if self.category_needs:
            top_need = max(self.category_needs.items(), key=lambda x: x[1])
            if top_need[1] > 0:
                reasons.append(f"Helps with {top_need[0].upper()} (need {top_need[1]:.1f})")
        
        # Lineup spot
        if self.lineup_spot.spot <= 3:
            reasons.append(f"Premium lineup spot (#{self.lineup_spot.spot}, ~{self.lineup_spot.plate_appearance_multiplier:.1f}x PA)")
        elif self.lineup_spot.spot >= 8:
            reasons.append(f"Weak lineup spot (#{self.lineup_spot.spot}, fewer PA)")
        
        return reasons


@dataclass
class LineupDecisionReport:
    """Complete report for daily lineup decisions."""
    date: datetime
    strategy: MatchupStrategy
    risk_profile: RiskProfile
    
    # All players with context
    players: List[PlayerDecisionContext] = field(default_factory=list)
    
    # Final lineup
    starters: List[PlayerDecisionContext] = field(default_factory=list)
    bench: List[PlayerDecisionContext] = field(default_factory=list)
    
    # Summary stats
    avg_confidence: float = 0.0
    expected_categories: Dict[str, float] = field(default_factory=dict)
    
    def generate_summary(self) -> str:
        """Generate executive summary of lineup decisions."""
        lines = [
            f"=== Lineup Report for {self.date.strftime('%Y-%m-%d')} ===",
            f"Strategy: {self.strategy.value} | Risk: {self.risk_profile.value}",
            f"Average Confidence: {self.avg_confidence:.0f}%",
            "",
            "STARTERS:",
        ]
        
        for p in self.starters:
            emoji = "✓" if p.confidence_score > 70 else "?" if p.confidence_score > 50 else "⚠️"
            lines.append(f"  {emoji} {p.player_name} ({p.team}) - {p.confidence_score:.0f}% confidence")
            for reason in p.reasoning[:2]:  # Top 2 reasons
                lines.append(f"      • {reason}")
        
        lines.extend(["", "BENCH:",])
        for p in self.bench:
            emoji = "✓" if p.opposing_pitcher and p.opposing_pitcher.is_ace else "•"
            lines.append(f"  {emoji} {p.player_name} ({p.team})")
        
        return "\n".join(lines)


class EliteManagerContextBuilder:
    """
    Build complete decision context for elite fantasy baseball managers.
    
    This is the 'second brain' that elite managers keep in their heads:
    - Who's the pitcher? How has he been lately?
    - What's the weather? Wind blowing out?
    - Is this guy hot or cold?
    - Where's he hitting in the order?
    - Am I winning or losing each category?
    """
    
    def __init__(self):
        self.pitcher_cache: Dict[str, PitcherDeepDive] = {}
        self.weather_cache: Dict[str, WeatherContext] = {}
    
    def build_context(
        self,
        player: Dict,
        game_info: Dict,
        matchup_state: Dict[str, float],
        risk_profile: RiskProfile = RiskProfile.BALANCED,
    ) -> PlayerDecisionContext:
        """Build complete decision context for a single player."""
        
        ctx = PlayerDecisionContext(
            player_name=player.get("name", ""),
            team=player.get("team", ""),
            opponent=game_info.get("opponent", ""),
            season_projection=player.get("projections", {}),
            park_factor=game_info.get("park_factor", 1.0),
            implied_team_runs=game_info.get("implied_runs", 4.5),
            is_home=game_info.get("is_home", False),
            category_needs=matchup_state,
        )
        
        # Fetch pitcher deep dive
        ctx.opposing_pitcher = self._fetch_pitcher_deep_dive(ctx.opponent)
        
        # Fetch weather
        ctx.weather = self._fetch_weather(game_info.get("venue", ""))
        
        # Fetch recent form
        ctx.recent_form = self._fetch_recent_form(ctx.player_name)
        
        # Fetch lineup spot
        ctx.lineup_spot = self._fetch_lineup_spot(ctx.player_name, ctx.team)
        
        # Fetch platoon splits
        ctx.platoon_splits = self._fetch_platoon_splits(ctx.player_name)
        
        # Generate reasoning
        ctx.reasoning = ctx.generate_reasoning()
        ctx.confidence_score = ctx.calculate_confidence()
        
        return ctx
    
    def _fetch_pitcher_deep_dive(self, team: str) -> Optional[PitcherDeepDive]:
        """Fetch comprehensive pitcher stats."""
        # TODO: Integrate with FanGraphs for full deep dive
        if team in self.pitcher_cache:
            return self.pitcher_cache[team]
        return None
    
    def _fetch_weather(self, venue: str) -> WeatherContext:
        """Fetch weather for venue."""
        # TODO: Integrate with weather API
        if venue in self.weather_cache:
            return self.weather_cache[venue]
        return WeatherContext()
    
    def _fetch_recent_form(self, player_name: str) -> RecentForm:
        """Fetch recent performance data."""
        # TODO: Integrate with Statcast/pybaseball
        return RecentForm()
    
    def _fetch_lineup_spot(self, player_name: str, team: str) -> LineupSpot:
        """Fetch current lineup spot."""
        # TODO: Scrape from MLB.com or fetch from API
        return LineupSpot()
    
    def _fetch_platoon_splits(self, player_name: str) -> Dict[str, float]:
        """Fetch platoon splits."""
        # TODO: Integrate with platoon_fetcher
        return {}
    
    def determine_strategy(self, category_needs: Dict[str, float]) -> MatchupStrategy:
        """Determine optimal strategy based on matchup state."""
        winning = sum(1 for v in category_needs.values() if v > 0)
        losing = sum(1 for v in category_needs.values() if v < 0)
        
        if losing >= 5:
            return MatchupStrategy.AGGRESSIVE
        elif winning >= 6:
            return MatchupStrategy.PROTECTIVE
        else:
            return MatchupStrategy.STANDARD
