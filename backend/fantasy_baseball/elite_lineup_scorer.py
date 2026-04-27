"""
ELITE LINEUP SCORER — Multi-Factor Daily Fantasy Optimization

Implements the research-backed scoring formula from daily_lineup_optimization_research.md:
- Multiplicative scoring (environment × matchup × platoon)
- Pitcher quality adjustments
- Platoon split weighting
- Recent form blending
- xwOBA regression indicators

As elite fantasy players know: It's not just about scoring runs—it's about:
1. WHO you're facing (pitcher quality)
2. YOUR specific advantage (platoon splits)
3. Recent momentum (form)
4. True talent (expected stats)
5. Opportunity (lineup spot, park)

Usage:
    from backend.fantasy_baseball.elite_lineup_scorer import EliteLineupScorer
    scorer = EliteLineupScorer()
    score = scorer.calculate_batter_score(player, game_context, matchup)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np

from backend.services.data_reliability_engine import (
    get_reliability_engine,
    DataSource,
    DataQualityTier,
)

logger = logging.getLogger(__name__)


@dataclass
class BatterProfile:
    """Complete batter profile for elite scoring."""
    player_id: str
    name: str
    team: str
    positions: List[str]
    
    # Season stats
    season_woba: float = 0.320
    season_avg: float = 0.250
    season_slg: float = 0.400
    season_wrc: float = 100.0
    
    # Platoon splits (CRITICAL)
    woba_vs_lhp: float = 0.320
    woba_vs_rhp: float = 0.320
    ops_vs_lhp: float = 0.720
    ops_vs_rhp: float = 0.720
    
    # Statcast expected stats
    xwoba: float = 0.320
    xslg: float = 0.400
    barrel_pct: float = 8.0
    hard_hit_pct: float = 35.0
    
    # Recent form (weighted rolling)
    last_7_woba: float = 0.320
    last_14_woba: float = 0.320
    last_30_woba: float = 0.320
    
    # Context
    lineup_spot: str = "5"  # 1-9
    games_last_7: int = 5
    
    # Data quality
    data_source: str = "steamer"
    data_quality_score: float = 1.0


@dataclass
class PitcherProfile:
    """Opposing pitcher profile for matchup adjustments."""
    name: str
    team: str
    handedness: str  # "L" or "R"
    
    # Quality metrics
    xera: float = 4.00
    xwoba_allowed: float = 0.320
    k_per_nine: float = 8.5
    bb_per_nine: float = 3.0
    whip: float = 1.30
    
    # Splits
    xwoba_vs_lhb: float = 0.320
    xwoba_vs_rhb: float = 0.320
    
    # Recent form
    last_3_starts_era: float = 4.00
    last_3_starts_ip: float = 6.0


@dataclass
class GameContext:
    """Game environment context."""
    implied_runs: float = 4.5
    park_factor: float = 1.0
    is_home: bool = True
    game_time: Optional[datetime] = None
    weather: Optional[Dict] = None


@dataclass
class EliteScore:
    """Complete scoring breakdown for transparency."""
    total_score: float
    
    # Component scores
    environment_score: float
    matchup_multiplier: float
    platoon_multiplier: float
    form_adjusted_woba: float
    regression_boost: float
    lineup_spot_bonus: float
    
    # Metadata
    confidence: float
    data_quality: str
    reasoning: str


class EliteLineupScorer:
    """
    Elite daily lineup scorer using multi-factor optimization.
    
    Key insight from research: Multiplicative beats additive.
    A great hitter in a great matchup in a great park is EXPONENTIALLY
    better, not additively better.
    """
    
    # League averages (2024 MLB)
    LEAGUE_AVG_WOBA = 0.310
    LEAGUE_AVG_ERA = 4.25
    LEAGUE_AVG_XERA = 4.20
    
    # Form weighting (30% recent, 70% season - research-backed)
    RECENT_FORM_WEIGHT = 0.30
    SEASON_FORM_WEIGHT = 0.70
    
    # Regression thresholds (Statcast research)
    XWOBA_LUCKY_THRESHOLD = 0.030   # wOBA > xwOBA by 30 pts = lucky
    XWOBA_UNLUCKY_THRESHOLD = 0.020  # xwOBA > wOBA by 20 pts = unlucky
    
    # Lineup spot bonuses (PA and RBI opportunity weighted)
    LINEUP_BONUSES = {
        '1': 0.05,   # Leadoff: most PA
        '2': 0.03,
        '3': 0.02,
        '4': 0.04,   # Cleanup: RBI opportunities
        '5': 0.02,
        '6': 0.00,
        '7': -0.01,
        '8': -0.02,
        '9': -0.03,  # Pitcher spot
    }
    
    def __init__(self):
        self.reliability = get_reliability_engine()
    
    def calculate_batter_score(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
        context: GameContext,
    ) -> EliteScore:
        """
        Calculate elite daily score for a batter.
        
        Formula:
            score = base × matchup_mult × platoon_mult + adjustments
        
        Where:
            base = implied_runs × park_factor
            matchup_mult = f(pitcher_quality vs batter)
            platoon_mult = f(batter_split vs pitcher_hand)
        """
        # Validate data quality first
        quality_check = self._validate_batter_data(batter)
        if quality_check.quality_tier == DataQualityTier.TIER_5_UNAVAILABLE:
            return self._fallback_score(batter.name)
        
        # 1. Base Environment (implied runs × park factor)
        environment_score = context.implied_runs * context.park_factor
        
        # 2. Matchup Quality Multiplier (CRITICAL - per research)
        # A .350 wOBA hitter vs 5.00 ERA pitcher = .380+ wOBA performance
        matchup_mult = self._calculate_matchup_multiplier(
            batter, pitcher
        )
        
        # 3. Platoon Split Multiplier (CRITICAL - 15-30% difference)
        platoon_mult = self._calculate_platoon_multiplier(
            batter, pitcher
        )
        
        # 4. Recent Form Adjustment (30% recent, 70% season)
        form_woba = self._blend_form(batter)
        form_mult = form_woba / self.LEAGUE_AVG_WOBA
        
        # 5. xwOBA Regression Indicator (buy low/sell high)
        regression_boost = self._calculate_regression_boost(batter)
        
        # 6. Lineup Spot Bonus (PA volume + RBI opportunity)
        lineup_bonus = self.LINEUP_BONUSES.get(batter.lineup_spot, 0.0)
        
        # COMBINE: Multiplicative for environment factors
        base_score = environment_score * matchup_mult * platoon_mult * form_mult
        
        # ADD: Linear adjustments for non-interacting factors
        adjusted_score = base_score + regression_boost + lineup_bonus
        
        # Confidence weighting (down-weight low-quality data)
        final_score = adjusted_score * quality_check.confidence_score
        
        # Build reasoning string
        reasoning = self._build_reasoning(
            batter, pitcher, context,
            matchup_mult, platoon_mult, form_woba,
            regression_boost, lineup_bonus
        )
        
        return EliteScore(
            total_score=round(final_score, 3),
            environment_score=round(environment_score, 3),
            matchup_multiplier=round(matchup_mult, 3),
            platoon_multiplier=round(platoon_mult, 3),
            form_adjusted_woba=round(form_woba, 3),
            regression_boost=round(regression_boost, 3),
            lineup_spot_bonus=round(lineup_bonus, 3),
            confidence=round(quality_check.confidence_score, 3),
            data_quality=quality_check.quality_tier.value,
            reasoning=reasoning,
        )
    
    def _calculate_matchup_multiplier(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
    ) -> float:
        """
        Calculate matchup quality multiplier.
        
        Elite insight: Pitcher quality shifts true talent by 20-30%.
        vs 3.00 ERA ace = -10% to offense
        vs 5.00 ERA gas can = +10% to offense
        """
        # Use xERA for stability (less noise than actual ERA)
        pitcher_xera = pitcher.xera
        
        # Linear adjustment around league average
        # 4.25 ERA = 1.0 (neutral)
        # 3.00 ERA = 0.90 (10% harder)
        # 5.50 ERA = 1.10 (10% easier)
        era_diff = self.LEAGUE_AVG_ERA - pitcher_xera
        multiplier = 1.0 + (era_diff * 0.08)
        
        # Clamp to reasonable range
        return max(0.85, min(1.15, multiplier))
    
    def _calculate_platoon_multiplier(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
    ) -> float:
        """
        Calculate platoon split multiplier.
        
        Elite insight: Average 15% wOBA difference, up to 30% for extremes.
        Kyle Schwarber: .920 OPS vs RHP, .650 vs LHP (29% difference!)
        """
        is_lhp = pitcher.handedness == "L"
        
        # Get platoon wOBA
        platoon_woba = batter.woba_vs_lhp if is_lhp else batter.woba_vs_rhp
        
        if platoon_woba == 0 or batter.season_woba == 0:
            return 1.0
        
        # Calculate multiplier
        mult = platoon_woba / batter.season_woba
        
        # Research shows most hitters are 10-15% better vs opposite hand
        # But we don't want to over-penalize reverse-split players
        # Clamp to reasonable range
        return max(0.80, min(1.25, mult))
    
    def _blend_form(self, batter: BatterProfile) -> float:
        """
        Blend recent form with season-long performance.
        
        Elite weighting: 30% recent, 70% season (regression to mean)
        Early season: trust prior more
        Late season: trust data more
        """
        season_woba = batter.season_woba
        recent_woba = batter.last_7_woba
        
        # If small sample (less than 3 games), weight prior more
        games_factor = min(batter.games_last_7 / 5.0, 1.0)
        recent_weight = self.RECENT_FORM_WEIGHT * games_factor
        
        blended = (
            recent_weight * recent_woba +
            (1 - recent_weight) * season_woba
        )
        
        return blended
    
    def _calculate_regression_boost(self, batter: BatterProfile) -> float:
        """
        Calculate regression indicator boost/penalty.
        
        Elite insight: xwOBA > wOBA by 20+ pts = "buy low" (unlucky)
        xwOBA < wOBA by 30+ pts = "sell high" (lucky)
        
        This identifies players due for positive/negative regression.
        """
        woba_diff = batter.xwoba - batter.season_woba
        
        if woba_diff > self.XWOBA_UNLUCKY_THRESHOLD:
            # Unlucky - expect positive regression
            # Cap at +0.03 to avoid over-weighting
            return min(0.03, woba_diff * 0.5)
        elif woba_diff < -self.XWOBA_LUCKY_THRESHOLD:
            # Lucky - expect negative regression
            # Cap at -0.03
            return max(-0.03, woba_diff * 0.5)
        
        return 0.0
    
    def _validate_batter_data(self, batter: BatterProfile):
        """Validate batter data quality."""
        # Build data dict for validation
        data = {
            "player_id": batter.player_id,
            "name": batter.name,
            "woba": batter.season_woba,
            "xwoba": batter.xwoba,
            "barrel_pct": batter.barrel_pct,
        }
        
        return self.reliability.validate_statcast_data(
            batter.player_id, data
        )
    
    def _fallback_score(self, player_name: str) -> EliteScore:
        """Return fallback score when data is unavailable."""
        return EliteScore(
            total_score=4.5,  # League average
            environment_score=4.5,
            matchup_multiplier=1.0,
            platoon_multiplier=1.0,
            form_adjusted_woba=self.LEAGUE_AVG_WOBA,
            regression_boost=0.0,
            lineup_spot_bonus=0.0,
            confidence=0.5,
            data_quality="tier_5_unavailable",
            reasoning=f"Limited data for {player_name}, using league average",
        )
    
    def _build_reasoning(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
        context: GameContext,
        matchup_mult: float,
        platoon_mult: float,
        form_woba: float,
        regression_boost: float,
        lineup_bonus: float,
    ) -> str:
        """Build human-readable reasoning string."""
        parts = []
        
        # Environment
        parts.append(f"{context.implied_runs:.1f}R env")
        if context.park_factor > 1.05:
            parts.append(f"hitter park ({context.park_factor:.2f}x)")
        elif context.park_factor < 0.95:
            parts.append(f"pitcher park ({context.park_factor:.2f}x)")
        
        # Matchup
        if matchup_mult > 1.05:
            parts.append(f"vs weak pitcher ({matchup_mult:.2f}x)")
        elif matchup_mult < 0.95:
            parts.append(f"vs strong pitcher ({matchup_mult:.2f}x)")
        
        # Platoon
        if platoon_mult > 1.10:
            parts.append(f"platoon adv ({platoon_mult:.2f}x)")
        elif platoon_mult < 0.90:
            parts.append(f"platoon dis ({platoon_mult:.2f}x)")
        
        # Regression
        if regression_boost > 0.01:
            parts.append(f"due for pos reg (+{regression_boost:.3f})")
        elif regression_boost < -0.01:
            parts.append(f"due for neg reg ({regression_boost:.3f})")
        
        # Lineup spot
        if lineup_bonus > 0:
            parts.append(f"good lineup spot (+{lineup_bonus:.2f})")
        
        return "; ".join(parts)
    
    def compare_to_simple_score(
        self,
        batter: BatterProfile,
        pitcher: PitcherProfile,
        context: GameContext,
    ) -> Dict[str, any]:
        """
        Compare elite score to simple implied-runs-only score.
        
        Shows the value added by elite multi-factor scoring.
        """
        elite = self.calculate_batter_score(batter, pitcher, context)
        
        # Simple score (old method)
        simple_score = context.implied_runs * context.park_factor
        
        return {
            "player": batter.name,
            "elite_score": elite.total_score,
            "simple_score": round(simple_score, 3),
            "value_added": round(elite.total_score - simple_score, 3),
            "percent_better": round(
                (elite.total_score - simple_score) / simple_score * 100, 1
            ),
            "breakdown": {
                "matchup_impact": f"{((elite.matchup_multiplier - 1) * 100):+.1f}%",
                "platoon_impact": f"{((elite.platoon_multiplier - 1) * 100):+.1f}%",
                "regression_boost": elite.regression_boost,
            }
        }


# Singleton
_elite_scorer: Optional[EliteLineupScorer] = None


def get_elite_scorer() -> EliteLineupScorer:
    """Get singleton elite scorer."""
    global _elite_scorer
    if _elite_scorer is None:
        _elite_scorer = EliteLineupScorer()
    return _elite_scorer
