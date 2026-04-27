"""
Daily Briefing System for Elite Fantasy Managers.

Morning workflow:
1. Slack/Discord notification: "📋 Your Daily Briefing is Ready"
2. Click → Full report with every decision explained
3. One-tap approve or override any recommendation
4. Track decision accuracy over time
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

from backend.fantasy_baseball.elite_context import (
    PlayerDecisionContext,
    LineupDecisionReport,
    RiskProfile,
    MatchupStrategy,
    EliteManagerContextBuilder,
)
from backend.fantasy_baseball.smart_lineup_selector import SmartLineupSelector, CategoryNeed
from backend.fantasy_baseball.category_tracker import CategoryTracker, get_category_tracker
from backend.fantasy_baseball.decision_tracker import DecisionTracker, PlayerDecision, get_decision_tracker

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    """Actions manager can take on a recommendation."""
    APPROVE = "approve"      # Accept the recommendation
    OVERRIDE = "override"    # Manually change it
    MONITOR = "monitor"      # Wait for lineup confirmation
    PUNT = "punt"           # Intentionally sacrifice category


@dataclass
class PlayerBriefing:
    """Briefing for a single player decision."""
    player_name: str
    team: str
    opponent: str
    
    # The recommendation
    recommendation: str  # "START", "BENCH", "MONITOR"
    confidence: int  # 0-100
    
    # Quick stats
    proj_stats: Dict[str, float] = field(default_factory=dict)  # hr, r, rbi, etc.
    matchup_rating: str = "NEUTRAL"  # "FAVORABLE", "NEUTRAL", "TOUGH"
    
    # Key factors (top 3)
    key_factors: List[str] = field(default_factory=list)
    
    # Full context available on click
    full_context: Optional[PlayerDecisionContext] = None
    
    def to_card(self) -> Dict:
        """Convert to a UI card format."""
        emoji = {
            "START": "✅",
            "BENCH": "🪑", 
            "MONITOR": "⏳"
        }.get(self.recommendation, "❓")
        
        return {
            "emoji": emoji,
            "name": self.player_name,
            "team": self.team,
            "vs": self.opponent,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "rating": self.matchup_rating,
            "factors": self.key_factors[:3],
        }


@dataclass
class CategoryBriefing:
    """Briefing for each category state."""
    category: str
    current: float
    opponent: float
    projected_final: float
    opponent_projected: float
    status: str  # "WINNING", "LOSING", "TIED", "PUNT"
    urgency: int  # 0-10
    
    def to_summary(self) -> str:
        """One-line summary."""
        icons = {
            "WINNING": "🟢",
            "LOSING": "🔴",
            "TIED": "🟡",
            "PUNT": "⚪"
        }
        icon = icons.get(self.status, "⚪")
        return f"{icon} {self.category.upper()}: {self.current:.1f} vs {self.opponent:.1f}"


@dataclass
class DailyBriefing:
    """Complete morning briefing."""
    date: datetime
    generated_at: datetime
    
    # Executive summary
    strategy: str
    risk_profile: str
    overall_confidence: int
    
    # Quick stats
    total_decisions: int
    easy_decisions: int  # confidence > 80
    tough_decisions: int  # confidence < 60
    monitor_count: int  # Waiting on lineup confirmation
    
    # Categories
    categories: List[CategoryBriefing] = field(default_factory=list)
    
    # Players
    start_recommendations: List[PlayerBriefing] = field(default_factory=list)
    bench_recommendations: List[PlayerBriefing] = field(default_factory=list)
    monitor_list: List[PlayerBriefing] = field(default_factory=list)
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    # Weather
    weather_highlights: List[str] = field(default_factory=list)
    games_at_risk: List[str] = field(default_factory=list)
    
    # Meta
    pitcher_scratches: List[str] = field(default_factory=list)
    late_game_notices: List[str] = field(default_factory=list)
    
    def to_slack_blocks(self) -> List[Dict]:
        """Convert to Slack Block Kit format."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📋 Daily Briefing: {self.date.strftime('%A, %B %d')}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Strategy:* {self.strategy} | *Risk:* {self.risk_profile}\n"
                           f"*Confidence:* {self.overall_confidence}% | "
                           f"{self.easy_decisions} easy, {self.tough_decisions} tough decisions"
                }
            },
            {"type": "divider"}
        ]
        
        # Category standings
        cat_lines = ["*Category Standings:*"]
        for cat in self.categories:
            cat_lines.append(cat.to_summary())
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(cat_lines)}
        })
        
        blocks.append({"type": "divider"})
        
        # Alerts
        if self.alerts:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🚨 *Alerts:*\n" + "\n".join(f"• {a}" for a in self.alerts[:5])
                }
            })
        
        # Starters
        if self.start_recommendations:
            starter_lines = ["*Recommended Starters:*"]
            for p in self.start_recommendations[:9]:
                card = p.to_card()
                conf_emoji = "🟢" if p.confidence > 80 else "🟡" if p.confidence > 60 else "🔴"
                starter_lines.append(
                    f"{card['emoji']} {conf_emoji} *{p.player_name}* ({p.team})\n"
                    f"   {p.matchup_rating} matchup vs {p.opponent} | {p.confidence}% conf"
                )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(starter_lines)}
            })
        
        return blocks
    
    def to_discord_embed(self) -> Dict:
        """Convert to Discord embed format."""
        embed = {
            "title": f"📋 Daily Briefing: {self.date.strftime('%A, %B %d')}",
            "description": f"Strategy: {self.strategy} | Risk: {self.risk_profile}\n"
                         f"Overall Confidence: {self.overall_confidence}%",
            "color": 0x00ff00 if self.overall_confidence > 70 else 0xffaa00,
            "fields": [],
            "timestamp": self.generated_at.isoformat()
        }
        
        # Categories
        cat_value = "\n".join(c.to_summary() for c in self.categories[:6])
        embed["fields"].append({
            "name": "Category Standings",
            "value": cat_value,
            "inline": True
        })
        
        # Decisions
        decisions_value = (
            f"✅ Start: {len(self.start_recommendations)}\n"
            f"🪑 Bench: {len(self.bench_recommendations)}\n"
            f"⏳ Monitor: {len(self.monitor_list)}"
        )
        embed["fields"].append({
            "name": "Decisions",
            "value": decisions_value,
            "inline": True
        })
        
        # Alerts
        if self.alerts:
            embed["fields"].append({
                "name": "🚨 Alerts",
                "value": "\n".join(self.alerts[:5]),
                "inline": False
            })
        
        return embed


class DailyBriefingGenerator:
    """Generate daily briefings for elite managers."""
    
    def __init__(self, record_decisions: bool = True):
        self.context_builder = EliteManagerContextBuilder()
        self.smart_selector = SmartLineupSelector()
        self.category_tracker = get_category_tracker()
        self.decision_tracker = get_decision_tracker()
        self.record_decisions = record_decisions
    
    def generate(
        self,
        roster: List[Dict],
        projections: List[Dict],
        game_date: Optional[str] = None,
    ) -> DailyBriefing:
        """Generate complete daily briefing."""
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")
        
        date_obj = datetime.strptime(game_date, "%Y-%m-%d")
        
        # Get category needs
        try:
            category_needs = self.category_tracker.get_category_needs()
        except Exception as e:
            logger.warning(f"Could not fetch category needs: {e}")
            category_needs = []
        
        # Get smart lineup
        smart_rankings, warnings = self.smart_selector.select_optimal_lineup(
            roster=roster,
            projections=projections,
            game_date=game_date,
            category_needs=category_needs,
        )
        
        # Determine strategy
        strategy = self._determine_strategy(category_needs)
        
        # Build player briefings
        starters = []
        bench = []
        monitor = []
        
        for ranking in smart_rankings:
            # Get full context
            game_info = {
                "opponent": self._get_opponent(ranking.team, game_date),
                "park_factor": ranking.park_factor,
                "implied_runs": ranking.implied_team_runs,
                "is_home": ranking.is_home,
            }
            
            matchup_state = {c.category: c.needed for c in category_needs}
            
            # Build full context (expensive, only for starters)
            if ranking.smart_score > smart_rankings[8].smart_score if len(smart_rankings) > 8 else True:
                context = self.context_builder.build_context(
                    player={"name": ranking.name, "team": ranking.team, "projections": {}},
                    game_info=game_info,
                    matchup_state=matchup_state,
                )
            else:
                context = None
            
            # Determine recommendation
            has_game = ranking.has_game
            is_starter = len(starters) < 9 and has_game
            
            if not has_game:
                recommendation = "BENCH"
                confidence = 95
            elif context and context.opposing_pitcher and context.opposing_pitcher.is_ace:
                recommendation = "MONITOR"
                confidence = max(30, ranking.smart_score * 10)
            elif is_starter:
                recommendation = "START"
                confidence = min(95, 50 + ranking.smart_score * 5)
            else:
                recommendation = "BENCH"
                confidence = min(95, 50 + (smart_rankings[8].smart_score - ranking.smart_score) * 10) if len(smart_rankings) > 8 else 50
            
            briefing = PlayerBriefing(
                player_name=ranking.name,
                team=ranking.team,
                opponent=game_info.get("opponent", ""),
                recommendation=recommendation,
                confidence=int(confidence),
                proj_stats={
                    "hr": ranking.proj_hr,
                    "r": ranking.proj_r,
                    "rbi": ranking.proj_rbi,
                },
                matchup_rating=self._rate_matchup(ranking),
                key_factors=self._extract_key_factors(ranking, context),
                full_context=context,
            )
            
            if recommendation == "START":
                starters.append(briefing)
            elif recommendation == "MONITOR":
                monitor.append(briefing)
            else:
                bench.append(briefing)
        
        # Build category briefings
        cat_briefings = []
        for need in category_needs:
            status = "WINNING" if need.needed > 0 else "LOSING" if need.needed < 0 else "TIED"
            cat_briefings.append(CategoryBriefing(
                category=need.category,
                current=need.current,
                opponent=need.projected_opponent,
                projected_final=need.current * 7,  # Rough projection
                opponent_projected=need.projected_opponent * 7,
                status=status,
                urgency=int(need.urgency * 10),
            ))
        
        # Calculate overall confidence
        all_confidences = [p.confidence for p in starters + bench]
        overall_confidence = int(sum(all_confidences) / len(all_confidences)) if all_confidences else 50
        
        # Generate alerts
        alerts = self._generate_alerts(starters, warnings)
        
        briefing = DailyBriefing(
            date=date_obj,
            generated_at=datetime.now(),
            strategy=strategy.value,
            risk_profile="BALANCED",
            overall_confidence=overall_confidence,
            total_decisions=len(roster),
            easy_decisions=sum(1 for p in starters + bench if p.confidence > 80),
            tough_decisions=sum(1 for p in starters + bench if p.confidence < 60),
            monitor_count=len(monitor),
            categories=cat_briefings,
            start_recommendations=starters,
            bench_recommendations=bench,
            monitor_list=monitor,
            alerts=alerts,
        )
        
        # Record all decisions for tracking
        if self.record_decisions:
            self._record_all_decisions(briefing, smart_rankings, game_date)
        
        return briefing
    
    def _record_all_decisions(
        self,
        briefing: DailyBriefing,
        rankings: List,
        game_date: str
    ) -> None:
        """Record all lineup decisions to the tracker."""
        recorded = 0
        
        # Create lookup for ranking data
        ranking_map = {r.name: r for r in rankings}
        
        for player_briefing in (
            briefing.start_recommendations + 
            briefing.bench_recommendations + 
            briefing.monitor_list
        ):
            try:
                ranking = ranking_map.get(player_briefing.player_name)
                if not ranking:
                    continue
                
                # Build projected stats
                proj_stats = {
                    "hr": ranking.proj_hr,
                    "r": ranking.proj_r,
                    "rbi": ranking.proj_rbi,
                    "sb": ranking.proj_sb,
                    "avg": ranking.proj_avg,
                }
                
                # Get weather factor
                weather_factor = getattr(ranking, 'hr_factor', 1.0)
                
                # Get venue from ranking context if available
                venue = ""
                if ranking.weather and hasattr(ranking.weather, 'roof_closed'):
                    # Weather context exists, try to find venue
                    venue = self._get_venue_for_team(ranking.team)
                
                decision = PlayerDecision(
                    decision_id=f"{player_briefing.player_name.replace(' ', '_').lower()}_{game_date}",
                    date=game_date,
                    player_name=player_briefing.player_name,
                    player_id=player_briefing.player_name.replace(' ', '_').lower(),
                    team=player_briefing.team,
                    recommended_action=player_briefing.recommendation,
                    confidence=player_briefing.confidence,
                    factors=player_briefing.key_factors,
                    opponent=player_briefing.opponent,
                    opposing_pitcher=ranking.opposing_pitcher.name if ranking.opposing_pitcher else None,
                    venue=venue,
                    weather_factor=weather_factor,
                    projected_stats=proj_stats,
                )
                
                self.decision_tracker.record_decision(decision)
                recorded += 1
                
            except Exception as e:
                logger.warning(f"Failed to record decision for {player_briefing.player_name}: {e}")
        
        logger.info(f"Recorded {recorded} decisions for {game_date}")
    
    def _get_venue_for_team(self, team: str) -> str:
        """Get home venue for a team."""
        # Simple mapping - could be expanded
        venue_map = {
            "CHC": "Wrigley Field",
            "BOS": "Fenway Park",
            "NYY": "Yankee Stadium",
            "COL": "Coors Field",
            "SF": "Oracle Park",
            "LAD": "Dodger Stadium",
            "SD": "Petco Park",
            "NYM": "Citi Field",
            "PHI": "Citizens Bank Park",
            "WSH": "Nationals Park",
            "ATL": "Truist Park",
            "MIA": "LoanDepot Park",
            "CIN": "Great American Ball Park",
            "STL": "Busch Stadium",
            "PIT": "PNC Park",
            "MIL": "American Family Field",
            "ARI": "Chase Field",
            "SEA": "T-Mobile Park",
            "HOU": "Minute Maid Park",
            "TEX": "Globe Life Field",
            "TB": "Tropicana Field",
            "BAL": "Camden Yards",
            "TOR": "Rogers Centre",
            "CLE": "Progressive Field",
            "DET": "Comerica Park",
            "CWS": "Guaranteed Rate Field",
            "MIN": "Target Field",
            "KC": "Kauffman Stadium",
            "LAA": "Angel Stadium",
            "OAK": "Oakland Coliseum",
        }
        return venue_map.get(team.upper(), "")
    
    def _determine_strategy(self, category_needs: List[CategoryNeed]) -> MatchupStrategy:
        """Determine optimal strategy."""
        if not category_needs:
            return MatchupStrategy.STANDARD
        
        winning = sum(1 for c in category_needs if c.needed > 0)
        losing = sum(1 for c in category_needs if c.needed < 0)
        
        if losing >= 5:
            return MatchupStrategy.AGGRESSIVE
        elif winning >= 6:
            return MatchupStrategy.PROTECTIVE
        else:
            return MatchupStrategy.STANDARD
    
    def _get_opponent(self, team: str, game_date: str) -> str:
        """Get opponent for team on date."""
        # TODO: Use schedule fetcher
        return "TBD"
    
    def _rate_matchup(self, ranking) -> str:
        """Rate matchup quality."""
        if not ranking.opposing_pitcher:
            return "NEUTRAL"
        if ranking.opposing_pitcher.is_ace:
            return "TOUGH"
        if ranking.opposing_pitcher.is_streamable:
            return "FAVORABLE"
        return "NEUTRAL"
    
    def _extract_key_factors(
        self, 
        ranking, 
        context: Optional[PlayerDecisionContext]
    ) -> List[str]:
        """Extract top 3 factors for quick view."""
        factors = []
        
        if not ranking.has_game:
            factors.append("No game today")
            return factors
        
        if ranking.opposing_pitcher:
            if ranking.opposing_pitcher.is_ace:
                factors.append(f"⚠️ vs ace {ranking.opposing_pitcher.name}")
            elif ranking.opposing_pitcher.is_streamable:
                factors.append(f"✓ Target {ranking.opposing_pitcher.name}")
        
        # Weather factors (NEW)
        if ranking.weather:
            if ranking.weather.hitter_friendly_score > 7:
                factors.append(f"☀️ Hitter weather ({ranking.weather.temperature}°F)")
            elif ranking.weather.hitter_friendly_score < 3:
                factors.append(f"❄️ Pitcher weather ({ranking.weather.temperature}°F)")
            
            if ranking.weather.wind_direction == "out" and ranking.weather.wind_speed > 10:
                factors.append(f"💨 {ranking.weather.wind_speed}mph wind OUT")
            elif ranking.weather.wind_direction == "in" and ranking.weather.wind_speed > 10:
                factors.append(f"🌬️ {ranking.weather.wind_speed}mph wind IN")
        
        # HR factor for power hitters
        if ranking.hr_factor > 1.15 and ranking.proj_hr > 0.2:
            factors.append(f"⚾ HR boost ({ranking.hr_factor:.2f}x)")
        elif ranking.hr_factor < 0.85 and ranking.proj_hr > 0.2:
            factors.append(f"⚾ HR suppress ({ranking.hr_factor:.2f}x)")
        
        if ranking.implied_team_runs > 5.0:
            factors.append(f"🔥 High run env ({ranking.implied_team_runs:.1f})")
        elif ranking.implied_team_runs < 3.5:
            factors.append(f"❄️ Low run env ({ranking.implied_team_runs:.1f})")
        
        if ranking.park_factor > 1.10:
            factors.append(f"⚾ Hitter park ({ranking.park_factor:.2f}x)")
        
        if context:
            if context.recent_form.is_hot:
                factors.append("🔥 Hot streak")
            elif context.recent_form.is_cold:
                factors.append("❄️ Cold streak")
        
        return factors[:3]
    
    def _generate_alerts(
        self, 
        starters: List[PlayerBriefing], 
        warnings: List[str]
    ) -> List[str]:
        """Generate actionable alerts."""
        alerts = []
        
        # Low confidence starters
        low_conf = [s for s in starters if s.confidence < 60]
        if low_conf:
            names = ", ".join(p.player_name for p in low_conf[:3])
            alerts.append(f"Low confidence starters: {names}")
        
        # Ace matchups
        ace_matchups = [s for s in starters if "ace" in str(s.key_factors).lower()]
        if ace_matchups:
            alerts.append(f"⚠️ {len(ace_matchups)} starters facing aces - consider benching")
        
        # Add warnings from selector
        for w in warnings[:3]:
            alerts.append(w)
        
        return alerts


def get_briefing_generator(record_decisions: bool = True) -> DailyBriefingGenerator:
    """Factory function."""
    return DailyBriefingGenerator(record_decisions=record_decisions)
