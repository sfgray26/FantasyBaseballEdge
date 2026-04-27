"""
Draft Analytics Engine - Statcast-Powered Recommendations

Integrates advanced metrics into the draft engine to provide:
1. Regression alerts (buy low / sell high)
2. Breakout candidate flags
3. Injury risk warnings
4. Stuff+ and pitch quality rankings
5. Barrel% and exit velocity leaderboards
6. Platoon split recommendations
7. Park factor adjustments

This creates a true competitive edge over standard fantasy platforms
that rely on surface-level stats.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.fantasy_baseball.advanced_metrics import (
    BATTING_THRESHOLDS,
    PITCHING_THRESHOLDS,
    StatcastBatter,
    StatcastPitcher,
    analyze_batter_regression,
    analyze_pitcher_regression,
    calculate_batter_contact_score,
    calculate_batter_discipline_score,
    calculate_batter_power_score,
    calculate_batter_speed_score,
    calculate_injury_risk_score,
    calculate_pitcher_stuff_score,
    calculate_pitcher_whiff_score,
    is_breakout_candidate_batter,
    is_breakout_candidate_pitcher,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projections"


@dataclass
class DraftRecommendation:
    """Recommendation for a player in the draft."""
    player_name: str
    player_id: str
    recommendation_type: str  # TARGET, AVOID, REACH, VALUE, BREAKOUT, REGRESSION
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    
    # Advanced metrics summary
    power_score: int = 0
    contact_score: int = 0
    speed_score: int = 0
    stuff_score: int = 0
    injury_risk: int = 0
    
    # Fantasy implications
    projected_value: float = 0.0  # $ value
    adp_value_gap: float = 0.0  # Positive = drafted later than worth


class DraftAnalyticsEngine:
    """
    Analytics engine that provides Statcast-powered draft recommendations.
    """
    
    def __init__(self):
        self.batters: Dict[str, StatcastBatter] = {}
        self.pitchers: Dict[str, StatcastPitcher] = {}
        self.recommendations: List[DraftRecommendation] = []
        
    def load_advanced_metrics(self) -> None:
        """Load advanced metrics from CSV files."""
        # Load batting metrics
        batting_file = DATA_DIR / "advanced_batting_2026.csv"
        if batting_file.exists():
            with open(batting_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = StatcastBatter(
                        name=row["Name"],
                        barrel_pct=float(row.get("Barrel_Pct", 0)),
                        exit_velo_avg=float(row.get("Exit_Velo", 0)),
                        hard_hit_pct=float(row.get("Hard_Hit_Pct", 0)),
                        sweet_spot_pct=float(row.get("Sweet_Spot_Pct", 0)),
                        xba=float(row.get("xBA", 0)),
                        xslg=float(row.get("xSLG", 0)),
                        xwoba=float(row.get("xwOBA", 0)),
                        xwoba_diff=float(row.get("xwOBA_Diff", 0)),
                        o_swing_pct=float(row.get("O_Swing_Pct", 0)),
                        z_contact_pct=float(row.get("Z_Contact_Pct", 0)),
                        swstr_pct=float(row.get("SwStr_Pct", 0)),
                        sprint_speed=float(row.get("Sprint_Speed", 0)),
                        power_score=float(row.get("Power_Score", 0)),
                        contact_score=float(row.get("Contact_Score", 0)),
                        discipline_score=float(row.get("Discipline_Score", 0)),
                        speed_score=float(row.get("Speed_Score", 0)),
                        overall_score=float(row.get("Overall_Score", 0)),
                    )
                    self.batters[metrics.name.lower().replace(" ", "_")] = metrics
        
        # Load pitching metrics
        pitching_file = DATA_DIR / "advanced_pitching_2026.csv"
        if pitching_file.exists():
            with open(pitching_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = StatcastPitcher(
                        name=row["Name"],
                        stuff_plus=float(row.get("Stuff_Plus", 100)),
                        location_plus=float(row.get("Location_Plus", 100)),
                        fb_velo_avg=float(row.get("FB_Velo", 0)),
                        spin_rate_fb=int(row.get("Spin_Rate_FB", 0)),
                        whiff_pct=float(row.get("Whiff_Pct", 0)),
                        chase_pct=float(row.get("Chase_Pct", 0)),
                        barrel_allowed_pct=float(row.get("Barrel_Allowed_Pct", 0)),
                        xera=float(row.get("xERA", 0)),
                        xera_diff=float(row.get("xERA_Diff", 0)),
                        stuff_score=float(row.get("Stuff_Score", 0)),
                        whiff_score=float(row.get("Whiff_Score", 0)),
                        overall_score=float(row.get("Overall_Score", 0)),
                    )
                    self.pitchers[metrics.name.lower().replace(" ", "_")] = metrics
        
        logger.info(f"Loaded {len(self.batters)} batters and {len(self.pitchers)} pitchers with advanced metrics")
    
    def generate_recommendations(self) -> List[DraftRecommendation]:
        """Generate all draft recommendations based on advanced metrics."""
        recommendations = []
        
        # Analyze batters
        for player_id, metrics in self.batters.items():
            rec = self._analyze_batter(player_id, metrics)
            if rec:
                recommendations.append(rec)
        
        # Analyze pitchers
        for player_id, metrics in self.pitchers.items():
            rec = self._analyze_pitcher(player_id, metrics)
            if rec:
                recommendations.append(rec)
        
        # Sort by confidence and value gap
        recommendations.sort(key=lambda r: (r.confidence, abs(r.adp_value_gap)), reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_batter(self, player_id: str, metrics: StatcastBatter) -> Optional[DraftRecommendation]:
        """Analyze a batter and generate recommendation."""
        reasons = []
        rec_type = "NEUTRAL"
        confidence = 0.5
        
        # Check for regression signals
        regression_verdict, regression_conf = analyze_batter_regression(metrics)
        if regression_verdict == "BUY_LOW":
            reasons.append(f"Unlucky xwOBA ({metrics.xwoba_diff:+.3f}) - positive regression coming")
            rec_type = "BUY_LOW"
            confidence = regression_conf
        elif regression_verdict == "SELL_HIGH":
            reasons.append(f"Lucky xwOBA ({metrics.xwoba_diff:+.3f}) - regression risk")
            rec_type = "SELL_HIGH"
            confidence = regression_conf
        
        # Check for elite power
        if metrics.barrel_pct > BATTING_THRESHOLDS["barrel_pct"]["elite"]:
            reasons.append(f"Elite barrel% ({metrics.barrel_pct:.1f}%) - 95th percentile power")
            if rec_type == "NEUTRAL":
                rec_type = "TARGET"
                confidence = max(confidence, 0.7)
        
        # Check for speed
        if metrics.sprint_speed > BATTING_THRESHOLDS["sprint_speed"]["elite"]:
            reasons.append(f"Elite speed ({metrics.sprint_speed:.1f} ft/s) - 30+ SB potential")
            if rec_type == "NEUTRAL":
                rec_type = "TARGET"
                confidence = max(confidence, 0.65)
        
        # Check for poor discipline (red flag)
        if metrics.o_swing_pct > BATTING_THRESHOLDS["o_swing_pct"]["poor"]:
            reasons.append(f"High chase% ({metrics.o_swing_pct:.1f}%) - OBP risk")
            if rec_type == "NEUTRAL":
                rec_type = "AVOID"
                confidence = 0.6
        
        if not reasons:
            return None
        
        return DraftRecommendation(
            player_name=metrics.name,
            player_id=player_id,
            recommendation_type=rec_type,
            confidence=round(confidence, 2),
            reasons=reasons,
            power_score=int(metrics.power_score),
            contact_score=int(metrics.contact_score),
            speed_score=int(metrics.speed_score),
        )
    
    def _analyze_pitcher(self, player_id: str, metrics: StatcastPitcher) -> Optional[DraftRecommendation]:
        """Analyze a pitcher and generate recommendation."""
        reasons = []
        rec_type = "NEUTRAL"
        confidence = 0.5
        
        # Check for regression signals
        regression_verdict, regression_conf = analyze_pitcher_regression(metrics)
        if regression_verdict == "BUY_LOW":
            reasons.append(f"Unlucky ERA (xERA {metrics.xera:.2f} vs actual) - positive regression")
            rec_type = "BUY_LOW"
            confidence = regression_conf
        elif regression_verdict == "SELL_HIGH":
            reasons.append(f"Lucky ERA (xERA {metrics.xera:.2f} vs actual) - regression risk")
            rec_type = "SELL_HIGH"
            confidence = regression_conf
        
        # Check for elite stuff
        if metrics.stuff_plus > PITCHING_THRESHOLDS["stuff_plus"]["elite"]:
            reasons.append(f"Elite stuff+ ({metrics.stuff_plus:.0f}) - dominant arsenal")
            if rec_type == "NEUTRAL":
                rec_type = "TARGET"
                confidence = max(confidence, 0.75)
        
        # Check for high whiff ability
        if metrics.whiff_pct > PITCHING_THRESHOLDS["whiff_pct"]["elite"]:
            reasons.append(f"Elite whiff% ({metrics.whiff_pct:.1f}%) - high K upside")
        
        # Check for velocity concerns
        if metrics.velo_decline > 1.5:
            reasons.append(f"Velo decline ({metrics.velo_decline:.1f} mph) - injury risk")
            if rec_type == "NEUTRAL":
                rec_type = "AVOID"
                confidence = 0.7
        
        # Check for breakout (high stuff, improving)
        if metrics.stuff_plus > 115 and metrics.velo_decline < 0:
            reasons.append(f"Breakout candidate ({metrics.stuff_plus:.0f} stuff+) - velo up")
            if rec_type == "NEUTRAL":
                rec_type = "BREAKOUT"
                confidence = 0.7
        
        if not reasons:
            return None
        
        return DraftRecommendation(
            player_name=metrics.name,
            player_id=player_id,
            recommendation_type=rec_type,
            confidence=round(confidence, 2),
            reasons=reasons,
            stuff_score=int(metrics.stuff_score),
            injury_risk=int(calculate_injury_risk_score(metrics)),
        )
    
    def get_targets(self, min_confidence: float = 0.6) -> List[DraftRecommendation]:
        """Get buy low / breakout targets."""
        return [r for r in self.recommendations 
                if r.recommendation_type in ("BUY_LOW", "BREAKOUT", "TARGET")
                and r.confidence >= min_confidence]
    
    def get_avoids(self, min_confidence: float = 0.6) -> List[DraftRecommendation]:
        """Get sell high / avoid recommendations."""
        return [r for r in self.recommendations 
                if r.recommendation_type in ("SELL_HIGH", "AVOID")
                and r.confidence >= min_confidence]
    
    def get_player_report(self, player_name: str) -> str:
        """Generate detailed advanced metric report for a player."""
        player_id = player_name.lower().replace(" ", "_")
        
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"ADVANCED METRICS REPORT: {player_name}")
        report.append(f"{'='*60}\n")
        
        if player_id in self.batters:
            m = self.batters[player_id]
            report.append("BATTING PROFILE:")
            report.append(f"  Power: {m.power_score:.0f}/100 (Barrel%: {m.barrel_pct:.1f}%, EV: {m.exit_velo_avg:.1f} mph)")
            report.append(f"  Contact: {m.contact_score:.0f}/100 (Zone Contact: {m.z_contact_pct:.1f}%)")
            report.append(f"  Discipline: {m.discipline_score:.0f}/100 (Chase%: {m.o_swing_pct:.1f}%)")
            report.append(f"  Speed: {m.speed_score:.0f}/100 (Sprint: {m.sprint_speed:.1f} ft/s)")
            report.append(f"  Expected Stats: xBA {m.xba:.3f}, xSLG {m.xslg:.3f}, xwOBA {m.xwoba:.3f}")
            
            if m.xwoba_diff < -0.015:
                report.append(f"  ⚠️  BUY LOW: xwOBA {abs(m.xwoba_diff):.3f} higher than actual (unlucky)")
            elif m.xwoba_diff > 0.020:
                report.append(f"  ⚠️  SELL HIGH: xwOBA {m.xwoba_diff:.3f} lower than actual (lucky)")
        
        elif player_id in self.pitchers:
            m = self.pitchers[player_id]
            report.append("PITCHING PROFILE:")
            report.append(f"  Stuff: {m.stuff_score:.0f}/100 (Stuff+: {m.stuff_plus:.0f}, Velo: {m.fb_velo_avg:.1f} mph)")
            report.append(f"  Whiff Ability: {m.whiff_score:.0f}/100 (Whiff%: {m.whiff_pct:.1f}%, Chase%: {m.chase_pct:.1f}%)")
            report.append(f"  Injury Risk: {calculate_injury_risk_score(m):.0f}/100")
            report.append(f"  Expected Stats: xERA {m.xera:.2f}")
            
            if m.xera_diff > 0.50:
                report.append(f"  ⚠️  BUY LOW: xERA {m.xera_diff:.2f} higher than actual (unlucky)")
            elif m.xera_diff < -0.40:
                report.append(f"  ⚠️  SELL HIGH: xERA {abs(m.xera_diff):.2f} lower than actual (lucky)")
            
            if m.velo_decline > 1.5:
                report.append(f"  🚨 INJURY RISK: Velo down {m.velo_decline:.1f} mph from previous year")
        
        else:
            report.append("No advanced metrics available for this player.")
        
        return "\n".join(report)


def generate_draft_cheat_sheet() -> str:
    """Generate printable cheat sheet with advanced metric rankings."""
    engine = DraftAnalyticsEngine()
    engine.load_advanced_metrics()
    engine.generate_recommendations()
    
    lines = []
    lines.append("=" * 80)
    lines.append("STATCAST-POWERED DRAFT CHEAT SHEET 2026")
    lines.append("=" * 80)
    lines.append("")
    
    # Top targets
    lines.append("TOP TARGETS (Buy Low + Breakouts)")
    lines.append("-" * 80)
    for rec in engine.get_targets(min_confidence=0.65)[:15]:
        lines.append(f"{rec.player_name:25} | {rec.recommendation_type:12} | {rec.confidence:.0%} | {', '.join(rec.reasons[:1])}")
    lines.append("")
    
    # Players to avoid
    lines.append("AVOID / SELL HIGH (Regression Risks)")
    lines.append("-" * 80)
    for rec in engine.get_avoids(min_confidence=0.60)[:10]:
        lines.append(f"{rec.player_name:25} | {rec.recommendation_type:12} | {rec.confidence:.0%} | {', '.join(rec.reasons[:1])}")
    lines.append("")
    
    # Power targets
    lines.append("ELITE POWER TARGETS (12%+ Barrel%)")
    lines.append("-" * 80)
    power_hitters = sorted(
        [(n, b) for n, b in engine.batters.items() if b.barrel_pct > 12],
        key=lambda x: x[1].barrel_pct,
        reverse=True
    )[:10]
    for name, batter in power_hitters:
        lines.append(f"{batter.name:25} | {batter.barrel_pct:5.1f}% Barrel | {batter.exit_velo_avg:.1f} mph EV")
    lines.append("")
    
    # Speed targets
    lines.append("ELITE SPEED TARGETS (29+ ft/s Sprint Speed)")
    lines.append("-" * 80)
    speedsters = sorted(
        [(n, b) for n, b in engine.batters.items() if b.sprint_speed > 29],
        key=lambda x: x[1].sprint_speed,
        reverse=True
    )[:8]
    for name, batter in speedsters:
        lines.append(f"{batter.name:25} | {batter.sprint_speed:.1f} ft/s | {batter.speed_score:.0f} Speed Score")
    lines.append("")
    
    # Stuff+ leaders
    lines.append("ELITE STUFF TARGETS (120+ Stuff+)")
    lines.append("-" * 80)
    stuff_leaders = sorted(
        [(n, p) for n, p in engine.pitchers.items() if p.stuff_plus > 120],
        key=lambda x: x[1].stuff_plus,
        reverse=True
    )[:10]
    for name, pitcher in stuff_leaders:
        lines.append(f"{pitcher.name:25} | {pitcher.stuff_plus:.0f} Stuff+ | {pitcher.fb_velo_avg:.1f} mph")
    lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def inject_advanced_analytics(board: list) -> None:
    """
    Merge Statcast advanced metrics from advanced_*_2026.csv into board dicts in-place.

    Adds to each player dict:
      statcast      — raw Statcast numbers (barrel_pct, xwoba, stuff_plus, etc.)
      rec_type      — BUY_LOW / BREAKOUT / TARGET / SELL_HIGH / AVOID / NEUTRAL
      rec_reasons   — list of plain-text reasons
      rec_confidence — 0.0-1.0
      adp_gap       — adp minus rank (positive = model ranks player higher = value)

    Completely decoupled: any exception leaves the board unchanged.
    """
    try:
        engine = DraftAnalyticsEngine()
        engine.load_advanced_metrics()
        engine.generate_recommendations()

        rec_map = {r.player_id: r for r in engine.recommendations}

        for p in board:
            pid = p["id"]

            statcast: dict = {}
            if pid in engine.batters:
                m = engine.batters[pid]
                statcast = {
                    "barrel_pct": round(m.barrel_pct, 1),
                    "exit_velo": round(m.exit_velo_avg, 1),
                    "hard_hit_pct": round(m.hard_hit_pct, 1),
                    "xwoba": round(m.xwoba, 3),
                    "xwoba_diff": round(m.xwoba_diff, 3),
                    "sprint_speed": round(m.sprint_speed, 1),
                    "power_score": round(m.power_score),
                    "contact_score": round(m.contact_score),
                    "speed_score": round(m.speed_score),
                    "overall_score": round(m.overall_score),
                }
            elif pid in engine.pitchers:
                m = engine.pitchers[pid]
                statcast = {
                    "stuff_plus": round(m.stuff_plus),
                    "fb_velo": round(m.fb_velo_avg, 1),
                    "whiff_pct": round(m.whiff_pct, 1),
                    "xera": round(m.xera, 2),
                    "xera_diff": round(m.xera_diff, 2),
                    "stuff_score": round(m.stuff_score),
                    "whiff_score": round(m.whiff_score),
                    "overall_score": round(m.overall_score),
                }
            p["statcast"] = statcast

            if pid in rec_map:
                rec = rec_map[pid]
                p["rec_type"] = rec.recommendation_type
                p["rec_reasons"] = rec.reasons
                p["rec_confidence"] = rec.confidence
            else:
                p.setdefault("rec_type", "NEUTRAL")
                p.setdefault("rec_reasons", [])
                p.setdefault("rec_confidence", 0.0)

            p["adp_gap"] = round(p.get("adp", 999) - p.get("rank", p.get("adp", 999)), 1)

    except Exception:
        logger.exception("inject_advanced_analytics failed — board returned unmodified")


def compute_value_score(p: dict) -> float:
    """
    Combined value score used by the value-board endpoint.

    Components (all additive so failures degrade gracefully):
      z_score          — primary projection quality (already league-normalised)
      adp_gap / 30     — rewards players whose model rank beats their ADP
      statcast bonus   — normalised overall_score above 50 baseline
      rec bonus        — BUY_LOW / BREAKOUT lift; SELL_HIGH / AVOID penalty
      avoid penalty    — hard penalty for CSV-flagged injury avoids
    """
    z = p.get("z_score", 0.0)
    # Only apply ADP gap when we have real ADP data (adp < 500 = meaningfully ranked)
    # Clamp to [-2, +2] so extreme late-round discrepancies don't dominate the score
    raw_gap = p.get("adp_gap", 0.0)
    if p.get("adp", 999) < 500:
        adp_gap_bonus = max(-2.0, min(2.0, raw_gap / 30.0))
    else:
        adp_gap_bonus = 0.0

    statcast = p.get("statcast") or {}
    statcast_bonus = (statcast.get("overall_score", 50) - 50) / 100.0

    rec_bonuses = {
        "BUY_LOW":  0.30,
        "BREAKOUT": 0.40,
        "TARGET":   0.20,
        "NEUTRAL":  0.00,
        "SELL_HIGH": -0.30,
        "AVOID":    -0.40,
    }
    rec_bonus = rec_bonuses.get(p.get("rec_type", "NEUTRAL"), 0.0)
    avoid_penalty = -1.0 if p.get("avoid") else 0.0

    return z + adp_gap_bonus + statcast_bonus + rec_bonus + avoid_penalty


if __name__ == "__main__":
    print(generate_draft_cheat_sheet())
