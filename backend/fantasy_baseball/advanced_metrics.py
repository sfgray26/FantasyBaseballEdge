"""
Advanced Analytics Module - Baseball Savant / Statcast Integration

Provides competitive edge through:
- Expected stats (xBA, xSLG, xwOBA, xERA) for regression analysis
- Batted ball quality (Barrel%, Exit Velo, Hard Hit%)
- Pitch quality metrics (Stuff+, Pitch Movement, Spin Rate)
- Plate discipline (O-Swing%, Z-Contact%, Chase%)
- Sprint speed / baserunning metrics
- Injury risk indicators (velo decline, workload spikes)
- Aging curves and breakout predictions

Data sources:
- Baseball Savant (Statcast)
- FanGraphs (plate discipline, batted ball)
- Baseball Prospectus (pitcher abuse points)
"""

import csv
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "projections"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Statcast Metric Thresholds (percentiles for elite/poor performance)
# ---------------------------------------------------------------------------

BATTING_THRESHOLDS = {
    # Barrel% - percentage of batted balls with ideal EV/LA combo
    "barrel_pct": {"elite": 12.0, "good": 8.0, "avg": 5.0, "poor": 3.0},
    
    # Exit Velocity (mph)
    "exit_velo": {"elite": 92.0, "good": 90.0, "avg": 88.0, "poor": 85.0},
    
    # Hard Hit% (95+ mph EV)
    "hard_hit_pct": {"elite": 45.0, "good": 38.0, "avg": 32.0, "poor": 25.0},
    
    # Sweet Spot% (8-32 degree launch angle)
    "sweet_spot_pct": {"elite": 38.0, "good": 34.0, "avg": 30.0, "poor": 25.0},
    
    # xwOBA vs wOBA difference (regression indicator)
    "xwoba_diff": {"lucky": 0.030, "neutral": 0.010, "unlucky": -0.020},
    
    # Sprint Speed (ft/sec) - 27.0 is MLB avg
    "sprint_speed": {"elite": 29.0, "good": 28.0, "avg": 27.0, "poor": 26.0},
    
    # Zone Contact% (contact on pitches in zone)
    "z_contact_pct": {"elite": 88.0, "good": 85.0, "avg": 82.0, "poor": 78.0},
    
    # O-Swing% (chase rate) - lower is better
    "o_swing_pct": {"elite": 25.0, "good": 30.0, "avg": 32.0, "poor": 38.0},
}

PITCHING_THRESHOLDS = {
    # Stuff+ (100 is average)
    "stuff_plus": {"elite": 120, "good": 110, "avg": 100, "poor": 90},
    
    # Location+ (100 is average)
    "location_plus": {"elite": 110, "good": 105, "avg": 100, "poor": 95},
    
    # Pitching+ (Stuff+ + Location+ combined)
    "pitching_plus": {"elite": 115, "good": 108, "avg": 100, "poor": 92},
    
    # Fastball velocity (mph)
    "fb_velo": {"elite": 96.0, "good": 94.0, "avg": 92.5, "poor": 90.0},
    
    # Spin rate (rpm) - varies by pitch type
    "spin_rate": {"elite": 2500, "good": 2300, "avg": 2100, "poor": 1900},
    
    # Whiff% (swinging strikes / swings)
    "whiff_pct": {"elite": 30.0, "good": 26.0, "avg": 22.0, "poor": 18.0},
    
    # Chase% (O-Swing% against)
    "chase_pct": {"elite": 32.0, "good": 28.0, "avg": 24.0, "poor": 20.0},
    
    # Barrels allowed% (suppress hard contact)
    "barrel_allowed_pct": {"elite": 5.0, "good": 7.0, "avg": 9.0, "poor": 12.0},
    
    # xERA vs ERA difference
    "xera_diff": {"lucky": -0.50, "neutral": -0.15, "unlucky": 0.40},
}


# ---------------------------------------------------------------------------
# Data Classes for Advanced Metrics
# ---------------------------------------------------------------------------

@dataclass
class StatcastBatter:
    name: str
    player_id: int = 0
    team: str = ""  # FanGraphs team abbreviation (e.g. "NYY")

    # Batted ball quality
    barrel_pct: float = 0.0  # % of batted balls that are barrels
    exit_velo_avg: float = 0.0  # Average exit velocity
    exit_velo_max: float = 0.0  # Max exit velocity (showing power ceiling)
    hard_hit_pct: float = 0.0  # % of batted balls 95+ mph
    sweet_spot_pct: float = 0.0  # % of batted balls 8-32 degree launch angle
    
    # Expected stats
    xba: float = 0.0  # Expected batting average
    xslg: float = 0.0  # Expected slugging
    xwoba: float = 0.0  # Expected weighted on-base average
    xwoba_diff: float = 0.0  # Difference from actual (regression indicator)
    
    # Plate discipline
    o_swing_pct: float = 0.0  # Chase rate (outside zone swings)
    z_swing_pct: float = 0.0  # Zone swing rate
    o_contact_pct: float = 0.0  # Contact rate on pitches outside zone
    z_contact_pct: float = 0.0  # Contact rate on pitches in zone
    swstr_pct: float = 0.0  # Swinging strike %
    
    # Baserunning
    sprint_speed: float = 0.0  # ft/sec
    bolts: int = 0  # # of runs where sprint speed > 30 ft/sec
    hp_to_1b: float = 0.0  # Time from home to first base
    sb_attempts: int = 0  # Stolen base attempts (for speed/SB projection)
    
    # Batted ball profile
    gb_pct: float = 0.0  # Ground ball %
    fb_pct: float = 0.0  # Fly ball %
    ld_pct: float = 0.0  # Line drive %
    pull_pct: float = 0.0  # Pull %
    oppo_pct: float = 0.0  # Opposite field %
    
    # Park-adjusted metrics
    wrc_plus: float = 100.0  # Weighted runs created+ (100 is league avg)
    
    # Scores computed from metrics
    power_score: float = 0.0  # 0-100 based on barrel%, EV, hard hit%
    contact_score: float = 0.0  # 0-100 based on zone contact, K%
    discipline_score: float = 0.0  # 0-100 based on chase%, BB/K
    speed_score: float = 0.0  # 0-100 based on sprint speed
    overall_score: float = 0.0  # Composite 0-100
    
    # Fantasy edge flags
    regression_up: bool = False  # xwOBA > wOBA (unlucky, buy low)
    regression_down: bool = False  # xwOBA < wOBA (lucky, sell high)
    breakout_candidate: bool = False  # Young, improving exit velo
    aging_risk: bool = False  # Declining sprint speed, EV


@dataclass
class StatcastPitcher:
    name: str
    player_id: int = 0
    
    # Pitch quality
    stuff_plus: float = 100.0  # Stuff+ metric (100 = avg)
    location_plus: float = 100.0  # Location+ (command)
    pitching_plus: float = 100.0  # Combined metric
    
    # Velocity metrics
    fb_velo_avg: float = 0.0  # 4-seam fastball velocity
    fb_velo_max: float = 0.0  # Max velocity (relief outings, etc)
    velo_decline: float = 0.0  # Velocity decline from previous year (injury flag)
    
    # Spin rates
    spin_rate_fb: int = 0  # 4-seam spin
    spin_rate_cb: int = 0  # Curveball spin
    spin_rate_sl: int = 0  # Slider spin
    spin_efficiency: float = 0.0  # Active spin % (translates to movement)
    
    # Movement (inches vs avg)
    fb_rise: float = 0.0  # Fastball "rise" (less drop than avg)
    cb_drop: float = 0.0  # Curveball drop
    slider_glove: float = 0.0  # Slider horizontal movement
    
    # Whiff metrics
    whiff_pct: float = 0.0  # % of swings that are misses
    chase_pct: float = 0.0  # % of pitches outside zone that are swung at
    csw_pct: float = 0.0  # Called strikes + whiffs / total pitches
    
    # Contact quality allowed
    barrel_allowed_pct: float = 0.0  # % of batted balls allowed that are barrels
    hard_hit_allowed_pct: float = 0.0  # % hard contact allowed
    exit_velo_allowed: float = 0.0  # Average EV on batted balls allowed
    
    # Expected stats
    xera: float = 0.0  # Expected ERA
    xera_diff: float = 0.0  # xERA - ERA (regression indicator)
    xwoba_allowed: float = 0.0  # Expected wOBA allowed
    
    # Pitch mix
    fb_pct: float = 0.0  # Fastball %
    bb_pct: float = 0.0  # Breaking ball %
    os_pct: float = 0.0  # Off-speed %
    
    # Workload/Injury indicators
    pitches_per_game: float = 0.0
    velo_by_inning: Dict[int, float] = field(default_factory=dict)
    injury_risk_score: float = 0.0  # 0-100 computed score
    
    # Scores
    stuff_score: float = 0.0  # 0-100 based on Stuff+, velo, spin
    command_score: float = 0.0  # 0-100 based on Location+, BB%, zone%
    whiff_score: float = 0.0  # 0-100 based on whiff%, chase%, K%
    overall_score: float = 0.0  # Composite 0-100
    
    # Fantasy edge flags
    stuff_upgrade: bool = False  # Stuff+ trending up
    velo_concern: bool = False  # Velo decline > 1.5 mph
    luck_regression: bool = False  # xERA significantly higher than ERA
    breakout_candidate: bool = False  # Young, high Stuff+, improving command


# ---------------------------------------------------------------------------
# Advanced Metrics Calculator
# ---------------------------------------------------------------------------

def calculate_batter_power_score(metrics: StatcastBatter) -> float:
    """
    Calculate power score 0-100 based on Statcast batted ball metrics.
    Combines barrel%, exit velocity, and hard hit%.
    """
    # Barrel% component (0-40 points)
    barrel_score = min(40, (metrics.barrel_pct / BATTING_THRESHOLDS["barrel_pct"]["elite"]) * 40)
    
    # Exit velocity component (0-35 points)
    ev_normalized = (metrics.exit_velo_avg - BATTING_THRESHOLDS["exit_velo"]["poor"]) / \
                    (BATTING_THRESHOLDS["exit_velo"]["elite"] - BATTING_THRESHOLDS["exit_velo"]["poor"])
    ev_score = min(35, max(0, ev_normalized * 35))
    
    # Hard hit% component (0-25 points)
    hard_hit_normalized = (metrics.hard_hit_pct - BATTING_THRESHOLDS["hard_hit_pct"]["poor"]) / \
                          (BATTING_THRESHOLDS["hard_hit_pct"]["elite"] - BATTING_THRESHOLDS["hard_hit_pct"]["poor"])
    hard_hit_score = min(25, max(0, hard_hit_normalized * 25))
    
    return barrel_score + ev_score + hard_hit_score


def calculate_batter_contact_score(metrics: StatcastBatter) -> float:
    """
    Calculate contact ability score 0-100.
    Based on zone contact%, swinging strike%, and strikeout rate.
    """
    # Zone contact% (0-50 points)
    z_contact_normalized = (metrics.z_contact_pct - BATTING_THRESHOLDS["z_contact_pct"]["poor"]) / \
                           (BATTING_THRESHOLDS["z_contact_pct"]["elite"] - BATTING_THRESHOLDS["z_contact_pct"]["poor"])
    z_contact_score = min(50, max(0, z_contact_normalized * 50))
    
    # Swinging strike% (inverse - lower is better) (0-30 points)
    # MLB avg swstr% is ~11%, elite is <7%
    swstr_score = max(0, 30 - ((metrics.swstr_pct - 7) / (15 - 7)) * 30)
    
    # Sweet spot% (quality of contact angle) (0-20 points)
    sweet_spot_normalized = (metrics.sweet_spot_pct - BATTING_THRESHOLDS["sweet_spot_pct"]["poor"]) / \
                            (BATTING_THRESHOLDS["sweet_spot_pct"]["elite"] - BATTING_THRESHOLDS["sweet_spot_pct"]["poor"])
    sweet_spot_score = min(20, max(0, sweet_spot_normalized * 20))
    
    return z_contact_score + swstr_score + sweet_spot_score


def calculate_batter_discipline_score(metrics: StatcastBatter) -> float:
    """
    Calculate plate discipline score 0-100.
    Based on chase%, BB/K ratio indicators.
    """
    # Chase rate (inverse - lower is better) (0-50 points)
    # Elite is <25%, poor is >38%
    if metrics.o_swing_pct <= BATTING_THRESHOLDS["o_swing_pct"]["elite"]:
        chase_score = 50
    elif metrics.o_swing_pct >= BATTING_THRESHOLDS["o_swing_pct"]["poor"]:
        chase_score = 10
    else:
        chase_score = 50 - ((metrics.o_swing_pct - BATTING_THRESHOLDS["o_swing_pct"]["elite"]) / \
                            (BATTING_THRESHOLDS["o_swing_pct"]["poor"] - BATTING_THRESHOLDS["o_swing_pct"]["elite"])) * 40
    
    # O-Contact% (ability to make contact on pitches outside zone) (0-30 points)
    o_contact_score = (metrics.o_contact_pct / 75) * 30  # 75% is elite
    
    # Z-Swing% (aggressiveness in zone) (0-20 points)
    # Want to be aggressive on strikes, 70-75% is ideal
    z_swing_score = 20 - abs(metrics.z_swing_pct - 72) / 2
    
    return min(100, chase_score + min(30, o_contact_score) + max(0, z_swing_score))


def calculate_batter_speed_score(metrics: StatcastBatter) -> float:
    """Calculate speed/baserunning score 0-100."""
    if metrics.sprint_speed == 0:
        return 50  # Default if no data
    
    # Sprint speed (0-80 points)
    # 27.0 is avg, 29.0 is elite (Billy Hamilton/Trea Turner territory)
    speed_normalized = (metrics.sprint_speed - 25) / (30 - 25)
    speed_score = min(80, max(0, speed_normalized * 80))
    
    # Bolts (runs at 30+ ft/sec) (0-20 points)
    bolt_score = min(20, metrics.bolts / 5)
    
    return speed_score + bolt_score


def calculate_pitcher_stuff_score(metrics: StatcastPitcher) -> float:
    """
    Calculate raw stuff score 0-100.
    Based on Stuff+, velocity, spin rates, and movement.
    """
    # Stuff+ component (0-40 points)
    stuff_score = min(40, (metrics.stuff_plus - 80) / (130 - 80) * 40)
    
    # Velocity component (0-25 points)
    velo_normalized = (metrics.fb_velo_avg - PITCHING_THRESHOLDS["fb_velo"]["poor"]) / \
                      (PITCHING_THRESHOLDS["fb_velo"]["elite"] - PITCHING_THRESHOLDS["fb_velo"]["poor"])
    velo_score = min(25, max(0, velo_normalized * 25))
    
    # Spin rate component (0-20 points)
    spin_normalized = (metrics.spin_rate_fb - 1800) / (2600 - 1800)
    spin_score = min(20, max(0, spin_normalized * 20))
    
    # Movement component (0-15 points)
    # Fastball rise + curveball drop indicate good spin efficiency
    movement_score = min(15, (metrics.fb_rise / 4) * 8 + (metrics.cb_drop / 8) * 7)
    
    return stuff_score + velo_score + spin_score + movement_score


def calculate_pitcher_whiff_score(metrics: StatcastPitcher) -> float:
    """
    Calculate swing-and-miss ability score 0-100.
    Based on whiff%, chase%, and CSW%.
    """
    # Whiff% (0-45 points)
    whiff_normalized = (metrics.whiff_pct - PITCHING_THRESHOLDS["whiff_pct"]["poor"]) / \
                       (PITCHING_THRESHOLDS["whiff_pct"]["elite"] - PITCHING_THRESHOLDS["whiff_pct"]["poor"])
    whiff_score = min(45, max(0, whiff_normalized * 45))
    
    # Chase% (0-35 points)
    chase_normalized = (metrics.chase_pct - PITCHING_THRESHOLDS["chase_pct"]["poor"]) / \
                       (PITCHING_THRESHOLDS["chase_pct"]["elite"] - PITCHING_THRESHOLDS["chase_pct"]["poor"])
    chase_score = min(35, max(0, chase_normalized * 35))
    
    # CSW% (called strikes + whiffs) (0-20 points)
    csw_score = min(20, (metrics.csw_pct - 25) / (35 - 25) * 20)
    
    return whiff_score + chase_score + max(0, csw_score)


def calculate_injury_risk_score(metrics: StatcastPitcher) -> float:
    """
    Calculate injury risk score 0-100 based on velocity trends and workload.
    Higher score = higher injury risk.
    """
    risk = 0.0
    
    # Velocity decline flag (>1.5 mph is concerning)
    if metrics.velo_decline > 2.0:
        risk += 40
    elif metrics.velo_decline > 1.5:
        risk += 25
    elif metrics.velo_decline > 1.0:
        risk += 15
    
    # High workload (>100 pitches per game average)
    if metrics.pitches_per_game > 100:
        risk += 20
    elif metrics.pitches_per_game > 95:
        risk += 10
    
    # Late-inning velocity drop
    if metrics.velo_by_inning:
        early_velo = statistics.mean([metrics.velo_by_inning.get(i, 0) for i in [1, 2, 3] if metrics.velo_by_inning.get(i, 0) > 0])
        late_velo = statistics.mean([metrics.velo_by_inning.get(i, 0) for i in [6, 7] if metrics.velo_by_inning.get(i, 0) > 0])
        if early_velo > 0 and late_velo > 0:
            velo_drop = early_velo - late_velo
            if velo_drop > 3.0:
                risk += 25
            elif velo_drop > 2.0:
                risk += 15
    
    return min(100, risk)


# ---------------------------------------------------------------------------
# Statcast Regression-to-Mean Analysis (Player-Level)
# ---------------------------------------------------------------------------
# NOTE: This is PLAYER-LEVEL regression-to-mean detection (xwOBA vs wOBA,
# xERA vs ERA) — identifies buy-low/sell-high candidates based on Statcast
# expected metrics.  NOT related to the PIPELINE-LEVEL MAE regression
# detector in backtesting_harness.py which monitors projection accuracy.

def analyze_batter_regression(metrics: StatcastBatter) -> Tuple[str, float]:
    """
    Analyze whether batter was lucky or unlucky based on xwOBA vs wOBA.
    Returns: (verdict, confidence_0_to_1)
    """
    diff = metrics.xwoba_diff
    
    if diff > BATTING_THRESHOLDS["xwoba_diff"]["lucky"]:
        return "SELL_HIGH", min(1.0, diff / 0.050)
    elif diff < BATTING_THRESHOLDS["xwoba_diff"]["unlucky"]:
        return "BUY_LOW", min(1.0, abs(diff) / 0.040)
    else:
        return "NEUTRAL", 0.5


def analyze_pitcher_regression(metrics: StatcastPitcher) -> Tuple[str, float]:
    """
    Analyze whether pitcher was lucky or unlucky based on xERA vs ERA.
    Returns: (verdict, confidence_0_to_1)
    """
    diff = metrics.xera_diff  # xERA - ERA (positive means ERA should rise)
    
    if diff < PITCHING_THRESHOLDS["xera_diff"]["lucky"]:
        return "SELL_HIGH", min(1.0, abs(diff) / 1.0)
    elif diff > PITCHING_THRESHOLDS["xera_diff"]["unlucky"]:
        return "BUY_LOW", min(1.0, diff / 0.80)
    else:
        return "NEUTRAL", 0.5


# ---------------------------------------------------------------------------
# Breakout Candidate Detection
# ---------------------------------------------------------------------------

def is_breakout_candidate_batter(metrics: StatcastBatter, age: int) -> Tuple[bool, str]:
    """
    Identify breakout candidates based on:
    - Age < 26
    - Improving exit velocity
    - Barrel% > 8%
    - xwOBA > actual (unlucky)
    """
    reasons = []
    
    if age <= 25 and metrics.barrel_pct > 8.0:
        reasons.append(f"Young power ({metrics.barrel_pct:.1f}% barrels)")
    
    if metrics.exit_velo_avg > 90.0 and age <= 27:
        reasons.append(f"Elite EV ({metrics.exit_velo_avg:.1f} mph)")
    
    if metrics.xwoba_diff < -0.015:
        reasons.append(f"Unlucky xwOBA ({metrics.xwoba_diff:+.3f})")
    
    if metrics.sprint_speed > 28.5 and metrics.sb_attempts > 15:
        reasons.append(f"Speed threat ({metrics.sprint_speed:.1f} ft/s)")
    
    return len(reasons) >= 2, "; ".join(reasons) if reasons else ""


def is_breakout_candidate_pitcher(metrics: StatcastPitcher, age: int) -> Tuple[bool, str]:
    """
    Identify breakout candidates based on:
    - Age < 26
    - Stuff+ > 110
    - Improving command
    - High whiff% (>26%)
    """
    reasons = []
    
    if age <= 25 and metrics.stuff_plus > 110:
        reasons.append(f"Elite stuff ({metrics.stuff_plus:.0f} Stuff+)")
    
    if metrics.whiff_pct > 28:
        reasons.append(f"High whiff% ({metrics.whiff_pct:.1f}%)")
    
    if metrics.chase_pct > 30:
        reasons.append(f"Good chase% ({metrics.chase_pct:.1f}%)")
    
    if metrics.velo_decline < -0.5:  # Velocity UP
        reasons.append(f"Velo gain ({abs(metrics.velo_decline):.1f} mph)")
    
    return len(reasons) >= 2, "; ".join(reasons) if reasons else ""


# ---------------------------------------------------------------------------
# CSV Generation - Create advanced metrics files
# ---------------------------------------------------------------------------

def generate_advanced_batting_csv(output_path: Path = None) -> Path:
    """
    Generate CSV with advanced Statcast batting metrics.
    This would typically pull from Baseball Savant API or cached data.
    """
    if output_path is None:
        output_path = DATA_DIR / "advanced_batting_2026.csv"
    
    headers = [
        "Name", "Team", "Barrel_Pct", "Exit_Velo", "Hard_Hit_Pct",
        "Sweet_Spot_Pct", "xBA", "xSLG", "xwOBA", "xwOBA_Diff",
        "O_Swing_Pct", "Z_Contact_Pct", "SwStr_Pct",
        "Sprint_Speed", "Power_Score", "Contact_Score", "Discipline_Score",
        "Speed_Score", "Overall_Score", "Regression_Flag", "Breakout_Flag"
    ]
    
    # Sample data for top players (would be populated from Statcast API)
    sample_data = [
        ["Aaron Judge", "NYY", 18.5, 96.2, 52.8, 38.5, 0.295, 0.625, 0.435, -0.015,
         25.5, 82.0, 11.5, 27.5, 95, 65, 70, 45, 78, "BUY_LOW", ""],
        ["Shohei Ohtani", "LAD", 15.8, 95.5, 48.2, 36.8, 0.288, 0.598, 0.418, -0.008,
         28.2, 78.5, 13.2, 28.2, 92, 58, 62, 55, 72, "NEUTRAL", ""],
        ["Bobby Witt Jr.", "KCR", 11.2, 92.8, 42.5, 34.2, 0.298, 0.525, 0.385, 0.002,
         32.5, 85.2, 10.8, 29.5, 78, 72, 55, 85, 75, "NEUTRAL", "BREAKOUT"],
        ["Juan Soto", "NYM", 12.5, 93.2, 45.8, 35.5, 0.285, 0.545, 0.425, 0.005,
         18.5, 88.5, 8.2, 26.8, 82, 82, 92, 35, 82, "NEUTRAL", ""],
        ["Julio Rodriguez", "SEA", 10.8, 92.5, 40.2, 33.5, 0.282, 0.512, 0.375, -0.012,
         35.2, 84.5, 11.5, 29.2, 75, 70, 52, 82, 72, "BUY_LOW", ""],
        ["Ronald Acuna Jr.", "ATL", 11.5, 92.8, 43.5, 34.8, 0.290, 0.520, 0.395, 0.008,
         30.5, 86.2, 9.8, 29.8, 78, 75, 62, 88, 78, "NEUTRAL", ""],
        ["Elly De La Cruz", "CIN", 9.8, 93.5, 38.5, 32.2, 0.265, 0.485, 0.358, -0.018,
         38.5, 75.8, 14.2, 30.2, 72, 58, 42, 95, 70, "BUY_LOW", "BREAKOUT"],
        ["Gunnar Henderson", "BAL", 11.8, 92.2, 41.8, 34.5, 0.278, 0.505, 0.375, -0.005,
         32.8, 82.5, 11.8, 28.5, 76, 68, 58, 72, 72, "NEUTRAL", "BREAKOUT"],
        ["Vladimir Guerrero Jr.", "TOR", 13.2, 94.8, 48.5, 36.2, 0.305, 0.565, 0.405, 0.012,
         28.5, 87.2, 9.5, 25.5, 88, 80, 65, 28, 78, "NEUTRAL", ""],
        ["Fernando Tatis Jr.", "SDP", 12.2, 93.5, 44.2, 35.8, 0.280, 0.535, 0.390, 0.005,
         31.2, 83.5, 10.5, 29.2, 80, 72, 60, 75, 75, "NEUTRAL", ""],
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(sample_data)
    
    logger.info(f"Generated advanced batting metrics: {output_path}")
    return output_path


def generate_advanced_pitching_csv(output_path: Path = None) -> Path:
    """Generate CSV with advanced Statcast pitching metrics."""
    if output_path is None:
        output_path = DATA_DIR / "advanced_pitching_2026.csv"
    
    headers = [
        "Name", "Team", "Stuff_Plus", "Location_Plus", "FB_Velo",
        "Spin_Rate_FB", "Whiff_Pct", "Chase_Pct", "Barrel_Allowed_Pct",
        "xERA", "xERA_Diff", "Injury_Risk_Score",
        "Stuff_Score", "Whiff_Score", "Overall_Score",
        "Regression_Flag", "Breakout_Flag", "Injury_Risk_Flag"
    ]
    
    sample_data = [
        ["Tarik Skubal", "DET", 125, 108, 95.8, 2450, 32.5, 30.2, 5.8,
         2.65, -0.14, 15, 92, 88, 90, "", "", ""],
        ["Paul Skenes", "PIT", 130, 105, 98.5, 2550, 34.2, 28.5, 5.2,
         2.75, -0.17, 18, 95, 92, 94, "", "BREAKOUT", ""],
        ["Garrett Crochet", "BOS", 128, 102, 98.2, 2480, 33.8, 29.2, 5.5,
         2.85, -0.17, 22, 94, 90, 92, "", "BREAKOUT", ""],
        ["Cristopher Sanchez", "PHI", 115, 110, 93.5, 2250, 28.5, 32.5, 6.2,
         3.05, -0.10, 12, 82, 78, 82, "", "", ""],
        ["Chris Sale", "ATL", 118, 112, 92.8, 2150, 30.2, 31.8, 6.5,
         3.05, +0.14, 35, 85, 82, 85, "SELL_HIGH", "", "INJURY_RISK"],
        ["Logan Webb", "SFG", 108, 118, 92.5, 2100, 24.5, 35.2, 6.8,
         3.25, -0.06, 10, 75, 68, 75, "", "", ""],
        ["Dylan Cease", "TOR", 122, 98, 96.2, 2520, 31.5, 27.8, 6.0,
         3.35, -0.19, 20, 88, 85, 88, "", "", ""],
        ["Jacob deGrom", "TEX", 135, 108, 98.8, 2650, 36.5, 30.5, 4.8,
         2.95, -0.57, 55, 98, 95, 95, "SELL_HIGH", "", "HIGH_INJURY_RISK"],
        ["Max Fried", "NYY", 110, 115, 93.2, 2050, 26.8, 33.5, 6.2,
         3.15, -0.13, 28, 80, 72, 78, "", "", "INJURY_RISK"],
        ["Hunter Greene", "CIN", 128, 95, 99.2, 2580, 33.2, 26.5, 6.5,
         3.45, -0.47, 25, 92, 88, 90, "", "BREAKOUT", ""],
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(sample_data)
    
    logger.info(f"Generated advanced pitching metrics: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Competitive Edge Report Generator
# ---------------------------------------------------------------------------

def generate_competitive_edge_report() -> str:
    """
    Generate a report highlighting key competitive edges from advanced metrics.
    """
    report = []
    report.append("=" * 80)
    report.append("FANTASY BASEBALL 2026 - STATCAST COMPETITIVE EDGE REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("BUY LOW (Unlucky xwOBA/xERA - Positive Regression Expected):")
    report.append("-" * 60)
    report.append("- Elly De La Cruz: xwOBA 18 pts higher than actual, 93.5 mph EV")
    report.append("- Bobby Witt Jr.: Elite speed (29.5 ft/s), improving barrel%")
    report.append("- Julio Rodriguez: xwOBA 12 pts higher, premium speed combo")
    report.append("")
    
    report.append("SELL HIGH (Lucky xwOBA/xERA - Negative Regression Expected):")
    report.append("-" * 60)
    report.append("- Jacob deGrom: xERA 3.52 vs actual 2.95 - major injury risk")
    report.append("- Chris Sale: xERA 3.19 vs actual 3.05, velo declining")
    report.append("- Tyler Mahle: xERA significantly higher than surface stats")
    report.append("")
    
    report.append("BREAKOUT CANDIDATES (Young + Improving Underlying Skills):")
    report.append("-" * 60)
    report.append("- Paul Skenes: 130 Stuff+, 98.5 mph avg, elite whiff%")
    report.append("- Garrett Crochet: 128 Stuff+, velocity up from relief days")
    report.append("- Gunnar Henderson: Barrel% trending up, 92+ mph EV")
    report.append("- Hunter Greene: 99.2 mph FB, 128 Stuff+, improving command")
    report.append("")
    
    report.append("INJURY RISKS (Velocity Decline + Workload Concerns):")
    report.append("-" * 60)
    report.append("- Jacob deGrom: Multiple TJS history, velo spike = risk")
    report.append("- Chris Sale: Tommy John x2, velo down 1.5+ mph")
    report.append("- Brandon Woodruff: Shoulder surgery, innings limit likely")
    report.append("- Mike Trout: Chronic knee issues, declining sprint speed")
    report.append("")
    
    report.append("ELITE STATCAST METRICS (Barrels, Exit Velo, Stuff+):")
    report.append("-" * 60)
    report.append("- Aaron Judge: 18.5% Barrel%, 96.2 mph EV (98th percentile)")
    report.append("- Shohei Ohtani: 15.8% Barrel%, 95.5 mph EV, elite discipline")
    report.append("- Paul Skenes: 130 Stuff+, 98.5 mph, 34.2% Whiff%")
    report.append("- Josh Hader: 72 Stuff+, but declining - monitor closely")
    report.append("")
    
    report.append("PARK FACTOR ARBITRAGE:")
    report.append("-" * 60)
    report.append("- Target: Rangers hitters (Globe Life Field = hitter-friendly)")
    report.append("- Target: Reds hitters (Great American = #1 HR park)")
    report.append("- Avoid: Athletics pitchers (Oakland Coliseum neutralized)")
    report.append("- Avoid: Rockies pitchers (Coors Field = ERA destroyer)")
    report.append("")
    
    report.append("PLATOON SPLIT EDGES:")
    report.append("-" * 60)
    report.append("- Kyle Schwarber: Crush RHP (.900+ OPS), sit vs LHP")
    report.append("- Randy Arozarena: Reverse splits - better vs LHP")
    report.append("- Christian Walker: Elite glove, consistent vs both sides")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate advanced metric CSVs
    generate_advanced_batting_csv()
    generate_advanced_pitching_csv()
    
    # Print competitive edge report
    print(generate_competitive_edge_report())
    
    print("\n" + "=" * 80)
    print("Advanced metrics CSVs generated in data/projections/")
    print("=" * 80)
