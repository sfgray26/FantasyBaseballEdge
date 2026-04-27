"""
Ballpark Factors + Player Risk Flags

Ballpark factors affect pitcher ERA/WHIP projections and batter HR/R projections.
Used by the recommender to adjust raw z-scores for park context.

Source: 5-year park factor averages (ESPN/Baseball Reference consensus).
Scale: 100 = neutral, >100 = hitter friendly, <100 = pitcher friendly.

Risk flags: age, injury history, role uncertainty — applied as draft penalties.
"""

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Park factors (2026 estimates based on 3-yr rolling average)
# run_factor: affects R, H, AVG, OPS for batters
# hr_factor:  affects HR for batters; HR allowed for pitchers
# era_factor: affects ERA/WHIP for pitchers (inverse of batter factor)
# ---------------------------------------------------------------------------

PARK_FACTORS: dict[str, dict] = {
    # Extreme hitter parks
    "COL": {"run": 1.38, "hr": 1.30, "era": 1.28},   # Coors — massive
    "CIN": {"run": 1.10, "hr": 1.14, "era": 1.10},   # GABP
    "PHI": {"run": 1.06, "hr": 1.10, "era": 1.06},   # Citizens Bank
    "TEX": {"run": 1.07, "hr": 1.08, "era": 1.06},   # Globe Life
    "TOR": {"run": 1.05, "hr": 1.06, "era": 1.04},   # Rogers Centre
    "BOS": {"run": 1.04, "hr": 1.00, "era": 1.03},   # Fenway (quirky)
    "NYY": {"run": 1.04, "hr": 1.12, "era": 1.04},   # Yankee Stadium (short porch)
    "ATL": {"run": 1.04, "hr": 1.06, "era": 1.03},   # Truist Park
    "MIL": {"run": 1.03, "hr": 1.04, "era": 1.02},   # American Family Field
    "ARI": {"run": 1.03, "hr": 1.04, "era": 1.02},   # Chase Field (roof)
    "CLE": {"run": 1.02, "hr": 1.02, "era": 1.01},   # Progressive Field
    "MIN": {"run": 1.02, "hr": 1.06, "era": 1.02},   # Target Field
    "MIA": {"run": 1.00, "hr": 0.98, "era": 1.00},   # LoanDepot Park
    "WSH": {"run": 1.00, "hr": 1.00, "era": 1.00},   # Nationals Park
    "STL": {"run": 1.00, "hr": 0.98, "era": 1.00},   # Busch Stadium
    "LAA": {"run": 0.99, "hr": 1.00, "era": 0.99},   # Angel Stadium
    "KC":  {"run": 0.99, "hr": 0.96, "era": 0.99},   # Kauffman
    "CHC": {"run": 0.99, "hr": 1.00, "era": 0.99},   # Wrigley (wind dependent)
    "PIT": {"run": 0.99, "hr": 0.98, "era": 0.99},   # PNC Park
    "DET": {"run": 0.98, "hr": 0.96, "era": 0.98},   # Comerica
    "CHW": {"run": 0.97, "hr": 0.98, "era": 0.98},   # Guaranteed Rate
    "CWS": {"run": 0.97, "hr": 0.98, "era": 0.98},   # same
    "NYM": {"run": 0.97, "hr": 0.96, "era": 0.97},   # Citi Field
    "TB":  {"run": 0.97, "hr": 0.96, "era": 0.97},   # Tropicana
    "OAK": {"run": 0.96, "hr": 0.94, "era": 0.96},   # Oakland Coliseum
    "HOU": {"run": 0.97, "hr": 0.98, "era": 0.97},   # Minute Maid
    # Extreme pitcher parks
    "SF":  {"run": 0.94, "hr": 0.90, "era": 0.94},   # Oracle Park — marine layer
    "SEA": {"run": 0.93, "hr": 0.90, "era": 0.93},   # T-Mobile Park
    "LAD": {"run": 0.95, "hr": 0.94, "era": 0.95},   # Dodger Stadium
    "SD":  {"run": 0.94, "hr": 0.92, "era": 0.94},   # Petco Park
    "BAL": {"run": 0.98, "hr": 1.00, "era": 0.98},   # Camden Yards
    "FA":  {"run": 1.00, "hr": 1.00, "era": 1.00},   # Free agent — neutral
    "free":{"run": 1.00, "hr": 1.00, "era": 1.00},
}

def get_park_factor(team: str, factor: str = "run", _db_session=None) -> float:
    """
    Return park factor for a team and factor type.

    Resolution order:
    1. ParkFactor table (canonical persisted context)
    2. PARK_FACTORS constant (hardcoded fallback)
    3. 1.0 neutral default

    Args:
        team: Team code (e.g., "COL", "BOS")
        factor: One of "run", "hr", "era" (defaults to "run")
        _db_session: Optional session for DB queries (for testing)

    Returns:
        Park factor value (1.0 = neutral)
    """
    from backend.models import ParkFactor, SessionLocal

    # Map ballpark_factors naming to ParkFactor column names
    factor_column_map = {"run": "run_factor", "hr": "hr_factor", "era": "era_factor"}

    column_name = factor_column_map.get(factor)
    if not column_name:
        return 1.0

    # Try DB first
    db = _db_session or SessionLocal()
    try:
        db_factor = db.query(ParkFactor).filter_by(park_name=team).first()
        if db_factor:
            return getattr(db_factor, column_name, 1.0)
    finally:
        if not _db_session:
            db.close()

    # Fall back to hardcoded constant
    return PARK_FACTORS.get(team, {}).get(factor, 1.0)


def park_adjusted_era(raw_era: float, team: str) -> float:
    """Adjust ERA for ballpark context."""
    pf = get_park_factor(team, "era")
    return raw_era / pf if pf > 0 else raw_era


def park_adjusted_hr(raw_hr: int, team: str, is_batter: bool = True) -> float:
    """Adjust HR projection for park. Batters: multiply by hr factor."""
    pf = get_park_factor(team, "hr")
    if is_batter:
        return raw_hr * pf
    else:  # pitcher — more HRs allowed in HR parks
        return raw_hr * pf


# ---------------------------------------------------------------------------
# Player risk flags
# ---------------------------------------------------------------------------

@dataclass
class RiskProfile:
    player_name: str
    age: int
    injury_risk: str          # "low", "medium", "high", "extreme"
    role_certainty: str       # "locked", "likely", "uncertain", "speculative"
    health_history: str       # brief note
    draft_penalty: float      # 0.0-1.0, subtracted from z_score in risk-adjusted mode
    notes: str = ""


# Risk profiles for players with meaningful flags
# draft_penalty: fraction of z_score to subtract (0.1 = 10% discount)
RISK_PROFILES: dict[str, RiskProfile] = {
    "aaron_judge": RiskProfile(
        "Aaron Judge", 34, "medium", "locked",
        "Hamstring/oblique prone — missed 50+ games in 2023",
        0.12, "Elite talent, health discount justified"
    ),
    "mookie_betts": RiskProfile(
        "Mookie Betts", 33, "low", "locked",
        "Broken hand 2024 but historically durable",
        0.06, "Minimal concern"
    ),
    "freddie_freeman": RiskProfile(
        "Freddie Freeman", 36, "medium", "locked",
        "Age 36 in 2026, ankle issue in 2024 WS",
        0.12, "Age curve starts here; monitor spring"
    ),
    "shohei_ohtani": RiskProfile(
        "Shohei Ohtani", 32, "low", "locked",
        "Post-TJS; batting only — no pitching in 2026",
        0.05, "Batting floor is enormous"
    ),
    "corey_seager": RiskProfile(
        "Corey Seager", 32, "medium", "locked",
        "Tommy John 2023; historically injury-prone",
        0.15, "Real injury history, discount appropriately"
    ),
    "spencer_strider": RiskProfile(
        "Spencer Strider", 27, "high", "locked",
        "TJS in April 2024 — return timeline aggressive for 2026",
        0.22, "TJS returns: velocity/stuff uncertainty is real"
    ),
    "blake_snell": RiskProfile(
        "Blake Snell", 33, "high", "likely",
        "Groin, FA instability — missed significant time",
        0.20, "Upside real but never 200 IP"
    ),
    "tyler_glasnow": RiskProfile(
        "Tyler Glasnow", 32, "high", "locked",
        "Tommy John history, blister issues, never 180+ IP",
        0.22, "Elite when healthy; floor is very low"
    ),
    "carlos_rodon": RiskProfile(
        "Carlos Rodon", 33, "high", "locked",
        "Flexor mass surgery 2024; chronic arm issues",
        0.25, "High-upside, high-risk; late-round gamble only"
    ),
    "nathan_eovaldi": RiskProfile(
        "Nathan Eovaldi", 36, "high", "locked",
        "Multiple arm surgeries; age 36",
        0.28, "Streaming only — too risky to roster"
    ),
    "mike_trout": RiskProfile(
        "Mike Trout", 35, "extreme", "uncertain",
        "Meniscus 2023, knee surgery — barely played 2023-25",
        0.40, "Avoid unless dramatic positional scarcity"
    ),
    "kodai_senga": RiskProfile(
        "Kodai Senga", 33, "high", "likely",
        "Shoulder capsule strain; missed all of 2024",
        0.28, "12-15 starts if healthy — upside real but health a ??"
    ),
    "shane_bieber": RiskProfile(
        "Shane Bieber", 31, "medium", "likely",
        "UCL repair 2024 — return full 2026",
        0.15, "If healthy: top-10 SP upside"
    ),
    "eury_perez": RiskProfile(
        "Eury Perez", 23, "high", "locked",
        "TJS 2024 — 23 years old, high ceiling if arm holds",
        0.20, "Best case: 165 elite IP. Worst: <100"
    ),
    "justin_verlander": RiskProfile(
        "Justin Verlander", 43, "extreme", "uncertain",
        "Age 43 in 2026 — likely final season if playing",
        0.35, "Only draft if confirmed starting"
    ),
    "aroldis_chapman": RiskProfile(
        "Aroldis Chapman", 38, "high", "uncertain",
        "Setup/closer committee role uncertain; velocity declining",
        0.25, "NSV upside only if gets closer role"
    ),
    "randy_arozarena": RiskProfile(
        "Randy Arozarena", 31, "low", "locked",
        "Healthy; Seattle park suppresses production",
        0.08, "SEA park adjustment hurts power/runs"
    ),
    "elly_de_la_cruz": RiskProfile(
        "Elly De La Cruz", 23, "low", "locked",
        "Youth upside; K rate (30%+) is the only drag",
        0.05, "K-heavy but speed/power combo is elite"
    ),
}


def get_risk_profile(player_id: str) -> Optional[RiskProfile]:
    """Return risk profile if available, else None."""
    return RISK_PROFILES.get(player_id)


def risk_adjusted_zscore(player: dict, apply_park: bool = True,
                          apply_health: bool = True) -> float:
    """
    Return z_score adjusted for park factor and health risk.

    Park adjustment: scale z_score by park factor (batters: run_factor,
    pitchers: 1/era_factor for ERA-based value).
    Health adjustment: subtract draft_penalty × |z_score|.
    """
    z = player.get("z_score", 0.0)
    team = player.get("team", "FA")
    player_id = player.get("id", "")
    ptype = player.get("type", "batter")

    if apply_park:
        if ptype == "batter":
            pf = get_park_factor(team, "run")
            # Neutral park = 1.0, Coors = 1.38 → Coors batter z boosted 19%
            # (using sqrt to avoid over-adjusting)
            park_mult = ((pf - 1.0) * 0.5) + 1.0
        else:
            pf = get_park_factor(team, "era")
            # Pitcher in SF (0.94 era factor) → z boosted slightly
            park_mult = ((1.0 / pf - 1.0) * 0.5) + 1.0
        z = z * park_mult

    if apply_health:
        profile = get_risk_profile(player_id)
        if profile:
            # Subtract penalty × absolute z_score (penalty shrinks value toward 0)
            z = z - (profile.draft_penalty * abs(z))

    return round(z, 3)


def annotate_board(board: list[dict]) -> list[dict]:
    """
    Add park-adjusted z_score and risk flags to each player in-place.
    Returns the same list (mutates each dict).
    """
    for p in board:
        p["z_park_adjusted"] = risk_adjusted_zscore(p, apply_park=True, apply_health=False)
        p["z_risk_adjusted"] = risk_adjusted_zscore(p, apply_park=True, apply_health=True)
        profile = get_risk_profile(p.get("id", ""))
        p["injury_risk"] = profile.injury_risk if profile else "low"
        p["risk_notes"] = profile.notes if profile else ""
        p["age"] = profile.age if profile else 0
    return board


# ---------------------------------------------------------------------------
# Park factor display helpers
# ---------------------------------------------------------------------------

def park_factor_tier(team: str) -> str:
    """Return a readable park tier label."""
    run = get_park_factor(team, "run")
    if run >= 1.15:
        return "🏟️ EXTREME hitter"
    elif run >= 1.05:
        return "🔴 Hitter friendly"
    elif run >= 0.98:
        return "⚪ Neutral"
    elif run >= 0.94:
        return "🔵 Pitcher friendly"
    else:
        return "❄️ EXTREME pitcher"


if __name__ == "__main__":
    print("Park factor tiers:")
    teams = sorted(PARK_FACTORS.keys())
    for t in teams:
        print(f"  {t:4s} run={get_park_factor(t, 'run'):.2f} hr={get_park_factor(t, 'hr'):.2f}  {park_factor_tier(t)}")
