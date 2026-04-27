"""
P16 -- Rest-of-Season Monte Carlo Simulation Engine.

Pure-computation module (no DB imports, no side effects).
All imports are at module top level -- no imports inside functions.

Algorithm:
  - Basis: 14-day decay-weighted rolling window (player_rolling_stats)
  - Simulations: N=1000 per player
  - Per-game draw: max(0, Normal(rate, rate * CV)) where CV=0.35
  - RNG: random.Random(seed) instance -- thread-safe, no global state

Output: SimulationResult dataclass with P10/P25/P50/P75/P90 percentiles
        for each counting and rate stat, plus composite risk metrics.

ADR-004: Never import betting_model or analysis.
"""

import random
from dataclasses import dataclass
from datetime import date
from typing import Optional

CV = 0.35                       # coefficient of variation per simulated game
N_SIMULATIONS = 1000
REMAINING_GAMES_DEFAULT = 130   # approximate remaining MLB games mid-April 2026


# ---------------------------------------------------------------------------
# SimulationResult dataclass (pure-computation output -- NOT the ORM model)
# In daily_ingestion.py import the ORM as:
#   from backend.models import SimulationResult as SimulationResultORM
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    bdl_player_id: int
    as_of_date: date
    window_days: int            # always 14 -- 14d window is the simulation basis
    remaining_games: int
    n_simulations: int          # always 1000
    player_type: str            # "hitter" | "pitcher" | "two_way" | "unknown"

    # Hitter stat percentiles (None for pure pitchers / unknown)
    proj_hr_p10: Optional[float] = None
    proj_hr_p25: Optional[float] = None
    proj_hr_p50: Optional[float] = None
    proj_hr_p75: Optional[float] = None
    proj_hr_p90: Optional[float] = None

    proj_rbi_p10: Optional[float] = None
    proj_rbi_p25: Optional[float] = None
    proj_rbi_p50: Optional[float] = None
    proj_rbi_p75: Optional[float] = None
    proj_rbi_p90: Optional[float] = None

    proj_sb_p10: Optional[float] = None
    proj_sb_p25: Optional[float] = None
    proj_sb_p50: Optional[float] = None
    proj_sb_p75: Optional[float] = None
    proj_sb_p90: Optional[float] = None

    proj_avg_p10: Optional[float] = None
    proj_avg_p25: Optional[float] = None
    proj_avg_p50: Optional[float] = None
    proj_avg_p75: Optional[float] = None
    proj_avg_p90: Optional[float] = None

    # Pitcher stat percentiles (None for pure hitters / unknown)
    proj_k_p10: Optional[float] = None
    proj_k_p25: Optional[float] = None
    proj_k_p50: Optional[float] = None
    proj_k_p75: Optional[float] = None
    proj_k_p90: Optional[float] = None

    proj_era_p10: Optional[float] = None
    proj_era_p25: Optional[float] = None
    proj_era_p50: Optional[float] = None
    proj_era_p75: Optional[float] = None
    proj_era_p90: Optional[float] = None

    proj_whip_p10: Optional[float] = None
    proj_whip_p25: Optional[float] = None
    proj_whip_p50: Optional[float] = None
    proj_whip_p75: Optional[float] = None
    proj_whip_p90: Optional[float] = None

    # Risk metrics (populated when composite z-scores are computable)
    composite_variance: Optional[float] = None
    downside_p25: Optional[float] = None    # P25 of per-simulation composite scores
    upside_p75: Optional[float] = None      # P75 of per-simulation composite scores
    prob_above_median: Optional[float] = None  # fraction of runs above P50 threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentiles(values: list) -> tuple:
    """
    Return (P10, P25, P50, P75, P90) from a list of floats.

    Uses 0-indexed position: P10 = values[int(0.10 * n)], etc.
    Empty list returns all 0.0.
    """
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(values)
    n = len(s)

    def _pct(p):
        idx = max(0, min(n - 1, int(p * n)))
        return s[idx]

    return (_pct(0.10), _pct(0.25), _pct(0.50), _pct(0.75), _pct(0.90))


def _sample_positive(rng, mu: float, sigma: float) -> float:
    """
    Draw from Normal(mu, sigma), floor at 0.

    Returns 0.0 immediately if mu <= 0 (avoids negative sigma).
    Uses the caller-provided rng instance for thread safety.
    """
    if mu <= 0:
        return 0.0
    return max(0.0, rng.gauss(mu, sigma))


def _draw_games(rng, rate: float, n_games: int) -> float:
    """
    Sum n_games independent draws from Normal(rate, rate*CV), each floored at 0.

    This is the inner loop shared by all counting stats.
    Returns the season total for one simulation run.
    """
    sigma = rate * CV
    total = 0.0
    for _ in range(n_games):
        total += _sample_positive(rng, rate, sigma)
    return total


def _compute_composite_risk(
    sim_composites: list,
) -> tuple:
    """
    Given 1000 composite scores (one per simulation run), return
    (composite_variance, downside_p25, upside_p75, prob_above_median).

    prob_above_median = fraction of runs exceeding the P50 composite value.
    """
    if not sim_composites:
        return (None, None, None, None)

    n = len(sim_composites)
    mean_c = sum(sim_composites) / n
    variance = sum((x - mean_c) ** 2 for x in sim_composites) / n

    p10, p25, p50, p75, p90 = _percentiles(sim_composites)
    prob_above = sum(1 for x in sim_composites if x > p50) / n

    return (variance, p25, p75, prob_above)


# ---------------------------------------------------------------------------
# Main simulation entry points
# ---------------------------------------------------------------------------

def simulate_player(
    rolling_row,
    remaining_games: int = REMAINING_GAMES_DEFAULT,
    n_simulations: int = N_SIMULATIONS,
    seed: Optional[int] = None,
    league_means: Optional[dict] = None,
    league_stds: Optional[dict] = None,
) -> SimulationResult:
    """
    Run Monte Carlo Rest-of-Season simulation for one player.

    Parameters
    ----------
    rolling_row : PlayerRollingStats ORM row (window_days=14)
        Must have bdl_player_id, as_of_date, games_in_window set.
        Batting fields present -> hitter/two_way path.
        Pitching fields present -> pitcher/two_way path.

    remaining_games : int
        Number of games remaining in the season (default 130 for mid-April 2026).

    n_simulations : int
        Number of Monte Carlo runs (default 1000).

    seed : int or None
        If provided, the RNG is seeded for reproducibility. Use seed=42 in tests.

    league_means : dict or None
        {"hr": float, "rbi": float, "sb": float, "avg": float,
         "k": float, "era": float, "whip": float}
        Required to compute composite risk metrics. If None, risk fields are None.

    league_stds : dict or None
        Same keys as league_means. Required alongside league_means.

    Returns
    -------
    SimulationResult dataclass with all applicable percentile fields populated.
    """
    rng = random.Random(seed)
    # M3 fix: use decay-weighted game count for consistent rate derivation.
    # Fall back to raw games_in_window for rows computed before w_games was added.
    _wg = getattr(rolling_row, 'w_games', None)
    g = _wg if isinstance(_wg, (int, float)) and _wg > 0 else (rolling_row.games_in_window or 1)

    has_batting = rolling_row.w_ab is not None
    has_pitching = rolling_row.w_ip is not None

    if has_batting and has_pitching:
        player_type = "two_way"
    elif has_batting:
        player_type = "hitter"
    elif has_pitching:
        player_type = "pitcher"
    else:
        player_type = "unknown"

    result = SimulationResult(
        bdl_player_id=rolling_row.bdl_player_id,
        as_of_date=rolling_row.as_of_date,
        window_days=14,
        remaining_games=remaining_games,
        n_simulations=n_simulations,
        player_type=player_type,
    )

    if player_type == "unknown":
        return result

    # ------------------------------------------------------------------
    # Batting simulation
    # ------------------------------------------------------------------
    sim_composites = []  # populated later if league parameters available

    if has_batting:
        hr_rate  = (rolling_row.w_home_runs     or 0.0) / g
        rbi_rate = (rolling_row.w_rbi           or 0.0) / g
        sb_rate  = (rolling_row.w_stolen_bases  or 0.0) / g
        ab_rate  = (rolling_row.w_ab            or 0.0) / g
        hit_rate = (rolling_row.w_hits          or 0.0) / g

        sim_hr  = []
        sim_rbi = []
        sim_sb  = []
        sim_avg = []

        for _ in range(n_simulations):
            total_hr  = _draw_games(rng, hr_rate,  remaining_games)
            total_rbi = _draw_games(rng, rbi_rate, remaining_games)
            total_sb  = _draw_games(rng, sb_rate,  remaining_games)
            total_ab  = _draw_games(rng, ab_rate,  remaining_games)
            total_hit = _draw_games(rng, hit_rate, remaining_games)

            avg = total_hit / total_ab if total_ab > 0 else 0.0

            sim_hr.append(total_hr)
            sim_rbi.append(total_rbi)
            sim_sb.append(total_sb)
            sim_avg.append(avg)

        (
            result.proj_hr_p10,
            result.proj_hr_p25,
            result.proj_hr_p50,
            result.proj_hr_p75,
            result.proj_hr_p90,
        ) = _percentiles(sim_hr)

        (
            result.proj_rbi_p10,
            result.proj_rbi_p25,
            result.proj_rbi_p50,
            result.proj_rbi_p75,
            result.proj_rbi_p90,
        ) = _percentiles(sim_rbi)

        (
            result.proj_sb_p10,
            result.proj_sb_p25,
            result.proj_sb_p50,
            result.proj_sb_p75,
            result.proj_sb_p90,
        ) = _percentiles(sim_sb)

        (
            result.proj_avg_p10,
            result.proj_avg_p25,
            result.proj_avg_p50,
            result.proj_avg_p75,
            result.proj_avg_p90,
        ) = _percentiles(sim_avg)

        # Build per-run composite z-scores for batting if league params available
        if (
            league_means is not None
            and league_stds is not None
            and league_stds.get("hr", 0) > 0
        ):
            hr_mean  = league_means.get("hr",  0.0)
            hr_std   = league_stds.get("hr",   1.0)
            rbi_mean = league_means.get("rbi", 0.0)
            rbi_std  = league_stds.get("rbi",  1.0)
            sb_mean  = league_means.get("sb",  0.0)
            sb_std   = league_stds.get("sb",   1.0)
            avg_mean = league_means.get("avg", 0.0)
            avg_std  = league_stds.get("avg",  1.0)

            for i in range(n_simulations):
                zs = []
                if hr_std > 0:
                    zs.append((sim_hr[i]  - hr_mean)  / hr_std)
                if rbi_std > 0:
                    zs.append((sim_rbi[i] - rbi_mean) / rbi_std)
                if sb_std > 0:
                    zs.append((sim_sb[i]  - sb_mean)  / sb_std)
                if avg_std > 0:
                    zs.append((sim_avg[i] - avg_mean) / avg_std)
                comp = sum(zs) / len(zs) if zs else 0.0
                sim_composites.append(comp)

    # ------------------------------------------------------------------
    # Pitching simulation
    # ------------------------------------------------------------------
    sim_k_list    = []
    sim_era_list  = []
    sim_whip_list = []

    if has_pitching:
        ip_rate  = (rolling_row.w_ip              or 0.0) / g
        k_rate   = (rolling_row.w_strikeouts_pit  or 0.0) / g
        er_rate  = (rolling_row.w_earned_runs     or 0.0) / g
        h_rate   = (rolling_row.w_hits_allowed    or 0.0) / g
        bb_rate  = (rolling_row.w_walks_allowed   or 0.0) / g

        for _ in range(n_simulations):
            total_ip  = _draw_games(rng, ip_rate,  remaining_games)
            total_k   = _draw_games(rng, k_rate,   remaining_games)
            total_er  = _draw_games(rng, er_rate,  remaining_games)
            total_h   = _draw_games(rng, h_rate,   remaining_games)
            total_bb  = _draw_games(rng, bb_rate,  remaining_games)

            era  = 9.0 * total_er / total_ip       if total_ip > 0 else 0.0
            whip = (total_h + total_bb) / total_ip if total_ip > 0 else 0.0

            sim_k_list.append(total_k)
            sim_era_list.append(era)
            sim_whip_list.append(whip)

        (
            result.proj_k_p10,
            result.proj_k_p25,
            result.proj_k_p50,
            result.proj_k_p75,
            result.proj_k_p90,
        ) = _percentiles(sim_k_list)

        (
            result.proj_era_p10,
            result.proj_era_p25,
            result.proj_era_p50,
            result.proj_era_p75,
            result.proj_era_p90,
        ) = _percentiles(sim_era_list)

        (
            result.proj_whip_p10,
            result.proj_whip_p25,
            result.proj_whip_p50,
            result.proj_whip_p75,
            result.proj_whip_p90,
        ) = _percentiles(sim_whip_list)

        # Augment sim_composites with pitcher K z-scores if league params available
        if (
            league_means is not None
            and league_stds is not None
            and league_stds.get("k", 0) > 0
            and not has_batting    # pure pitcher -- composites built here
        ):
            k_mean  = league_means.get("k",  0.0)
            k_std   = league_stds.get("k",   1.0)
            for i in range(n_simulations):
                zs = []
                if k_std > 0:
                    zs.append((sim_k_list[i] - k_mean) / k_std)
                comp = sum(zs) / len(zs) if zs else 0.0
                sim_composites.append(comp)

    # ------------------------------------------------------------------
    # Risk metrics (require sim_composites)
    # ------------------------------------------------------------------
    if sim_composites:
        (
            result.composite_variance,
            result.downside_p25,
            result.upside_p75,
            result.prob_above_median,
        ) = _compute_composite_risk(sim_composites)

    return result


def simulate_all_players(
    rolling_rows: list,
    remaining_games: int = REMAINING_GAMES_DEFAULT,
    n_simulations: int = N_SIMULATIONS,
    league_means: Optional[dict] = None,
    league_stds: Optional[dict] = None,
) -> list:
    """
    Run simulate_player for every row in rolling_rows.

    Rows where player_type resolves to "unknown" are silently skipped
    (no batting AND no pitching data -- not useful for projection).

    Parameters
    ----------
    rolling_rows : list of PlayerRollingStats ORM objects (window_days=14)
    remaining_games : int
    n_simulations : int
    league_means : dict or None -- forwarded to simulate_player
    league_stds : dict or None -- forwarded to simulate_player

    Returns
    -------
    list of SimulationResult dataclass objects (unknown types excluded)
    """
    results = []
    for row in rolling_rows:
        r = simulate_player(
            row,
            remaining_games=remaining_games,
            n_simulations=n_simulations,
            seed=None,
            league_means=league_means,
            league_stds=league_stds,
        )
        if r.player_type != "unknown":
            results.append(r)
    return results
