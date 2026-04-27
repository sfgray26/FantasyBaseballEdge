"""
Tests for backend/services/simulation_engine.py (P16 Monte Carlo ROS Simulation).

All tests use seed=42 for deterministic output.
No DB, no network, no numpy/scipy -- stdlib only.
"""

import math
from datetime import date
from unittest.mock import MagicMock

import pytest

from backend.services.simulation_engine import (
    SimulationResult,
    _percentiles,
    _sample_positive,
    simulate_player,
    simulate_all_players,
    CV,
    N_SIMULATIONS,
    REMAINING_GAMES_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hitter(
    bdl_player_id=1,
    as_of_date=None,
    games_in_window=14,
    w_ab=50.0,
    w_hits=14.0,
    w_home_runs=3.0,
    w_rbi=10.0,
    w_stolen_bases=2.0,
    w_walks=5.0,
):
    """Return a MagicMock that looks like a PlayerRollingStats hitter row."""
    row = MagicMock()
    row.bdl_player_id = bdl_player_id
    row.as_of_date = as_of_date or date(2026, 4, 5)
    row.games_in_window = games_in_window
    row.w_ab = w_ab
    row.w_hits = w_hits
    row.w_home_runs = w_home_runs
    row.w_rbi = w_rbi
    row.w_stolen_bases = w_stolen_bases
    row.w_walks = w_walks
    row.w_ip = None
    row.w_strikeouts_pit = None
    row.w_earned_runs = None
    row.w_hits_allowed = None
    row.w_walks_allowed = None
    return row


def _make_pitcher(
    bdl_player_id=2,
    as_of_date=None,
    games_in_window=5,
    w_ip=30.0,
    w_strikeouts_pit=35.0,
    w_earned_runs=8.0,
    w_hits_allowed=25.0,
    w_walks_allowed=10.0,
):
    """Return a MagicMock that looks like a PlayerRollingStats pitcher row."""
    row = MagicMock()
    row.bdl_player_id = bdl_player_id
    row.as_of_date = as_of_date or date(2026, 4, 5)
    row.games_in_window = games_in_window
    row.w_ab = None
    row.w_hits = None
    row.w_home_runs = None
    row.w_rbi = None
    row.w_stolen_bases = None
    row.w_walks = None
    row.w_ip = w_ip
    row.w_strikeouts_pit = w_strikeouts_pit
    row.w_earned_runs = w_earned_runs
    row.w_hits_allowed = w_hits_allowed
    row.w_walks_allowed = w_walks_allowed
    return row


def _make_two_way(bdl_player_id=3, as_of_date=None):
    """Return a MagicMock that has both batting and pitching fields."""
    row = _make_hitter(bdl_player_id=bdl_player_id, as_of_date=as_of_date)
    row.w_ip = 15.0
    row.w_strikeouts_pit = 20.0
    row.w_earned_runs = 4.0
    row.w_hits_allowed = 12.0
    row.w_walks_allowed = 5.0
    return row


def _make_unknown(bdl_player_id=99, as_of_date=None):
    """Return a row with no batting or pitching data."""
    row = MagicMock()
    row.bdl_player_id = bdl_player_id
    row.as_of_date = as_of_date or date(2026, 4, 5)
    row.games_in_window = 5
    row.w_ab = None
    row.w_hits = None
    row.w_home_runs = None
    row.w_rbi = None
    row.w_stolen_bases = None
    row.w_walks = None
    row.w_ip = None
    row.w_strikeouts_pit = None
    row.w_earned_runs = None
    row.w_hits_allowed = None
    row.w_walks_allowed = None
    return row


# ===========================================================================
# _percentiles
# ===========================================================================

def test_percentiles_sorted():
    """_percentiles must return P10 <= P25 <= P50 <= P75 <= P90."""
    import random
    rng = random.Random(42)
    values = [rng.gauss(50, 15) for _ in range(1000)]
    p10, p25, p50, p75, p90 = _percentiles(values)
    assert p10 <= p25
    assert p25 <= p50
    assert p50 <= p75
    assert p75 <= p90


def test_percentiles_single_value():
    """A list of one element should return that value for all percentiles."""
    p10, p25, p50, p75, p90 = _percentiles([7.0])
    assert p10 == 7.0
    assert p25 == 7.0
    assert p50 == 7.0
    assert p75 == 7.0
    assert p90 == 7.0


def test_percentiles_empty():
    """Empty list should return all 0.0."""
    result = _percentiles([])
    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_percentiles_known_values():
    """Verify specific index mapping for a sorted list of 1000 elements."""
    values = list(range(1000))   # 0..999
    p10, p25, p50, p75, p90 = _percentiles(values)
    # int(0.10 * 1000) = 100 -> values[100] = 100
    assert p10 == 100
    # int(0.25 * 1000) = 250 -> values[250] = 250
    assert p25 == 250
    # int(0.50 * 1000) = 500 -> values[500] = 500
    assert p50 == 500
    # int(0.75 * 1000) = 750 -> values[750] = 750
    assert p75 == 750
    # int(0.90 * 1000) = 900 -> values[900] = 900
    assert p90 == 900


# ===========================================================================
# simulate_player -- hitter
# ===========================================================================

def test_simulate_player_hitter_returns_batting_percentiles():
    """Hitter with hr_rate > 0 should have non-None batting percentiles and None pitching."""
    row = _make_hitter()
    result = simulate_player(row, remaining_games=130, seed=42)

    assert result.player_type == "hitter"

    # Batting percentiles populated
    assert result.proj_hr_p50 is not None
    assert result.proj_hr_p50 > 0
    assert result.proj_rbi_p50 is not None
    assert result.proj_sb_p50 is not None
    assert result.proj_avg_p50 is not None

    # Pitching fields must be None
    assert result.proj_k_p10 is None
    assert result.proj_k_p50 is None
    assert result.proj_era_p50 is None
    assert result.proj_whip_p50 is None


def test_simulate_player_spread_p90_greater_than_p10():
    """HR projections should show variance: P90 > P10."""
    row = _make_hitter()
    result = simulate_player(row, remaining_games=130, seed=42)
    assert result.proj_hr_p90 > result.proj_hr_p10


def test_simulate_player_zero_rate_gives_near_zero_projection():
    """Player with w_home_runs=0 should produce near-zero HR projections."""
    row = _make_hitter(w_home_runs=0.0)
    result = simulate_player(row, remaining_games=130, seed=42)
    # With mu=0, _sample_positive always returns 0 -> all percentiles exactly 0
    assert result.proj_hr_p50 == 0.0
    assert result.proj_hr_p90 == 0.0


def test_simulate_player_avg_between_0_and_1():
    """All AVG percentile projections must lie in [0, 1]."""
    row = _make_hitter()
    result = simulate_player(row, remaining_games=130, seed=42)
    for p in [
        result.proj_avg_p10,
        result.proj_avg_p25,
        result.proj_avg_p50,
        result.proj_avg_p75,
        result.proj_avg_p90,
    ]:
        assert p is not None
        assert 0.0 <= p <= 1.0, f"AVG percentile out of range: {p}"


# ===========================================================================
# simulate_player -- pitcher
# ===========================================================================

def test_simulate_player_pitcher_returns_pitching_percentiles():
    """Pitcher with k_rate > 0 should have non-None pitching percentiles and None batting."""
    row = _make_pitcher()
    result = simulate_player(row, remaining_games=130, seed=42)

    assert result.player_type == "pitcher"

    # Pitching percentiles populated
    assert result.proj_k_p50 is not None
    assert result.proj_k_p50 > 0
    assert result.proj_era_p50 is not None
    assert result.proj_whip_p50 is not None

    # Batting fields must be None
    assert result.proj_hr_p50 is None
    assert result.proj_rbi_p50 is None
    assert result.proj_sb_p50 is None
    assert result.proj_avg_p50 is None


def test_simulate_player_era_non_negative():
    """All ERA percentile projections must be >= 0."""
    row = _make_pitcher()
    result = simulate_player(row, remaining_games=130, seed=42)
    for p in [
        result.proj_era_p10,
        result.proj_era_p25,
        result.proj_era_p50,
        result.proj_era_p75,
        result.proj_era_p90,
    ]:
        assert p is not None
        assert p >= 0.0, f"ERA percentile negative: {p}"


def test_simulate_player_era_sorted():
    """ERA percentiles should be sorted (ERA is monotonically non-decreasing in percentiles)."""
    row = _make_pitcher()
    result = simulate_player(row, remaining_games=130, seed=42)
    assert result.proj_era_p10 <= result.proj_era_p25
    assert result.proj_era_p25 <= result.proj_era_p50
    assert result.proj_era_p50 <= result.proj_era_p75
    assert result.proj_era_p75 <= result.proj_era_p90


# ===========================================================================
# simulate_player -- two_way
# ===========================================================================

def test_simulate_player_two_way_has_both():
    """Two-way player should have both batting AND pitching percentiles populated."""
    row = _make_two_way()
    result = simulate_player(row, remaining_games=130, seed=42)

    assert result.player_type == "two_way"

    # Batting populated
    assert result.proj_hr_p50 is not None
    assert result.proj_rbi_p50 is not None
    assert result.proj_avg_p50 is not None

    # Pitching populated
    assert result.proj_k_p50 is not None
    assert result.proj_era_p50 is not None
    assert result.proj_whip_p50 is not None


# ===========================================================================
# simulate_player -- unknown type
# ===========================================================================

def test_simulate_player_unknown_type_returns_all_none():
    """Row with w_ab=None and w_ip=None -> player_type='unknown', all projections None."""
    row = _make_unknown()
    result = simulate_player(row, remaining_games=130, seed=42)

    assert result.player_type == "unknown"
    assert result.proj_hr_p50 is None
    assert result.proj_rbi_p50 is None
    assert result.proj_k_p50 is None
    assert result.proj_era_p50 is None
    assert result.composite_variance is None


# ===========================================================================
# simulate_player -- comparative / scaling tests
# ===========================================================================

def test_simulate_player_higher_rate_gives_higher_p50():
    """Player with 2x HR rate should produce approximately 2x proj_hr_p50 (seed=42)."""
    row_base   = _make_hitter(w_home_runs=2.0, games_in_window=14)
    row_double = _make_hitter(w_home_runs=4.0, games_in_window=14)

    r_base   = simulate_player(row_base,   remaining_games=130, seed=42)
    r_double = simulate_player(row_double, remaining_games=130, seed=42)

    # P50 should scale roughly proportionally (allow 30% tolerance for Monte Carlo variance)
    assert r_double.proj_hr_p50 > r_base.proj_hr_p50 * 1.5, (
        f"Expected double HR player to have >1.5x P50: "
        f"base={r_base.proj_hr_p50:.2f} double={r_double.proj_hr_p50:.2f}"
    )


def test_remaining_games_parameter_affects_p50():
    """Same player, remaining_games=50 vs 130 -> p50_hr should differ substantially."""
    row = _make_hitter()
    r_short  = simulate_player(row, remaining_games=50,  seed=42)
    r_full   = simulate_player(row, remaining_games=130, seed=42)

    assert r_full.proj_hr_p50 > r_short.proj_hr_p50 * 1.5, (
        f"Expected 130-game horizon to give >1.5x HR vs 50-game: "
        f"short={r_short.proj_hr_p50:.2f} full={r_full.proj_hr_p50:.2f}"
    )


# ===========================================================================
# simulate_player -- metadata fields
# ===========================================================================

def test_simulate_player_metadata_passthrough():
    """SimulationResult metadata must match the input row and parameters."""
    as_of = date(2026, 4, 5)
    row = _make_hitter(bdl_player_id=77, as_of_date=as_of)
    result = simulate_player(row, remaining_games=110, n_simulations=500, seed=42)

    assert result.bdl_player_id == 77
    assert result.as_of_date == as_of
    assert result.window_days == 14
    assert result.remaining_games == 110
    assert result.n_simulations == 500


# ===========================================================================
# simulate_all_players
# ===========================================================================

def test_simulate_all_players_skips_unknown():
    """simulate_all_players with 2 hitters + 1 unknown should return 2 results."""
    rows = [
        _make_hitter(bdl_player_id=1),
        _make_hitter(bdl_player_id=2),
        _make_unknown(bdl_player_id=99),
    ]
    results = simulate_all_players(rows, remaining_games=30, n_simulations=50)
    assert len(results) == 2
    ids = {r.bdl_player_id for r in results}
    assert ids == {1, 2}
    assert 99 not in ids


def test_simulate_all_players_returns_correct_count():
    """simulate_all_players with 3 hitters should return exactly 3 SimulationResult objects."""
    rows = [
        _make_hitter(bdl_player_id=i) for i in range(3)
    ]
    results = simulate_all_players(rows, remaining_games=30, n_simulations=50)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, SimulationResult)


def test_simulate_all_players_empty_input():
    """Empty input list should return empty list."""
    results = simulate_all_players([], remaining_games=130)
    assert results == []


def test_simulate_all_players_mixed_types():
    """Batch with hitter, pitcher, two-way, unknown should exclude only unknown."""
    rows = [
        _make_hitter(bdl_player_id=1),
        _make_pitcher(bdl_player_id=2),
        _make_two_way(bdl_player_id=3),
        _make_unknown(bdl_player_id=99),
    ]
    results = simulate_all_players(rows, remaining_games=30, n_simulations=50)
    assert len(results) == 3
    types = {r.player_type for r in results}
    assert types == {"hitter", "pitcher", "two_way"}


# ===========================================================================
# Constants
# ===========================================================================

def test_constants_have_expected_values():
    """CV, N_SIMULATIONS, REMAINING_GAMES_DEFAULT must match spec."""
    assert CV == 0.35
    assert N_SIMULATIONS == 1000
    assert REMAINING_GAMES_DEFAULT == 130
