"""
test_scoring_engine_statcast.py -- P28 Phase 2: z_power_quality tests.

Validates:
  - z_power_quality is computed from barrel%, hard_hit%, exit_velocity
  - z_power_quality enters composite_z for hitters and two_way players
  - z_power_quality is None for pitchers
  - Missing Statcast components are handled gracefully
  - Component Z-scores are properly capped
  - Backward compatibility: existing categories still work
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pytest

from backend.services.scoring_engine import (
    PlayerScoreResult,
    compute_league_zscores,
    _compute_component_z,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeRollingRow:
    """Minimal fake PlayerRollingStats row for testing."""
    bdl_player_id: int
    games_in_window: int
    # Hitting
    w_ab: Optional[float] = None
    w_hits: Optional[float] = None
    w_doubles: Optional[float] = None
    w_triples: Optional[float] = None
    w_home_runs: Optional[float] = None
    w_rbi: Optional[float] = None
    w_walks: Optional[float] = None
    w_strikeouts_bat: Optional[float] = None
    w_stolen_bases: Optional[float] = None
    w_caught_stealing: Optional[float] = None
    w_net_stolen_bases: Optional[float] = None
    w_avg: Optional[float] = None
    w_obp: Optional[float] = None
    # Pitching
    w_ip: Optional[float] = None
    w_earned_runs: Optional[float] = None
    w_hits_allowed: Optional[float] = None
    w_walks_allowed: Optional[float] = None
    w_strikeouts_pit: Optional[float] = None
    w_era: Optional[float] = None
    w_whip: Optional[float] = None
    w_k_per_9: Optional[float] = None
    # Statcast (P28)
    w_barrel_pct: Optional[float] = None
    w_hard_hit_pct: Optional[float] = None
    w_exit_velocity_avg: Optional[float] = None


def _make_hitter(
    pid: int,
    *,
    hr: float = 0.0,
    rbi: float = 0.0,
    nsb: float = 0.0,
    avg: float = 0.250,
    obp: float = 0.320,
    barrel_pct: Optional[float] = None,
    hard_hit_pct: Optional[float] = None,
    exit_velocity: Optional[float] = None,
) -> FakeRollingRow:
    """Create a hitter row with all required fields."""
    return FakeRollingRow(
        bdl_player_id=pid,
        games_in_window=10,
        w_ab=40.0,
        w_hits=avg * 40.0,
        w_home_runs=hr,
        w_rbi=rbi,
        w_stolen_bases=max(0, nsb),
        w_caught_stealing=max(0, -nsb),
        w_net_stolen_bases=nsb,
        w_avg=avg,
        w_obp=obp,
        w_barrel_pct=barrel_pct,
        w_hard_hit_pct=hard_hit_pct,
        w_exit_velocity_avg=exit_velocity,
    )


def _make_pitcher(
    pid: int,
    *,
    era: float = 4.00,
    whip: float = 1.30,
    k9: float = 8.5,
) -> FakeRollingRow:
    """Create a pitcher row with all required fields."""
    return FakeRollingRow(
        bdl_player_id=pid,
        games_in_window=10,
        w_ip=50.0,
        w_earned_runs=era * 50.0 / 9.0,
        w_hits_allowed=whip * 50.0 * 0.8,  # approx
        w_walks_allowed=whip * 50.0 * 0.2,
        w_strikeouts_pit=k9 * 50.0 / 9.0,
        w_era=era,
        w_whip=whip,
        w_k_per_9=k9,
    )


# ---------------------------------------------------------------------------
# _compute_component_z tests
# ---------------------------------------------------------------------------

def test_component_z_basic():
    """Component Z-scores are computed and capped."""
    pairs = [
        (1, 0.10),  # low barrel%
        (2, 0.15),  # medium
        (3, 0.20),  # high
        (4, 0.25),  # very high
        (5, 0.30),  # elite
    ]
    result = _compute_component_z(pairs)
    assert len(result) == 5
    # Higher values should have higher Z
    assert result[5] > result[3] > result[1]
    # All Zs should be within cap
    for z in result.values():
        assert -3.0 <= z <= 3.0


def test_component_z_insufficient_sample():
    """Fewer than min_sample pairs returns empty dict."""
    pairs = [(1, 0.10)]  # Only 1, below default min_sample=2
    result = _compute_component_z(pairs)
    assert result == {}


def test_component_z_custom_min_sample():
    """Custom min_sample parameter works."""
    pairs = [(1, 0.10), (2, 0.15), (3, 0.20)]  # 3 items
    result = _compute_component_z(pairs, min_sample=4)  # Require 4
    assert result == {}  # Not enough

    result = _compute_component_z(pairs, min_sample=2)  # Require 2
    assert len(result) == 3  # Works


def test_component_z_degenerate():
    """All identical values returns empty dict."""
    pairs = [(1, 0.15), (2, 0.15), (3, 0.15), (4, 0.15), (5, 0.15)]
    result = _compute_component_z(pairs)
    assert result == {}


# ---------------------------------------------------------------------------
# z_power_quality computation tests
# ---------------------------------------------------------------------------

def test_power_quality_computed_for_hitters():
    """Hitters with Statcast data get z_power_quality."""
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300, barrel_pct=0.15, hard_hit_pct=0.45, exit_velocity=92.0),
        _make_hitter(2, hr=3, rbi=10, avg=0.280, barrel_pct=0.08, hard_hit_pct=0.35, exit_velocity=88.0),
        _make_hitter(3, hr=8, rbi=20, avg=0.320, barrel_pct=0.20, hard_hit_pct=0.50, exit_velocity=95.0),
        _make_hitter(4, hr=2, rbi=8,  avg=0.260, barrel_pct=0.05, hard_hit_pct=0.30, exit_velocity=85.0),
        _make_hitter(5, hr=6, rbi=18, avg=0.290, barrel_pct=0.12, hard_hit_pct=0.40, exit_velocity=90.0),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)

    by_pid = {r.bdl_player_id: r for r in results}
    assert len(by_pid) == 5

    # Player 3 has best Statcast profile -> highest z_power_quality
    assert by_pid[3].z_power_quality > by_pid[1].z_power_quality
    assert by_pid[3].z_power_quality > by_pid[2].z_power_quality

    # Player 4 has worst Statcast profile -> lowest z_power_quality
    assert by_pid[4].z_power_quality < by_pid[2].z_power_quality

    # All hitters should have z_power_quality
    for r in results:
        assert r.z_power_quality is not None


def test_power_quality_none_for_pitchers():
    """Pitchers do not get z_power_quality."""
    rows = [
        _make_pitcher(1, era=3.50, whip=1.20, k9=9.0),
        _make_pitcher(2, era=4.50, whip=1.40, k9=7.5),
        _make_pitcher(3, era=2.80, whip=1.10, k9=10.0),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)

    for r in results:
        assert r.z_power_quality is None


def test_power_quality_affects_composite():
    """z_power_quality is included in composite_z calculation."""
    # Need 5+ players with varied traditional stats for Z-score computation
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300, barrel_pct=0.15, hard_hit_pct=0.45, exit_velocity=92.0),
        _make_hitter(2, hr=3, rbi=10, avg=0.280, barrel_pct=0.08, hard_hit_pct=0.35, exit_velocity=88.0),
        _make_hitter(3, hr=8, rbi=20, avg=0.320, barrel_pct=0.20, hard_hit_pct=0.50, exit_velocity=95.0),
        _make_hitter(4, hr=2, rbi=8,  avg=0.260, barrel_pct=0.05, hard_hit_pct=0.30, exit_velocity=85.0),
        _make_hitter(5, hr=6, rbi=18, avg=0.290, barrel_pct=0.12, hard_hit_pct=0.40, exit_velocity=90.0),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Player 3 has best Statcast -> highest z_power_quality
    assert by_pid[3].z_power_quality > by_pid[1].z_power_quality
    assert by_pid[3].z_power_quality > by_pid[2].z_power_quality

    # Player 4 has worst Statcast -> lowest z_power_quality
    assert by_pid[4].z_power_quality < by_pid[2].z_power_quality

    # z_power_quality should affect composite: Player 3 > Player 1 (better Statcast)
    assert by_pid[3].composite_z > by_pid[1].composite_z


def test_power_quality_missing_components():
    """Players with partial Statcast data still get z_power_quality from available components."""
    # Need varied traditional stats so category Zs compute, plus enough players per component
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300, barrel_pct=0.15),  # Only barrel%
        _make_hitter(2, hr=4, rbi=12, avg=0.290, barrel_pct=0.12),  # Barrel% (for component sample)
        _make_hitter(3, hr=3, rbi=10, avg=0.280, hard_hit_pct=0.45),  # Only hard_hit%
        _make_hitter(4, hr=2, rbi=8,  avg=0.270, hard_hit_pct=0.40),  # Hard_hit% (for component sample)
        _make_hitter(5, hr=6, rbi=18, avg=0.310, exit_velocity=92.0),  # Only exit velocity
        _make_hitter(6, hr=7, rbi=20, avg=0.320, exit_velocity=90.0),  # Exit velocity (for component sample)
        _make_hitter(7, hr=8, rbi=22, avg=0.330, barrel_pct=0.20, hard_hit_pct=0.50, exit_velocity=95.0),  # All three
        _make_hitter(8, hr=1, rbi=5,  avg=0.250),  # No Statcast at all
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Players with at least one component should have z_power_quality
    assert by_pid[1].z_power_quality is not None  # Has barrel%
    assert by_pid[3].z_power_quality is not None  # Has hard_hit%
    assert by_pid[5].z_power_quality is not None  # Has exit velocity
    assert by_pid[7].z_power_quality is not None  # Has all three

    # Player with no Statcast should have None
    assert by_pid[8].z_power_quality is None

    # Player with all components should have highest power quality (best stats)
    assert by_pid[7].z_power_quality > by_pid[1].z_power_quality


def test_power_quality_two_way_players():
    """Two-way players get z_power_quality along with hitting AND pitching Zs."""
    # Need at least 5 players with pitching stats for pitcher Z-score computation
    rows = [
        # Player 1: Two-way with excellent stats
        FakeRollingRow(
            bdl_player_id=1,
            games_in_window=10,
            w_ab=40.0, w_hits=12.0, w_home_runs=5.0, w_rbi=15.0,
            w_stolen_bases=2.0, w_caught_stealing=0.0, w_net_stolen_bases=2.0,
            w_avg=0.300, w_obp=0.350,
            w_ip=20.0, w_earned_runs=10.0, w_hits_allowed=18.0, w_walks_allowed=6.0,
            w_strikeouts_pit=25.0, w_era=4.50, w_whip=1.20, w_k_per_9=11.25,
            w_barrel_pct=0.15, w_hard_hit_pct=0.45, w_exit_velocity_avg=92.0,
        ),
        # Player 2: Two-way with moderate stats
        FakeRollingRow(
            bdl_player_id=2,
            games_in_window=10,
            w_ab=38.0, w_hits=10.0, w_home_runs=3.0, w_rbi=10.0,
            w_stolen_bases=1.0, w_caught_stealing=0.0, w_net_stolen_bases=1.0,
            w_avg=0.263, w_obp=0.320,
            w_ip=18.0, w_earned_runs=9.0, w_hits_allowed=17.0, w_walks_allowed=5.0,
            w_strikeouts_pit=20.0, w_era=4.50, w_whip=1.22, w_k_per_9=10.0,
            w_barrel_pct=0.10, w_hard_hit_pct=0.38, w_exit_velocity_avg=89.0,
        ),
        # Player 3: Two-way with poor stats
        FakeRollingRow(
            bdl_player_id=3,
            games_in_window=10,
            w_ab=35.0, w_hits=8.0, w_home_runs=2.0, w_rbi=8.0,
            w_stolen_bases=0.0, w_caught_stealing=1.0, w_net_stolen_bases=-1.0,
            w_avg=0.229, w_obp=0.280,
            w_ip=15.0, w_earned_runs=10.0, w_hits_allowed=16.0, w_walks_allowed=6.0,
            w_strikeouts_pit=15.0, w_era=6.00, w_whip=1.47, w_k_per_9=9.0,
            w_barrel_pct=0.05, w_hard_hit_pct=0.30, w_exit_velocity_avg=85.0,
        ),
        # Player 4: Pure pitcher (to meet MIN_SAMPLE for pitcher categories)
        _make_pitcher(4, era=3.50, whip=1.20, k9=9.5),
        # Player 5: Pure pitcher (to meet MIN_SAMPLE for pitcher categories)
        _make_pitcher(5, era=4.00, whip=1.25, k9=8.0),
        # Player 6: Pure hitter (to meet MIN_SAMPLE for hitter categories)
        _make_hitter(6, hr=4, rbi=12, avg=0.275, barrel_pct=0.12, hard_hit_pct=0.40, exit_velocity=90.0),
        # Player 7: Pure hitter (to meet MIN_SAMPLE for hitter categories)
        _make_hitter(7, hr=6, rbi=18, avg=0.290, barrel_pct=0.14, hard_hit_pct=0.42, exit_velocity=91.0),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Player 1 is two_way
    assert by_pid[1].player_type == "two_way"
    assert by_pid[1].z_hr is not None      # Has hitting Zs (7 hitters total)
    assert by_pid[1].z_era is not None     # Has pitching Zs (5 pitchers total)
    assert by_pid[1].z_power_quality is not None  # Has Statcast power quality


def test_power_quality_no_statcast_pool():
    """When no hitters have Statcast data, z_power_quality is None for all."""
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300),  # No Statcast
        _make_hitter(2, hr=3, rbi=10, avg=0.280),  # No Statcast
        _make_hitter(3, hr=8, rbi=20, avg=0.320),  # No Statcast
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    for r in results:
        assert r.z_power_quality is None


def test_power_quality_capped():
    """z_power_quality is capped at +/- Z_CAP (3.0)."""
    # Create extreme outliers
    rows = [
        _make_hitter(1, hr=1, rbi=5, avg=0.200, barrel_pct=0.50, hard_hit_pct=0.80, exit_velocity=105.0),
        _make_hitter(2, hr=1, rbi=5, avg=0.200, barrel_pct=0.01, hard_hit_pct=0.10, exit_velocity=75.0),
    ]
    # Add 3 more to meet MIN_SAMPLE=5
    for i in range(3, 6):
        rows.append(_make_hitter(i, hr=5, rbi=15, avg=0.300, barrel_pct=0.10, hard_hit_pct=0.35, exit_velocity=88.0))

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Outliers should be capped
    assert abs(by_pid[1].z_power_quality) <= 3.0
    assert abs(by_pid[2].z_power_quality) <= 3.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_existing_categories_unchanged():
    """Traditional Z-scores (z_hr, z_rbi, etc.) still work exactly as before."""
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300, obp=0.350, nsb=2),
        _make_hitter(2, hr=3, rbi=10, avg=0.280, obp=0.320, nsb=0),
        _make_hitter(3, hr=8, rbi=20, avg=0.320, obp=0.380, nsb=5),
        _make_hitter(4, hr=2, rbi=8,  avg=0.260, obp=0.300, nsb=-1),
        _make_hitter(5, hr=6, rbi=18, avg=0.290, obp=0.340, nsb=3),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Player 3 has best stats -> highest composite
    assert by_pid[3].composite_z > by_pid[1].composite_z

    # Player 4 has worst stats -> lowest composite
    assert by_pid[4].composite_z < by_pid[2].composite_z

    # All traditional categories present
    for r in results:
        assert r.z_hr is not None
        assert r.z_rbi is not None
        assert r.z_nsb is not None
        assert r.z_avg is not None
        assert r.z_obp is not None


def test_score_0_100_ranking():
    """score_0_100 percentile ranks are computed correctly with z_power_quality."""
    rows = [
        _make_hitter(1, hr=5, rbi=15, avg=0.300, barrel_pct=0.15, hard_hit_pct=0.45, exit_velocity=92.0),
        _make_hitter(2, hr=3, rbi=10, avg=0.280, barrel_pct=0.08, hard_hit_pct=0.35, exit_velocity=88.0),
        _make_hitter(3, hr=8, rbi=20, avg=0.320, barrel_pct=0.20, hard_hit_pct=0.50, exit_velocity=95.0),
        _make_hitter(4, hr=2, rbi=8,  avg=0.260, barrel_pct=0.05, hard_hit_pct=0.30, exit_velocity=85.0),
        _make_hitter(5, hr=6, rbi=18, avg=0.290, barrel_pct=0.12, hard_hit_pct=0.40, exit_velocity=90.0),
    ]

    results = compute_league_zscores(rows, as_of_date=date(2026, 4, 10), window_days=7)
    by_pid = {r.bdl_player_id: r for r in results}

    # Best player should have highest score
    assert by_pid[3].score_0_100 >= by_pid[1].score_0_100
    assert by_pid[3].score_0_100 >= by_pid[2].score_0_100

    # Worst player should have lowest score
    assert by_pid[4].score_0_100 <= by_pid[2].score_0_100

    # All scores in 0-100 range
    for r in results:
        assert 0.0 <= r.score_0_100 <= 100.0
