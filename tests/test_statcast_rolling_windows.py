"""
test_statcast_rolling_windows.py -- P28 Phase 1: Statcast-enhanced rolling window tests.

Validates:
  - StatcastDailyRow dataclass construction
  - compute_rolling_window_with_statcast merges Statcast data correctly
  - compute_all_rolling_windows_with_statcast handles multiple players/windows
  - Decay weighting applies to Statcast metrics
  - xwOBA - wOBA luck differential computation
  - Graceful fallback when Statcast data is missing
  - Graceful fallback when BDL data is missing but Statcast exists
"""

from datetime import date, timedelta
from typing import Optional

import pytest

from backend.services.rolling_window_engine import (
    RollingWindowResult,
    StatcastDailyRow,
    compute_rolling_window,
    compute_rolling_window_with_statcast,
    compute_all_rolling_windows_with_statcast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeBDLRow:
    """Minimal fake BDL row for testing."""
    def __init__(
        self,
        bdl_player_id: int,
        game_date: date,
        ab: Optional[int] = None,
        hits: Optional[int] = None,
        doubles: Optional[int] = None,
        triples: Optional[int] = None,
        home_runs: Optional[int] = None,
        rbi: Optional[int] = None,
        walks: Optional[int] = None,
        strikeouts_bat: Optional[int] = None,
        stolen_bases: Optional[int] = None,
        caught_stealing: Optional[int] = None,
        innings_pitched: Optional[str] = None,
        earned_runs: Optional[int] = None,
        hits_allowed: Optional[int] = None,
        walks_allowed: Optional[int] = None,
        strikeouts_pit: Optional[int] = None,
    ):
        self.bdl_player_id = bdl_player_id
        self.game_date = game_date
        self.ab = ab
        self.hits = hits
        self.doubles = doubles
        self.triples = triples
        self.home_runs = home_runs
        self.rbi = rbi
        self.walks = walks
        self.strikeouts_bat = strikeouts_bat
        self.stolen_bases = stolen_bases
        self.caught_stealing = caught_stealing
        self.innings_pitched = innings_pitched
        self.earned_runs = earned_runs
        self.hits_allowed = hits_allowed
        self.walks_allowed = walks_allowed
        self.strikeouts_pit = strikeouts_pit


# ---------------------------------------------------------------------------
# StatcastDailyRow tests
# ---------------------------------------------------------------------------

def test_statcast_daily_row_construction():
    """StatcastDailyRow can be constructed with all fields."""
    row = StatcastDailyRow(
        player_id="mlbam:12345",
        game_date=date(2026, 4, 1),
        exit_velocity_avg=92.5,
        launch_angle_avg=15.0,
        hard_hit_pct=0.45,
        barrel_pct=0.12,
        xwoba=0.380,
        xba=0.285,
        xslg=0.520,
        woba=0.350,
    )
    assert row.player_id == "mlbam:12345"
    assert row.exit_velocity_avg == 92.5
    assert row.barrel_pct == 0.12


def test_statcast_daily_row_defaults():
    """StatcastDailyRow defaults optional fields to None."""
    row = StatcastDailyRow(
        player_id="mlbam:12345",
        game_date=date(2026, 4, 1),
    )
    assert row.exit_velocity_avg is None
    assert row.xwoba is None


# ---------------------------------------------------------------------------
# compute_rolling_window_with_statcast tests
# ---------------------------------------------------------------------------

def test_statcast_merge_basic():
    """Statcast metrics are merged and decay-weighted correctly."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(
            bdl_player_id=player_id,
            game_date=as_of_date,
            ab=4,
            hits=2,
            home_runs=1,
            rbi=2,
        ),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date,
            exit_velocity_avg=95.0,
            hard_hit_pct=0.50,
            barrel_pct=0.15,
            xwoba=0.400,
            xba=0.300,
            xslg=0.550,
            woba=0.350,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.bdl_player_id == player_id
    assert result.w_exit_velocity_avg == pytest.approx(95.0, abs=0.001)
    assert result.w_hard_hit_pct == pytest.approx(0.50, abs=0.001)
    assert result.w_barrel_pct == pytest.approx(0.15, abs=0.001)
    assert result.w_xwoba == pytest.approx(0.400, abs=0.001)
    assert result.w_xba == pytest.approx(0.300, abs=0.001)
    assert result.w_xslg == pytest.approx(0.550, abs=0.001)
    assert result.w_xwoba_minus_woba == pytest.approx(0.050, abs=0.001)  # 0.400 - 0.350


def test_statcast_decay_weighting():
    """Older Statcast data receives lower weight."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345
    decay_lambda = 0.95

    # Two games: today and 2 days ago
    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=1),
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date - timedelta(days=2), ab=4, hits=1),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date,
            exit_velocity_avg=100.0,  # hot
            barrel_pct=0.20,
        ),
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date - timedelta(days=2),
            exit_velocity_avg=90.0,   # cold
            barrel_pct=0.10,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
        decay_lambda=decay_lambda,
    )

    # Weights: today=1.0, 2 days ago=0.95^2=0.9025
    # Expected EV = (100*1.0 + 90*0.9025) / (1.0 + 0.9025) = 95.25
    assert result is not None
    assert result.w_exit_velocity_avg > 90.0  # Should be closer to 100 than 90
    assert result.w_exit_velocity_avg < 100.0
    assert result.w_barrel_pct > 0.10
    assert result.w_barrel_pct < 0.20


def test_missing_statcast_data():
    """When no Statcast data exists, result still returns with base stats."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=2),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        [],  # No Statcast data
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.w_exit_velocity_avg is None
    assert result.w_barrel_pct is None
    assert result.w_xwoba is None


def test_statcast_no_matching_player():
    """Statcast rows for different player ID are ignored."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=2),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:99999",  # Different player
            game_date=as_of_date,
            exit_velocity_avg=95.0,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.w_exit_velocity_avg is None  # No match


def test_statcast_no_matching_date():
    """Statcast rows for different dates are ignored."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=2),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=date(2026, 3, 1),  # Different date, outside window
            exit_velocity_avg=95.0,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.w_exit_velocity_avg is None


def test_statcast_partial_metrics():
    """Player with only some Statcast metrics (e.g., no xwOBA)."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=2),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date,
            exit_velocity_avg=92.0,
            hard_hit_pct=0.40,
            # No xwoba, xba, xslg, woba
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.w_exit_velocity_avg == pytest.approx(92.0, abs=0.001)
    assert result.w_hard_hit_pct == pytest.approx(0.40, abs=0.001)
    assert result.w_xwoba is None  # No data
    assert result.w_xwoba_minus_woba is None  # Can't compute without xwOBA and wOBA


def test_no_bdl_data():
    """If no BDL data in window, returns None regardless of Statcast."""
    as_of_date = date(2026, 4, 10)

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date,
            exit_velocity_avg=95.0,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        [],  # No BDL data
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is None


# ---------------------------------------------------------------------------
# compute_all_rolling_windows_with_statcast tests
# ---------------------------------------------------------------------------

def test_batch_multiple_players():
    """Batch computation handles multiple players with mixed Statcast coverage."""
    as_of_date = date(2026, 4, 10)

    # Player 1: Has both BDL and Statcast
    bdl_p1 = [
        FakeBDLRow(bdl_player_id=111, game_date=as_of_date, ab=4, hits=2),
    ]
    sc_p1 = [
        StatcastDailyRow(
            player_id="mlbam:111",
            game_date=as_of_date,
            exit_velocity_avg=95.0,
            barrel_pct=0.15,
        ),
    ]

    # Player 2: BDL only, no Statcast
    bdl_p2 = [
        FakeBDLRow(bdl_player_id=222, game_date=as_of_date, ab=3, hits=1),
    ]

    # Player 3: Statcast only (no BDL, will be excluded)
    sc_p3 = [
        StatcastDailyRow(
            player_id="mlbam:333",
            game_date=as_of_date,
            exit_velocity_avg=90.0,
        ),
    ]

    all_bdl = bdl_p1 + bdl_p2
    all_sc = sc_p1 + sc_p3

    results = compute_all_rolling_windows_with_statcast(
        all_bdl,
        all_sc,
        as_of_date=as_of_date,
        window_sizes=[7],
    )

    assert len(results) == 2  # Players 1 and 2 only

    # Find results by player
    by_player = {r.bdl_player_id: r for r in results}

    assert 111 in by_player
    assert by_player[111].w_exit_velocity_avg == pytest.approx(95.0, abs=0.001)
    assert by_player[111].w_barrel_pct == pytest.approx(0.15, abs=0.001)

    assert 222 in by_player
    assert by_player[222].w_exit_velocity_avg is None  # No Statcast


def test_batch_multiple_windows():
    """Batch computation handles multiple window sizes."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(
            bdl_player_id=player_id,
            game_date=as_of_date - timedelta(days=i),
            ab=4,
            hits=2,
        )
        for i in range(5)
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date - timedelta(days=i),
            exit_velocity_avg=90.0 + i,
        )
        for i in range(5)
    ]

    results = compute_all_rolling_windows_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_sizes=[7, 14, 30],
    )

    assert len(results) == 3  # 3 windows
    by_window = {r.window_days: r for r in results}
    assert 7 in by_window
    assert 14 in by_window
    assert 30 in by_window

    # All windows should have Statcast data (5 games within all windows)
    for window in [7, 14, 30]:
        assert by_window[window].w_exit_velocity_avg is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_statcast_zero_metrics():
    """Statcast rows with all zero metrics are treated as missing."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(bdl_player_id=player_id, game_date=as_of_date, ab=4, hits=2),
    ]

    statcast_rows = [
        StatcastDailyRow(
            player_id="mlbam:12345",
            game_date=as_of_date,
            exit_velocity_avg=0.0,
            hard_hit_pct=0.0,
            barrel_pct=0.0,
            xwoba=0.0,
            xba=0.0,
            xslg=0.0,
            woba=0.0,
        ),
    ]

    result = compute_rolling_window_with_statcast(
        bdl_rows,
        statcast_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    # All metrics should be None because zeros are treated as missing
    assert result.w_exit_velocity_avg is None
    assert result.w_barrel_pct is None
    assert result.w_xwoba is None


def test_backward_compatibility():
    """compute_rolling_window (without Statcast) still works as before."""
    as_of_date = date(2026, 4, 10)
    player_id = 12345

    bdl_rows = [
        FakeBDLRow(
            bdl_player_id=player_id,
            game_date=as_of_date,
            ab=4,
            hits=2,
            home_runs=1,
            rbi=2,
        ),
    ]

    result = compute_rolling_window(
        bdl_rows,
        as_of_date=as_of_date,
        window_days=7,
    )

    assert result is not None
    assert result.bdl_player_id == player_id
    assert result.w_hits == 2.0
    assert result.w_home_runs == 1.0
    assert result.w_exit_velocity_avg is None  # No Statcast in base function
