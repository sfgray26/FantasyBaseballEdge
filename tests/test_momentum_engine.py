"""
Tests for backend/services/momentum_engine.py (P15 Momentum Layer).

Covers:
  - classify_signal boundary semantics (all 5 signals + exact boundaries)
  - MomentumResult field computation (delta_z, confidence, player_type)
  - compute_all_momentum pairing logic (skip missing 14d / missing 30d)
  - End-to-end SURGING and COLLAPSING player scenarios
"""

import pytest
from datetime import date
from types import SimpleNamespace

from backend.services.momentum_engine import (
    classify_signal,
    compute_player_momentum,
    compute_all_momentum,
    SURGING,
    HOT,
    STABLE,
    COLD,
    COLLAPSING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score(bdl_player_id=1, as_of_date=None, window_days=14,
                player_type="hitter", composite_z=0.0, score_0_100=50.0,
                confidence=0.8, games_in_window=10):
    """Construct a minimal mock PlayerScore-like namespace."""
    return SimpleNamespace(
        bdl_player_id=bdl_player_id,
        as_of_date=as_of_date or date(2026, 4, 5),
        window_days=window_days,
        player_type=player_type,
        composite_z=composite_z,
        score_0_100=score_0_100,
        confidence=confidence,
        games_in_window=games_in_window,
    )


# ---------------------------------------------------------------------------
# classify_signal -- core signal mapping
# ---------------------------------------------------------------------------

def test_classify_signal_surging():
    assert classify_signal(0.6) == SURGING


def test_classify_signal_hot():
    assert classify_signal(0.35) == HOT


def test_classify_signal_stable_positive():
    assert classify_signal(0.1) == STABLE


def test_classify_signal_stable_zero():
    assert classify_signal(0.0) == STABLE


def test_classify_signal_stable_negative():
    assert classify_signal(-0.1) == STABLE


def test_classify_signal_cold():
    assert classify_signal(-0.35) == COLD


def test_classify_signal_collapsing():
    assert classify_signal(-0.6) == COLLAPSING


# ---------------------------------------------------------------------------
# classify_signal -- exact boundary semantics
# ---------------------------------------------------------------------------

def test_boundary_0_5_is_hot():
    """delta_z == 0.5 must return HOT (not SURGING; strictly > 0.5 for SURGING)."""
    assert classify_signal(0.5) == HOT


def test_boundary_neg_0_5_is_cold():
    """delta_z == -0.5 must return COLD (not COLLAPSING; strictly < -0.5 for COLLAPSING)."""
    assert classify_signal(-0.5) == COLD


def test_boundary_0_2_is_hot():
    """delta_z == 0.2 must return HOT (not STABLE; >= 0.2 triggers HOT)."""
    assert classify_signal(0.2) == HOT


def test_boundary_neg_0_2_is_cold():
    """delta_z == -0.2 must return COLD (not STABLE; >= -0.5 and not > -0.2)."""
    assert classify_signal(-0.2) == COLD


# ---------------------------------------------------------------------------
# compute_player_momentum -- field correctness
# ---------------------------------------------------------------------------

def test_delta_z_is_14d_minus_30d():
    s14 = _make_score(window_days=14, composite_z=1.2)
    s30 = _make_score(window_days=30, composite_z=0.8)
    result = compute_player_momentum(s14, s30)
    assert abs(result.delta_z - 0.4) < 1e-9


def test_confidence_is_min_of_two():
    s14 = _make_score(window_days=14, confidence=0.9)
    s30 = _make_score(window_days=30, confidence=0.6)
    result = compute_player_momentum(s14, s30)
    assert result.confidence == 0.6


def test_confidence_is_min_of_two_reversed():
    """Verify min works regardless of which window has lower confidence."""
    s14 = _make_score(window_days=14, confidence=0.4)
    s30 = _make_score(window_days=30, confidence=0.95)
    result = compute_player_momentum(s14, s30)
    assert result.confidence == 0.4


def test_player_type_from_14d_row():
    """player_type on the result must come from the 14d row, not the 30d row."""
    s14 = _make_score(window_days=14, player_type="pitcher")
    s30 = _make_score(window_days=30, player_type="hitter")
    result = compute_player_momentum(s14, s30)
    assert result.player_type == "pitcher"


# ---------------------------------------------------------------------------
# compute_all_momentum -- pairing and skip logic
# ---------------------------------------------------------------------------

def test_skip_player_missing_14d():
    """Player only in 30d list must be skipped."""
    s30 = _make_score(bdl_player_id=99, window_days=30)
    results = compute_all_momentum([], [s30])
    assert results == []


def test_skip_player_missing_30d():
    """Player only in 14d list must be skipped."""
    s14 = _make_score(bdl_player_id=99, window_days=14)
    results = compute_all_momentum([s14], [])
    assert results == []


def test_compute_all_momentum_pairs_correctly():
    """Two players both present in both lists -> two results."""
    s14_a = _make_score(bdl_player_id=1, window_days=14, composite_z=0.5)
    s30_a = _make_score(bdl_player_id=1, window_days=30, composite_z=0.1)
    s14_b = _make_score(bdl_player_id=2, window_days=14, composite_z=-0.3)
    s30_b = _make_score(bdl_player_id=2, window_days=30, composite_z=0.2)
    results = compute_all_momentum([s14_a, s14_b], [s30_a, s30_b])
    assert len(results) == 2
    player_ids = {r.bdl_player_id for r in results}
    assert player_ids == {1, 2}


def test_compute_all_momentum_partial_overlap():
    """Player 2 only in 14d -> only player 1 in results."""
    s14_a = _make_score(bdl_player_id=1, window_days=14, composite_z=1.0)
    s30_a = _make_score(bdl_player_id=1, window_days=30, composite_z=0.3)
    s14_b = _make_score(bdl_player_id=2, window_days=14, composite_z=0.5)
    results = compute_all_momentum([s14_a, s14_b], [s30_a])
    assert len(results) == 1
    assert results[0].bdl_player_id == 1


# ---------------------------------------------------------------------------
# End-to-end signal scenarios
# ---------------------------------------------------------------------------

def test_surging_player_classified_correctly():
    """Player with composite_z rising strongly -> SURGING."""
    s14 = _make_score(bdl_player_id=10, window_days=14, composite_z=2.0, score_0_100=92.0)
    s30 = _make_score(bdl_player_id=10, window_days=30, composite_z=1.3, score_0_100=78.0)
    results = compute_all_momentum([s14], [s30])
    assert len(results) == 1
    r = results[0]
    assert r.signal == SURGING
    assert abs(r.delta_z - 0.7) < 1e-9
    assert r.score_14d == 92.0
    assert r.score_30d == 78.0


def test_collapsing_player_classified_correctly():
    """Player with composite_z dropping sharply -> COLLAPSING."""
    s14 = _make_score(bdl_player_id=20, window_days=14, composite_z=-1.5, score_0_100=8.0)
    s30 = _make_score(bdl_player_id=20, window_days=30, composite_z=-0.8, score_0_100=22.0)
    results = compute_all_momentum([s14], [s30])
    assert len(results) == 1
    r = results[0]
    assert r.signal == COLLAPSING
    assert abs(r.delta_z - (-0.7)) < 1e-9


def test_momentum_result_as_of_date_from_14d():
    """as_of_date on result comes from the 14d score row."""
    target_date = date(2026, 4, 5)
    s14 = _make_score(window_days=14, as_of_date=target_date)
    s30 = _make_score(window_days=30, as_of_date=target_date)
    result = compute_player_momentum(s14, s30)
    assert result.as_of_date == target_date


def test_momentum_result_composite_z_fields():
    """composite_z_14d and composite_z_30d must be stored verbatim."""
    s14 = _make_score(window_days=14, composite_z=1.75)
    s30 = _make_score(window_days=30, composite_z=0.95)
    result = compute_player_momentum(s14, s30)
    assert result.composite_z_14d == 1.75
    assert result.composite_z_30d == 0.95
