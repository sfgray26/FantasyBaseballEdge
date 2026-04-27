"""
P14 Scoring Engine -- comprehensive pure-function tests.

All tests use plain Python objects (SimpleNamespace) as ORM row stubs.
Zero DB or I/O dependencies.
"""

import math
from datetime import date
from types import SimpleNamespace

import pytest

from backend.services.scoring_engine import (
    MIN_SAMPLE,
    Z_CAP,
    PlayerScoreResult,
    compute_league_zscores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AS_OF = date(2026, 4, 5)
WINDOW = 7


def _hitter(
    pid: int,
    hr: float = None,
    rbi: float = None,
    sb: float = None,
    avg: float = None,
    obp: float = None,
    games: int = 5,
) -> SimpleNamespace:
    """Build a hitter-style PlayerRollingStats stub (w_ip=None)."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=games,
        w_ab=15.0,          # non-None marks as hitter
        w_ip=None,
        w_home_runs=hr,
        w_rbi=rbi,
        w_stolen_bases=sb,
        w_avg=avg,
        w_obp=obp,
        # pitcher fields absent from hitters
        w_era=None,
        w_whip=None,
        w_k_per_9=None,
    )


def _pitcher(
    pid: int,
    era: float = None,
    whip: float = None,
    k9: float = None,
    games: int = 3,
) -> SimpleNamespace:
    """Build a pitcher-style PlayerRollingStats stub (w_ab=None)."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=games,
        w_ab=None,
        w_ip=5.0,           # non-None marks as pitcher
        w_home_runs=None,
        w_rbi=None,
        w_stolen_bases=None,
        w_avg=None,
        w_obp=None,
        w_era=era,
        w_whip=whip,
        w_k_per_9=k9,
    )


def _two_way(
    pid: int,
    hr: float = None,
    rbi: float = None,
    sb: float = None,
    avg: float = None,
    obp: float = None,
    era: float = None,
    whip: float = None,
    k9: float = None,
    games: int = 5,
) -> SimpleNamespace:
    """Build a two-way (Ohtani-style) stub (both w_ab and w_ip non-None)."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=games,
        w_ab=10.0,
        w_ip=3.0,
        w_home_runs=hr,
        w_rbi=rbi,
        w_stolen_bases=sb,
        w_avg=avg,
        w_obp=obp,
        w_era=era,
        w_whip=whip,
        w_k_per_9=k9,
    )


def _unknown(pid: int) -> SimpleNamespace:
    """Build a stub with no batting or pitching signal (should be excluded)."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=2,
        w_ab=None,
        w_ip=None,
        w_home_runs=None,
        w_rbi=None,
        w_stolen_bases=None,
        w_avg=None,
        w_obp=None,
        w_era=None,
        w_whip=None,
        w_k_per_9=None,
    )


def _make_hitter_pool(n: int, hr_values: list[float]) -> list[SimpleNamespace]:
    """Create n hitters with given HR values; other stats constant."""
    assert len(hr_values) == n
    return [
        _hitter(pid=i + 1, hr=hr_values[i], rbi=10.0, sb=2.0, avg=0.270, obp=0.340)
        for i in range(n)
    ]


def _result_for(results: list[PlayerScoreResult], pid: int) -> PlayerScoreResult:
    for r in results:
        if r.bdl_player_id == pid:
            return r
    raise KeyError(f"No result for pid={pid}")


# ===========================================================================
# 1. Below MIN_SAMPLE: all Z = None, composite = 0.0
# ===========================================================================

def test_single_hitter_all_categories_null_when_below_min_sample():
    """With only 1 player, no category has >= MIN_SAMPLE values. All Z = None."""
    rows = [_hitter(pid=1, hr=10.0, rbi=30.0, sb=5.0, avg=0.300, obp=0.370)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    assert len(results) == 1
    r = results[0]
    assert r.z_hr is None
    assert r.z_rbi is None
    assert r.z_sb is None
    assert r.z_avg is None
    assert r.z_obp is None
    assert r.composite_z == 0.0


# ===========================================================================
# 2. Identical values => std = 0 => category skipped
# ===========================================================================

def test_identical_values_have_zero_std_skipped():
    """All players with same HR => std=0 => z_hr=None for all."""
    rows = [_hitter(pid=i, hr=5.0, rbi=10.0, sb=2.0, avg=0.250, obp=0.320) for i in range(1, 8)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    assert len(results) == 7
    for r in results:
        # HR column is degenerate (all 5.0), so z_hr must be None
        assert r.z_hr is None, f"Expected z_hr=None for pid={r.bdl_player_id}, got {r.z_hr}"


# ===========================================================================
# 3. Below-average hitter has negative z_hr
# ===========================================================================

def test_below_average_hitter_has_negative_z():
    """Player with HR below the pool mean should have z_hr < 0."""
    # 6 players with HR=[10,10,10,10,10,1]: player 6 is below average
    rows = _make_hitter_pool(6, hr_values=[10.0, 10.0, 10.0, 10.0, 10.0, 1.0])
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)
    assert r.z_hr is not None
    assert r.z_hr < 0.0, f"Expected z_hr < 0, got {r.z_hr}"


# ===========================================================================
# 4. Above-average hitter has positive z_hr
# ===========================================================================

def test_above_average_hitter_has_positive_z():
    """Player with HR above the pool mean should have z_hr > 0."""
    rows = _make_hitter_pool(6, hr_values=[1.0, 1.0, 1.0, 1.0, 1.0, 20.0])
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)
    assert r.z_hr is not None
    assert r.z_hr > 0.0, f"Expected z_hr > 0, got {r.z_hr}"


# ===========================================================================
# 5. ERA: lower is better -> low ERA yields positive z_era
# ===========================================================================

def test_era_inverted_low_era_is_positive_z():
    """Pitcher with ERA below pool mean (better) should have z_era > 0."""
    # 6 pitchers with ERA=[5,5,5,5,5,1]: pitcher 6 has best ERA
    rows = [
        _pitcher(pid=i, era=5.0, whip=1.3, k9=8.0)
        for i in range(1, 6)
    ] + [_pitcher(pid=6, era=1.0, whip=1.3, k9=8.0)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)
    assert r.z_era is not None
    assert r.z_era > 0.0, f"Low ERA should give positive z_era, got {r.z_era}"


# ===========================================================================
# 6. WHIP: lower is better -> low WHIP yields positive z_whip
# ===========================================================================

def test_whip_inverted_low_whip_is_positive_z():
    """Pitcher with WHIP below pool mean should have z_whip > 0."""
    rows = [
        _pitcher(pid=i, era=3.5, whip=1.4, k9=8.0)
        for i in range(1, 6)
    ] + [_pitcher(pid=6, era=3.5, whip=0.8, k9=8.0)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)
    assert r.z_whip is not None
    assert r.z_whip > 0.0, f"Low WHIP should give positive z_whip, got {r.z_whip}"


# ===========================================================================
# 7 & 8. Z capped at +/- Z_CAP
# ===========================================================================

def test_z_capped_at_plus_3():
    """An extreme outlier above the mean should be capped at Z_CAP = 3.0.

    With population std, max achievable Z = sqrt(n-1). Need n >= 11 for sqrt(n-1) > 3.
    Use 11 normal hitters + 1 extreme outlier (n=12, max raw Z = sqrt(11) ~= 3.317 > 3.0).
    """
    rows = [_hitter(pid=i, hr=1.0, rbi=10.0, sb=2.0, avg=0.250, obp=0.320) for i in range(1, 12)]
    rows.append(_hitter(pid=12, hr=1_000_000.0, rbi=10.0, sb=2.0, avg=0.250, obp=0.320))
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=12)
    assert r.z_hr is not None
    assert r.z_hr == Z_CAP, f"z_hr should be capped at {Z_CAP}, got {r.z_hr}"


def test_z_capped_at_minus_3():
    """An extreme outlier below the mean should be capped at -Z_CAP = -3.0.

    Same reasoning: need n >= 12 for raw Z < -3.0 to be possible.
    """
    rows = [_hitter(pid=i, hr=100.0, rbi=50.0, sb=20.0, avg=0.350, obp=0.450) for i in range(1, 12)]
    rows.append(_hitter(pid=12, hr=0.000001, rbi=50.0, sb=20.0, avg=0.350, obp=0.450))
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=12)
    assert r.z_hr is not None
    assert r.z_hr == -Z_CAP, f"z_hr should be capped at -{Z_CAP}, got {r.z_hr}"


# ===========================================================================
# 9. composite_z is mean of applicable non-None Z-scores
# ===========================================================================

def test_composite_z_is_mean_of_applicable():
    """
    Hitter with 5 non-None category Z-scores -> composite = mean of those 5.
    Uses controlled values so we can predict composite_z precisely.
    """
    # 6 identical hitters except pid=6 has different values in all categories
    base = [
        _hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)
        for i in range(1, 6)
    ]
    # pid=6: above average in all categories -- should have positive composite
    outlier = _hitter(pid=6, hr=15.0, rbi=50.0, sb=10.0, avg=0.400, obp=0.500)
    rows = base + [outlier]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)

    # All five hitter categories should be non-None
    z_vals = [r.z_hr, r.z_rbi, r.z_sb, r.z_avg, r.z_obp]
    assert all(v is not None for v in z_vals), f"Expected all 5 hitter Z-scores, got {z_vals}"

    expected_composite = sum(z_vals) / len(z_vals)
    assert math.isclose(r.composite_z, expected_composite, rel_tol=1e-9), (
        f"composite_z mismatch: {r.composite_z} != {expected_composite}"
    )


# ===========================================================================
# 10. confidence formula
# ===========================================================================

def test_confidence_formula():
    """5 games in a 7-day window -> confidence = 5/7."""
    rows = [_hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330, games=5)
            for i in range(1, 7)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    for r in results:
        expected = 5 / 7
        assert math.isclose(r.confidence, expected, rel_tol=1e-9), (
            f"confidence mismatch: {r.confidence} != {expected}"
        )


def test_confidence_capped_at_1():
    """10 games in a 7-day window -> confidence capped at 1.0."""
    rows = [_hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330, games=10)
            for i in range(1, 7)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    for r in results:
        assert r.confidence == 1.0, f"confidence should be capped at 1.0, got {r.confidence}"


# ===========================================================================
# 11 & 12. score_0_100 extremes
# ===========================================================================

def test_score_0_100_best_player_near_100():
    """Best composite_z in a hitter pool should receive score near 100."""
    # 5 hitters, pid=5 has the highest composite_z
    rows = [
        _hitter(pid=1, hr=1.0,  rbi=5.0,  sb=1.0, avg=0.220, obp=0.290),
        _hitter(pid=2, hr=3.0,  rbi=10.0, sb=2.0, avg=0.240, obp=0.310),
        _hitter(pid=3, hr=5.0,  rbi=15.0, sb=3.0, avg=0.260, obp=0.330),
        _hitter(pid=4, hr=8.0,  rbi=25.0, sb=4.0, avg=0.280, obp=0.360),
        _hitter(pid=5, hr=20.0, rbi=60.0, sb=15.0, avg=0.380, obp=0.460),
    ]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    # Sort by composite_z to find best
    best = max(results, key=lambda r: r.composite_z)
    assert best.score_0_100 == 100.0, (
        f"Best player should have score_0_100=100.0, got {best.score_0_100}"
    )


def test_score_0_100_worst_player_near_0():
    """Worst composite_z in a hitter pool should receive the lowest score."""
    rows = [
        _hitter(pid=1, hr=0.5,  rbi=2.0,  sb=0.5, avg=0.200, obp=0.260),
        _hitter(pid=2, hr=3.0,  rbi=10.0, sb=2.0, avg=0.250, obp=0.320),
        _hitter(pid=3, hr=5.0,  rbi=15.0, sb=3.0, avg=0.265, obp=0.335),
        _hitter(pid=4, hr=8.0,  rbi=25.0, sb=5.0, avg=0.285, obp=0.355),
        _hitter(pid=5, hr=12.0, rbi=40.0, sb=8.0, avg=0.320, obp=0.400),
    ]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    worst = min(results, key=lambda r: r.composite_z)
    # Worst player's score should be the minimum in the group
    min_score = min(r.score_0_100 for r in results)
    assert worst.score_0_100 == min_score
    # With 5 players, the worst has 1 player at or below them (themselves)
    # percentile = 1/5 * 100 = 20.0
    assert worst.score_0_100 == 20.0, (
        f"Worst player score_0_100 should be 20.0, got {worst.score_0_100}"
    )


# ===========================================================================
# 13. Pitcher gets only pitcher categories
# ===========================================================================

def test_pitcher_gets_pitcher_categories_only():
    """Pitcher row -> hitter Z-scores (z_hr etc.) all None; pitcher Z-scores computed."""
    rows = [
        _pitcher(pid=i, era=3.5, whip=1.2, k9=9.0)
        for i in range(1, 7)
    ]
    # Give one pitcher distinct values so categories aren't degenerate
    rows[0] = _pitcher(pid=1, era=2.0, whip=0.9, k9=11.0)

    results = compute_league_zscores(rows, AS_OF, WINDOW)
    for r in results:
        assert r.player_type == "pitcher"
        # Hitter categories must be None
        assert r.z_hr is None,  f"pid={r.bdl_player_id}: z_hr should be None for pitcher"
        assert r.z_rbi is None, f"pid={r.bdl_player_id}: z_rbi should be None for pitcher"
        assert r.z_sb is None,  f"pid={r.bdl_player_id}: z_sb should be None for pitcher"
        assert r.z_avg is None, f"pid={r.bdl_player_id}: z_avg should be None for pitcher"
        assert r.z_obp is None, f"pid={r.bdl_player_id}: z_obp should be None for pitcher"


# ===========================================================================
# 14. Hitter gets only hitter categories
# ===========================================================================

def test_hitter_gets_hitter_categories_only():
    """Hitter row -> pitcher Z-scores (z_era etc.) all None; hitter Z-scores computed."""
    rows = [
        _hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)
        for i in range(1, 7)
    ]
    rows[0] = _hitter(pid=1, hr=15.0, rbi=40.0, sb=8.0, avg=0.340, obp=0.420)

    results = compute_league_zscores(rows, AS_OF, WINDOW)
    for r in results:
        assert r.player_type == "hitter"
        # Pitcher categories must be None
        assert r.z_era is None,     f"pid={r.bdl_player_id}: z_era should be None for hitter"
        assert r.z_whip is None,    f"pid={r.bdl_player_id}: z_whip should be None for hitter"
        assert r.z_k_per_9 is None, f"pid={r.bdl_player_id}: z_k_per_9 should be None for hitter"


# ===========================================================================
# 15. Two-way player gets all categories
# ===========================================================================

def test_two_way_player_gets_all_categories():
    """Two-way row -> both hitter and pitcher Z-scores are computed (where sample permits)."""
    # Build a pool large enough: 3 two-way players + 2 hitters + 2 pitchers
    # But for simplicity, just use 6 two-way players with varying stats
    rows = [
        _two_way(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330,
                 era=3.5, whip=1.2, k9=9.0)
        for i in range(1, 6)
    ]
    rows.append(
        _two_way(pid=6, hr=12.0, rbi=40.0, sb=8.0, avg=0.330, obp=0.410,
                 era=2.0, whip=0.90, k9=12.0)
    )
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=6)
    assert r.player_type == "two_way"
    # Best player in all categories should have positive Z for all categories
    # (ERA/WHIP are inverted so lower = positive Z, which pid=6 has)
    # All categories should be non-None
    all_z = [r.z_hr, r.z_rbi, r.z_sb, r.z_avg, r.z_obp, r.z_era, r.z_whip, r.z_k_per_9]
    assert all(v is not None for v in all_z), (
        f"Two-way player should have all 8 Z-scores, got: {all_z}"
    )


# ===========================================================================
# 16. Unknown player type is excluded
# ===========================================================================

def test_unknown_player_type_excluded():
    """Row with w_ab=None and w_ip=None -> not included in results."""
    rows = [
        _hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)
        for i in range(1, 7)
    ]
    rows.append(_unknown(pid=99))
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    pids = {r.bdl_player_id for r in results}
    assert 99 not in pids, "Unknown player type should be excluded from results"
    assert len(results) == 6


# ===========================================================================
# 17. score_0_100 percentile ranking at median
# ===========================================================================

def test_score_0_100_percentile_ranking():
    """Player at the exact median of a 5-player pool should have score near 60.0."""
    # 5 hitters with distinct composite_z values [low, low, mid, high, high]
    # Player at position 3 (mid) has 3/5 = 60.0 percentile rank (weak ordering)
    rows = [
        _hitter(pid=1, hr=1.0,  rbi=5.0,  sb=1.0, avg=0.220, obp=0.290),
        _hitter(pid=2, hr=3.0,  rbi=10.0, sb=2.0, avg=0.240, obp=0.310),
        _hitter(pid=3, hr=5.0,  rbi=15.0, sb=3.0, avg=0.260, obp=0.330),  # median
        _hitter(pid=4, hr=8.0,  rbi=25.0, sb=4.0, avg=0.280, obp=0.360),
        _hitter(pid=5, hr=15.0, rbi=50.0, sb=10.0, avg=0.360, obp=0.440),
    ]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    # Sort results by composite_z to find median player
    sorted_results = sorted(results, key=lambda r: r.composite_z)
    median_r = sorted_results[2]  # index 2 of 5 = median

    # Median player: 3 players at or below them -> 3/5 * 100 = 60.0
    assert median_r.score_0_100 == 60.0, (
        f"Median player score_0_100 should be 60.0, got {median_r.score_0_100}"
    )


# ===========================================================================
# 18. Empty input returns empty list
# ===========================================================================

def test_empty_input_returns_empty_list():
    """compute_league_zscores with no rows returns an empty list."""
    results = compute_league_zscores([], AS_OF, WINDOW)
    assert results == []


# ===========================================================================
# 19. Single-player cohort gets score_0_100 = 50.0
# ===========================================================================

def test_single_player_in_cohort_gets_50():
    """When only one player of a type exists, score_0_100 = 50.0."""
    # Put 6 hitters + 1 pitcher; the lone pitcher gets 50.0
    hitters = [
        _hitter(pid=i, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)
        for i in range(1, 7)
    ]
    lone_pitcher = _pitcher(pid=99, era=3.5, whip=1.2, k9=9.0)
    rows = hitters + [lone_pitcher]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    r = _result_for(results, pid=99)
    assert r.score_0_100 == 50.0, (
        f"Single-player cohort should get score_0_100=50.0, got {r.score_0_100}"
    )


# ===========================================================================
# 20. composite_z = 0.0 when all applicable Z-scores are None
# ===========================================================================

def test_composite_z_is_zero_when_all_z_none():
    """When no category reaches MIN_SAMPLE, composite_z = 0.0."""
    # Only 1 hitter: all Z = None, composite must be 0.0
    rows = [_hitter(pid=1, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)]
    results = compute_league_zscores(rows, AS_OF, WINDOW)
    assert len(results) == 1
    assert results[0].composite_z == 0.0


# ===========================================================================
# 21. Player type detection
# ===========================================================================

def test_player_type_detected_correctly():
    """player_type field is set correctly for hitter, pitcher, and two_way."""
    hitter = _hitter(pid=1, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330)
    pitcher = _pitcher(pid=2, era=3.5, whip=1.2, k9=9.0)
    two_way = _two_way(pid=3, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330,
                       era=3.5, whip=1.2, k9=9.0)

    # Use a pool of 6 to satisfy MIN_SAMPLE -- mix all types
    rows = (
        [_hitter(pid=i + 10, hr=5.0, rbi=20.0, sb=3.0, avg=0.260, obp=0.330) for i in range(5)]
        + [hitter]
        + [_pitcher(pid=i + 20, era=3.5, whip=1.2, k9=9.0) for i in range(5)]
        + [pitcher]
        + [two_way]
    )
    results = compute_league_zscores(rows, AS_OF, WINDOW)

    r_hitter = _result_for(results, pid=1)
    r_pitcher = _result_for(results, pid=2)
    r_two_way = _result_for(results, pid=3)

    assert r_hitter.player_type == "hitter"
    assert r_pitcher.player_type == "pitcher"
    assert r_two_way.player_type == "two_way"
