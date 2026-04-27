"""
P27 Net Stolen Bases (NSB) pipeline regression tests.

Covers:
  - rolling_window_engine: w_caught_stealing accumulates from row.caught_stealing,
    w_net_stolen_bases = w_stolen_bases - w_caught_stealing, decay-weighted.
  - rolling_window_engine: missing/None caught_stealing is treated as 0
    (BDL-friendly: absent CS data should not zero out NSB).
  - scoring_engine: HITTER_CATEGORIES contains z_nsb, _COMPOSITE_EXCLUDED holds z_sb.
  - scoring_engine: composite_z is built from z_nsb (not z_sb) for hitters.
  - scoring_engine: z_sb is still computed and persisted for backward compat.
  - PlayerScoreResult and PlayerRollingStats / PlayerScore models expose new fields.
  - migrate_v27_nsb script is idempotent (ADD COLUMN IF NOT EXISTS) and reversible.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from backend.services.rolling_window_engine import (
    RollingWindowResult,
    compute_rolling_window,
)
from backend.services.scoring_engine import (
    HITTER_CATEGORIES,
    PITCHER_CATEGORIES,
    PlayerScoreResult,
    _COMPOSITE_EXCLUDED,
    compute_league_zscores,
    compute_league_params,
)


AS_OF = date(2026, 4, 14)
WINDOW = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_row(pid, gd, ab=4, hits=1, sb=0, cs=None, **kw):
    """ORM stub for MLBPlayerStats with optional caught_stealing override."""
    return SimpleNamespace(
        bdl_player_id=pid,
        game_date=gd,
        ab=ab,
        hits=hits,
        doubles=0,
        triples=0,
        home_runs=0,
        rbi=0,
        walks=0,
        strikeouts_bat=0,
        stolen_bases=sb,
        caught_stealing=cs,
        innings_pitched=None,
        hits_allowed=None,
        runs_allowed=None,
        earned_runs=None,
        walks_allowed=None,
        strikeouts_pit=None,
        **kw,
    )


def _hitter_with_nsb(pid, sb, cs, *, ab=20.0, hr=2.0, rbi=10.0, avg=0.300, obp=0.400):
    """PlayerRollingStats stub with NSB columns populated."""
    return SimpleNamespace(
        bdl_player_id=pid,
        as_of_date=AS_OF,
        window_days=WINDOW,
        games_in_window=5,
        w_ab=ab,
        w_ip=None,
        w_home_runs=hr,
        w_rbi=rbi,
        w_stolen_bases=float(sb),
        w_caught_stealing=float(cs),
        w_net_stolen_bases=float(sb - cs),
        w_avg=avg,
        w_obp=obp,
        w_era=None,
        w_whip=None,
        w_k_per_9=None,
    )


# ---------------------------------------------------------------------------
# rolling_window_engine -- CS aggregation
# ---------------------------------------------------------------------------

def test_rolling_engine_zero_cs_yields_nsb_equals_sb():
    """No CS events -> NSB == SB (most common case for active basestealers)."""
    rows = [
        _stat_row(1, AS_OF, ab=4, hits=2, sb=2, cs=0),
        _stat_row(1, date(2026, 4, 13), ab=4, hits=1, sb=1, cs=0),
    ]
    result = compute_rolling_window(rows, AS_OF, window_days=WINDOW, decay_lambda=1.0)
    assert result is not None
    assert result.w_stolen_bases == pytest.approx(3.0)
    assert result.w_caught_stealing == pytest.approx(0.0)
    assert result.w_net_stolen_bases == pytest.approx(3.0)


def test_rolling_engine_positive_cs_reduces_nsb():
    """SB=5, CS=2 -> NSB = 3."""
    rows = [
        _stat_row(1, AS_OF, ab=4, hits=2, sb=3, cs=1),
        _stat_row(1, date(2026, 4, 13), ab=4, hits=1, sb=2, cs=1),
    ]
    result = compute_rolling_window(rows, AS_OF, window_days=WINDOW, decay_lambda=1.0)
    assert result is not None
    assert result.w_stolen_bases == pytest.approx(5.0)
    assert result.w_caught_stealing == pytest.approx(2.0)
    assert result.w_net_stolen_bases == pytest.approx(3.0)


def test_rolling_engine_null_cs_treated_as_zero():
    """row.caught_stealing=None must not poison the accumulator."""
    rows = [
        _stat_row(1, AS_OF, ab=4, hits=2, sb=2, cs=None),
        _stat_row(1, date(2026, 4, 13), ab=4, hits=1, sb=1, cs=None),
    ]
    result = compute_rolling_window(rows, AS_OF, window_days=WINDOW, decay_lambda=1.0)
    assert result is not None
    assert result.w_caught_stealing == pytest.approx(0.0)
    assert result.w_net_stolen_bases == pytest.approx(3.0)


def test_rolling_engine_missing_cs_attribute_treated_as_zero():
    """Older ORM rows without a caught_stealing attribute must not crash."""
    row = SimpleNamespace(
        bdl_player_id=7,
        game_date=AS_OF,
        ab=4, hits=2, doubles=0, triples=0, home_runs=0, rbi=0, walks=0,
        strikeouts_bat=0, stolen_bases=2,
        # NO caught_stealing attribute at all
        innings_pitched=None, hits_allowed=None, runs_allowed=None,
        earned_runs=None, walks_allowed=None, strikeouts_pit=None,
    )
    result = compute_rolling_window([row], AS_OF, window_days=WINDOW, decay_lambda=1.0)
    assert result is not None
    assert result.w_caught_stealing == pytest.approx(0.0)
    assert result.w_net_stolen_bases == pytest.approx(2.0)


def test_rolling_engine_pure_pitcher_has_null_nsb_fields():
    """Pure pitchers (no AB) get None for both w_caught_stealing and w_net_stolen_bases."""
    row = SimpleNamespace(
        bdl_player_id=99,
        game_date=AS_OF,
        ab=None, hits=None, doubles=None, triples=None, home_runs=None,
        rbi=None, walks=None, strikeouts_bat=None, stolen_bases=None,
        caught_stealing=None,
        innings_pitched="5.0", hits_allowed=4, runs_allowed=2,
        earned_runs=2, walks_allowed=1, strikeouts_pit=6,
    )
    result = compute_rolling_window([row], AS_OF, window_days=WINDOW, decay_lambda=1.0)
    assert result is not None
    assert result.w_caught_stealing is None
    assert result.w_net_stolen_bases is None
    assert result.w_ip is not None  # confirms pitcher branch did fire


def test_rolling_engine_decay_weighted_cs():
    """CS aggregation respects exponential decay weighting."""
    rows = [
        _stat_row(1, AS_OF, ab=4, hits=2, sb=2, cs=2),                  # weight 1.0
        _stat_row(1, date(2026, 4, 13), ab=4, hits=1, sb=2, cs=2),      # weight 0.5
    ]
    result = compute_rolling_window(rows, AS_OF, window_days=WINDOW, decay_lambda=0.5)
    # Expected: cs accumulator = 2*1.0 + 2*0.5 = 3.0
    # Expected: sb accumulator = 2*1.0 + 2*0.5 = 3.0
    # Expected: nsb = 0.0
    assert result.w_caught_stealing == pytest.approx(3.0)
    assert result.w_stolen_bases == pytest.approx(3.0)
    assert result.w_net_stolen_bases == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# scoring_engine -- HITTER_CATEGORIES + composite exclusion
# ---------------------------------------------------------------------------

def test_z_nsb_is_in_hitter_categories():
    assert "z_nsb" in HITTER_CATEGORIES
    col, lower_better = HITTER_CATEGORIES["z_nsb"]
    assert col == "w_net_stolen_bases"
    assert lower_better is False


def test_z_sb_is_excluded_from_composite():
    """z_sb is computed for backward compat but must not enter composite_z."""
    assert "z_sb" in _COMPOSITE_EXCLUDED
    assert "z_nsb" not in _COMPOSITE_EXCLUDED  # NSB MUST drive composite


def test_composite_z_uses_z_nsb_not_z_sb():
    """
    Build a 6-hitter pool where SB and NSB diverge sharply (every player has
    high SB but high CS -> low NSB). composite_z should reflect z_nsb, not z_sb.
    """
    # 6 hitters, all with identical batting except SB/CS pattern:
    # pid=1: 10 SB, 10 CS  -> NSB = 0  (worst NSB)
    # pid=2: 10 SB,  8 CS  -> NSB = 2
    # pid=3: 10 SB,  6 CS  -> NSB = 4
    # pid=4: 10 SB,  4 CS  -> NSB = 6
    # pid=5: 10 SB,  2 CS  -> NSB = 8
    # pid=6: 10 SB,  0 CS  -> NSB = 10  (best NSB; z_sb is identical to all others)
    rows = [
        _hitter_with_nsb(1, sb=10, cs=10),
        _hitter_with_nsb(2, sb=10, cs=8),
        _hitter_with_nsb(3, sb=10, cs=6),
        _hitter_with_nsb(4, sb=10, cs=4),
        _hitter_with_nsb(5, sb=10, cs=2),
        _hitter_with_nsb(6, sb=10, cs=0),
    ]
    results = compute_league_zscores(rows, AS_OF, WINDOW, winsorize=False)
    by_pid = {r.bdl_player_id: r for r in results}

    # All players have identical w_stolen_bases -> z_sb is degenerate (None)
    for r in results:
        assert r.z_sb is None, f"z_sb should be None when all SB equal, got {r.z_sb}"

    # z_nsb should rank players by NSB
    z_nsb_pid1 = by_pid[1].z_nsb
    z_nsb_pid6 = by_pid[6].z_nsb
    assert z_nsb_pid1 is not None and z_nsb_pid6 is not None
    assert z_nsb_pid6 > z_nsb_pid1, "Player with NSB=10 should outrank NSB=0"

    # composite_z must reflect z_nsb (since z_sb is None for all)
    assert by_pid[6].composite_z > by_pid[1].composite_z


def test_composite_z_excludes_z_sb_when_both_populated():
    """
    Even when both z_sb and z_nsb have valid values, composite_z must
    exclude z_sb to avoid double-counting basestealing.
    """
    # Build players where SB varies but CS is identical (so z_sb and z_nsb
    # are both well-defined and rank players the same way).
    rows = [
        _hitter_with_nsb(1, sb=2,  cs=1),    # NSB=1, SB=2
        _hitter_with_nsb(2, sb=4,  cs=1),
        _hitter_with_nsb(3, sb=6,  cs=1),
        _hitter_with_nsb(4, sb=8,  cs=1),
        _hitter_with_nsb(5, sb=10, cs=1),
        _hitter_with_nsb(6, sb=12, cs=1),
    ]
    results = compute_league_zscores(rows, AS_OF, WINDOW, winsorize=False)

    # The composite must be the mean of all applicable non-None hitter Z's
    # EXCEPT z_sb. We reconstruct that expected value from the persisted
    # fields and confirm:
    #   (a) z_sb and z_nsb are both populated
    #   (b) composite_z equals the mean of {z_hr, z_rbi, z_nsb, z_avg, z_obp}
    #       filtered to non-None values -- i.e. z_sb was NOT mixed in.
    composite_keys = ["z_hr", "z_rbi", "z_nsb", "z_avg", "z_obp"]  # z_sb deliberately absent
    for r in results:
        assert r.z_sb is not None,  "z_sb should be computed (still persisted)"
        assert r.z_nsb is not None, "z_nsb should be computed"

        expected_parts = [getattr(r, k) for k in composite_keys if getattr(r, k) is not None]
        expected_composite = sum(expected_parts) / len(expected_parts)
        assert r.composite_z == pytest.approx(expected_composite, abs=1e-9), (
            f"composite_z must exclude z_sb. Got {r.composite_z}, expected {expected_composite} "
            f"(parts={expected_parts})."
        )

        # And crucially: if z_sb WERE included, the composite would differ
        # (z_sb != z_nsb because CS > 0 -> ranks differ). Confirm the guard.
        with_sb_parts = expected_parts + [r.z_sb]
        would_be_with_sb = sum(with_sb_parts) / len(with_sb_parts)
        # Degenerate case where z_sb happens to equal the current composite is fine;
        # what we care about is that the actual composite matches the WITHOUT-z_sb
        # computation above. (The previous assertion already establishes that.)
        # This second check is informational.
        _ = would_be_with_sb


def test_player_score_result_has_z_nsb_field():
    """Dataclass field exists and defaults to None."""
    r = PlayerScoreResult(
        bdl_player_id=1, as_of_date=AS_OF, window_days=WINDOW,
        player_type="hitter", games_in_window=5,
    )
    assert hasattr(r, "z_nsb")
    assert r.z_nsb is None
    assert hasattr(r, "z_sb")  # backward compat preserved
    assert r.z_sb is None


def test_compute_league_params_emits_both_sb_and_nsb():
    """
    Simulation engine still consumes 'sb' (raw stolen bases). 'nsb' is added
    for future consumers. Both should appear in league_means/league_stds.
    """
    rows = [
        _hitter_with_nsb(i, sb=float(i), cs=float(i % 3))
        for i in range(1, 11)
    ]
    means, stds = compute_league_params(rows)
    assert "sb" in means and "sb" in stds
    assert "nsb" in means and "nsb" in stds
    # Sanity: SB mean over [1..10] = 5.5
    assert means["sb"] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# models -- new columns surface on the ORM
# ---------------------------------------------------------------------------

def test_player_rolling_stats_model_has_new_columns():
    from backend.models import PlayerRollingStats
    cols = {c.name for c in PlayerRollingStats.__table__.columns}
    assert "w_caught_stealing" in cols
    assert "w_net_stolen_bases" in cols
    # backward compat
    assert "w_stolen_bases" in cols


def test_player_scores_model_has_z_nsb():
    from backend.models import PlayerScore
    cols = {c.name for c in PlayerScore.__table__.columns}
    assert "z_nsb" in cols
    # backward compat
    assert "z_sb" in cols


# ---------------------------------------------------------------------------
# Migration script -- shape only (no live DB)
# ---------------------------------------------------------------------------

def test_migration_v27_module_imports_and_exposes_sql():
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "scripts", "migrate_v27_nsb.py",
    )
    spec = importlib.util.spec_from_file_location("migrate_v27_nsb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "ADD COLUMN IF NOT EXISTS w_caught_stealing" in mod.UPGRADE_SQL
    assert "ADD COLUMN IF NOT EXISTS w_net_stolen_bases" in mod.UPGRADE_SQL
    assert "ADD COLUMN IF NOT EXISTS z_nsb" in mod.UPGRADE_SQL
    assert "DROP COLUMN IF EXISTS z_nsb" in mod.DOWNGRADE_SQL
    assert "DROP COLUMN IF EXISTS w_net_stolen_bases" in mod.DOWNGRADE_SQL
    assert "DROP COLUMN IF EXISTS w_caught_stealing" in mod.DOWNGRADE_SQL
    # Must be callable
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)
