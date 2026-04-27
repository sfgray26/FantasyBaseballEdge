"""
Smoke tests for admin_scoring_diagnostics (P27 NSB rollout verification).

Mirrors the structure of test_admin_statcast_diagnostics: verify the module
imports, endpoints register under expected paths, SQL references real model
columns, and helpers/whitelists behave as documented. No DB required.
"""

import inspect
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from backend import admin_scoring_diagnostics as diag
from backend.models import (
    PlayerRollingStats,
    PlayerScore,
    PlayerIDMapping,
    DataIngestionLog,
    Base,
)


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------

EXPECTED_PATHS = {
    "/diagnose-scoring/nsb-rollout",
    "/diagnose-scoring/nsb-leaders",
    "/diagnose-scoring/nsb-player",
    "/diagnose-scoring/layer3-freshness",
    "/diagnose-decision/pipeline-freshness",
}


def test_all_endpoints_registered():
    registered = {r.path for r in diag.router.routes}
    missing = EXPECTED_PATHS - registered
    assert not missing, f"Missing expected endpoints: {missing}"


def test_all_endpoints_are_get():
    for r in diag.router.routes:
        if r.path in EXPECTED_PATHS:
            assert "GET" in r.methods, f"{r.path} must be GET (read-only diagnostic)"


# ---------------------------------------------------------------------------
# Column-existence sanity check (catches typos before prod)
# ---------------------------------------------------------------------------

def _model_columns(model):
    return {c.name for c in model.__table__.columns}


def test_rollout_sql_uses_real_rolling_stats_columns():
    """
    The rollout endpoint FILTERs on player_rolling_stats NSB columns. If any of
    these column names drift, the endpoint should fail in CI, not prod.
    """
    source = inspect.getsource(diag.diagnose_nsb_rollout)
    prs_cols = _model_columns(PlayerRollingStats)
    for col in ("w_ab", "w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases",
                "as_of_date", "window_days"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats model"
        assert col in source, f"rollout endpoint should reference {col}"


def test_rollout_sql_uses_real_player_scores_columns():
    source = inspect.getsource(diag.diagnose_nsb_rollout)
    ps_cols = _model_columns(PlayerScore)
    for col in ("player_type", "z_sb", "z_nsb", "as_of_date", "window_days"):
        assert col in ps_cols, f"{col} missing from PlayerScore model"
        assert col in source, f"rollout endpoint should reference {col}"


def test_leaders_sql_uses_real_columns():
    """
    The leaders endpoint joins player_scores + player_id_mapping + player_rolling_stats.
    All columns it references must exist on the models.
    """
    source = inspect.getsource(diag.diagnose_nsb_leaders)
    ps_cols = _model_columns(PlayerScore)
    pim_cols = _model_columns(PlayerIDMapping)
    prs_cols = _model_columns(PlayerRollingStats)

    for col in ("bdl_player_id", "games_in_window", "z_sb", "z_nsb",
                "composite_z", "score_0_100"):
        assert col in ps_cols, f"{col} missing from PlayerScore"
    assert "bdl_id" in pim_cols
    assert "full_name" in pim_cols
    for col in ("w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats"

    # And the SQL actually names the aliased columns
    assert "ps.z_nsb" in source
    assert "pim.bdl_id" in source
    assert "pim.full_name" in source
    assert "prs.w_net_stolen_bases" in source


def test_player_detail_sql_uses_real_columns():
    source = inspect.getsource(diag.diagnose_nsb_player)
    ps_cols = _model_columns(PlayerScore)
    prs_cols = _model_columns(PlayerRollingStats)
    for col in ("window_days", "games_in_window", "player_type", "z_sb", "z_nsb",
                "composite_z", "score_0_100"):
        assert col in ps_cols, f"{col} missing from PlayerScore"
    for col in ("w_stolen_bases", "w_caught_stealing", "w_net_stolen_bases",
                "w_ab", "w_hits"):
        assert col in prs_cols, f"{col} missing from PlayerRollingStats"
    assert "ps.z_nsb" in source
    assert "prs.w_net_stolen_bases" in source


# ---------------------------------------------------------------------------
# Whitelist enforcement
# ---------------------------------------------------------------------------

def test_allowed_windows_constant():
    """The only rolling windows we compute downstream are 7 / 14 / 30."""
    assert diag._ALLOWED_WINDOWS == frozenset({7, 14, 30})


def test_rollout_rejects_bad_window_days():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_rollout(window_days=99, as_of_date=None)
    assert exc.value.status_code == 400
    assert "window_days" in exc.value.detail


def test_leaders_rejects_bad_window_days():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_leaders(
            direction="top", limit=20, window_days=42, as_of_date=None,
        )
    assert exc.value.status_code == 400
    assert "window_days" in exc.value.detail


def test_leaders_rejects_bad_direction():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        diag.diagnose_nsb_leaders(
            direction="sideways", limit=20, window_days=14, as_of_date=None,
        )
    assert exc.value.status_code == 400
    assert "direction" in exc.value.detail


# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------

def test_helpers_handle_none():
    assert diag._f(None) is None
    assert diag._f("3.14") == 3.14
    assert diag._f(7) == 7.0
    assert diag._f("not a number") is None
    assert diag._i(None) is None
    assert diag._i("5") == 5
    assert diag._i("nope") is None
    assert diag._d(None) is None


def test_nsb_integrity_check_ok():
    assert diag._nsb_integrity_check(10.0, 3.0, 7.0) == "ok"


def test_nsb_integrity_check_mismatch():
    result = diag._nsb_integrity_check(10.0, 3.0, 99.0)
    assert result.startswith("mismatch")


def test_nsb_integrity_check_all_none():
    assert diag._nsb_integrity_check(None, None, None).startswith("n/a")


def test_nsb_integrity_check_partial_source():
    # sb or cs missing but nsb populated -> partial_source
    assert diag._nsb_integrity_check(None, 2.0, 5.0) == "partial_source"
    assert diag._nsb_integrity_check(5.0, None, 5.0) == "partial_source"


def test_nsb_integrity_check_nsb_null():
    assert diag._nsb_integrity_check(10.0, 3.0, None) == "nsb_null"


def test_interpret_rollout_messages():
    """Each verdict branch must produce a non-empty human-readable string."""
    for verdict in ("no_data", "healthy", "partial", "empty"):
        msg = diag._interpret_rollout(verdict, 50.0)
        assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# Layer 3 Freshness endpoint tests
# ---------------------------------------------------------------------------


def test_freshness_verdict_computation():
    """Verdict logic covers all expected states."""
    today = date(2026, 4, 15)
    yesterday = date(2026, 4, 14)
    three_days_ago = date(2026, 4, 12)

    # Both fresh
    assert diag._compute_freshness_verdict(True, True, yesterday, yesterday) == "healthy"
    # Both stale
    assert diag._compute_freshness_verdict(False, False, three_days_ago, three_days_ago) == "stale"
    # One fresh, one stale
    assert diag._compute_freshness_verdict(True, False, yesterday, three_days_ago) == "partial"
    # Both missing
    assert diag._compute_freshness_verdict(None, None, None, None) == "missing"
    # One missing
    assert diag._compute_freshness_verdict(True, None, yesterday, None) == "partial"
    assert diag._compute_freshness_verdict(None, True, None, yesterday) == "partial"


def test_freshness_interpretation_messages():
    """Each freshness verdict produces a non-empty human-readable string."""
    from zoneinfo import ZoneInfo

    now = datetime(2026, 4, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    today = date(2026, 4, 15)
    stale = date(2026, 4, 12)

    for verdict, prs_d, ps_d in [
        ("missing", None, None),
        ("healthy", today, today),
        ("partial", today, None),
        ("partial", None, today),
        ("partial", today, stale),
        ("stale", stale, stale),
    ]:
        msg = diag._interpret_freshness_verdict(verdict, prs_d, ps_d, now)
        assert isinstance(msg, str) and len(msg) > 0, f"Empty message for verdict={verdict}"


def test_format_audit_log():
    """Audit log formatting handles null values correctly."""
    class MockRow:
        id = 123
        job_type = "rolling_windows"
        target_date = date(2026, 4, 15)
        status = "SUCCESS"
        records_processed = 1000
        records_failed = 0
        processing_time_seconds = 5.5
        started_at = datetime(2026, 4, 15, 3, 0, 0)
        completed_at = datetime(2026, 4, 15, 3, 5, 0)

    result = diag._format_audit_log(MockRow())
    assert result["id"] == 123
    assert result["job_type"] == "rolling_windows"
    assert result["target_date"] == "2026-04-15"
    assert result["status"] == "SUCCESS"
    assert result["records_processed"] == 1000
    assert result["records_failed"] == 0
    assert result["processing_time_seconds"] == 5.5
    assert result["started_at"] == "2026-04-15T03:00:00"
    assert result["completed_at"] == "2026-04-15T03:05:00"


def test_layer3_freshness_endpoint_registered():
    """Layer 3 freshness endpoint is registered and uses GET."""
    from fastapi import HTTPException

    # Endpoint exists
    registered = {r.path for r in diag.router.routes}
    assert "/diagnose-scoring/layer3-freshness" in registered

    # Uses GET method
    for r in diag.router.routes:
        if r.path == "/diagnose-scoring/layer3-freshness":
            assert "GET" in r.methods
            # No query parameters expected
            assert not getattr(r, "dependant", None) or len(
                getattr(r.dependant, "query_params", [])
            ) == 0


# ---------------------------------------------------------------------------
# Helper function for Layer 3 freshness tests
# ---------------------------------------------------------------------------


def _fake_db_with_results(results):
    """
    Create a fake DB object that returns queued results from fetchone() calls.

    Args:
        results: List of objects with attributes matching what the endpoint expects.

    Returns a fake DB with:
        - execute() that returns a result object
        - fetchone() that pops from the results queue
        - fetchall() that returns empty list
        - close() that does nothing
    """
    from types import SimpleNamespace

    result_queue = list(results)

    class FakeResult:
        def fetchone(self):
            return result_queue.pop(0) if result_queue else None

        def fetchall(self):
            return []

    fake_db = SimpleNamespace(
        execute=lambda sql, params=None: FakeResult(),
        close=lambda: None,
    )
    return fake_db


# ---------------------------------------------------------------------------
# Behavior-level tests for Layer 3 freshness endpoint
# ---------------------------------------------------------------------------


def test_layer3_freshness_healthy_response(monkeypatch):
    """Healthy case: both tables have fresh data, audit logs present."""
    from types import SimpleNamespace

    today = date(2026, 4, 15)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)

    # Queue up results in the order the endpoint will query:
    # 1-2: Latest date queries (prs, ps)
    # 3-8: Count queries for windows 7, 14, 30 (prs, ps)
    # 9-10: Audit log queries (rw, ps)
    results = [
        # Latest dates
        SimpleNamespace(latest_date=today),
        SimpleNamespace(latest_date=today),
        # Counts for window 7
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        # Counts for window 14
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        # Counts for window 30
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        # Audit logs
        SimpleNamespace(
            id=1,
            job_type="rolling_windows",
            target_date=today,
            status="SUCCESS",
            records_processed=500,
            records_failed=0,
            processing_time_seconds=12.5,
            started_at=datetime(2026, 4, 15, 3, 0, 0),
            completed_at=datetime(2026, 4, 15, 3, 12, 30),
        ),
        SimpleNamespace(
            id=2,
            job_type="player_scores",
            target_date=today,
            status="SUCCESS",
            records_processed=500,
            records_failed=0,
            processing_time_seconds=12.5,
            started_at=datetime(2026, 4, 15, 4, 0, 0),
            completed_at=datetime(2026, 4, 15, 4, 12, 30),
        ),
    ]

    fake_db = _fake_db_with_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_layer3_freshness()

    assert result["verdict"] == "healthy"
    assert result["player_rolling_stats"]["latest_as_of_date"] == "2026-04-15"
    assert result["player_scores"]["latest_as_of_date"] == "2026-04-15"
    assert result["player_rolling_stats"]["row_counts_by_window"] == {7: 100, 14: 100, 30: 100}
    assert result["player_scores"]["row_counts_by_window"] == {7: 100, 14: 100, 30: 100}
    assert result["player_rolling_stats"]["latest_audit"]["id"] == 1
    assert result["player_scores"]["latest_audit"]["id"] == 2
    assert result["player_rolling_stats"]["latest_audit"]["status"] == "SUCCESS"
    assert "checked_at" in result
    assert "message" in result
    assert "schedule_expectations" in result


def test_layer3_freshness_missing_data(monkeypatch):
    """Missing case: both tables are empty."""
    from types import SimpleNamespace

    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)

    # Both tables empty: latest dates are None
    results = [
        SimpleNamespace(latest_date=None),
        SimpleNamespace(latest_date=None),
        # No count queries needed when dates are None
        # No audit logs
    ]

    fake_db = _fake_db_with_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_layer3_freshness()

    assert result["verdict"] == "missing"
    assert result["player_rolling_stats"]["latest_as_of_date"] is None
    assert result["player_scores"]["latest_as_of_date"] is None
    assert "layer 3 data" in result["message"].lower()


def test_layer3_freshness_stale_data(monkeypatch):
    """Stale case: both tables have data but it's 2+ days old."""
    from types import SimpleNamespace

    stale_date = date(2026, 4, 12)  # 3 days ago from April 15
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)

    results = [
        # Latest dates (stale)
        SimpleNamespace(latest_date=stale_date),
        SimpleNamespace(latest_date=stale_date),
        # Counts for windows 7, 14, 30
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        SimpleNamespace(n=100),
        # Audit logs
        SimpleNamespace(
            id=10,
            job_type="rolling_windows",
            target_date=stale_date,
            status="SUCCESS",
            records_processed=500,
            records_failed=0,
            processing_time_seconds=12.5,
            started_at=datetime(2026, 4, 12, 3, 0, 0),
            completed_at=datetime(2026, 4, 12, 3, 12, 30),
        ),
        SimpleNamespace(
            id=11,
            job_type="player_scores",
            target_date=stale_date,
            status="SUCCESS",
            records_processed=500,
            records_failed=0,
            processing_time_seconds=12.5,
            started_at=datetime(2026, 4, 12, 4, 0, 0),
            completed_at=datetime(2026, 4, 12, 4, 12, 30),
        ),
    ]

    fake_db = _fake_db_with_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_layer3_freshness()

    assert result["verdict"] == "stale"
    assert result["player_rolling_stats"]["latest_as_of_date"] == "2026-04-12"
    assert result["player_scores"]["latest_as_of_date"] == "2026-04-12"
    assert "stale" in result["message"].lower()


def test_layer3_freshness_partial_data(monkeypatch):
    """Partial case: player_rolling_stats has data, player_scores is empty."""
    from types import SimpleNamespace

    today = date(2026, 4, 15)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)

    results = [
        # Latest dates: prs has data, ps is None
        SimpleNamespace(latest_date=today),
        SimpleNamespace(latest_date=None),
        # Counts: only for prs (ps has no data)
        SimpleNamespace(n=100),  # prs window 7
        SimpleNamespace(n=100),  # prs window 14
        SimpleNamespace(n=100),  # prs window 30
        # Audit log: only for rolling_windows
        SimpleNamespace(
            id=20,
            job_type="rolling_windows",
            target_date=today,
            status="SUCCESS",
            records_processed=300,
            records_failed=0,
            processing_time_seconds=8.5,
            started_at=datetime(2026, 4, 15, 3, 0, 0),
            completed_at=datetime(2026, 4, 15, 3, 8, 30),
        ),
    ]

    fake_db = _fake_db_with_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_layer3_freshness()

    assert result["verdict"] == "partial"
    assert result["player_rolling_stats"]["latest_as_of_date"] == "2026-04-15"
    assert result["player_scores"]["latest_as_of_date"] is None
    assert "partial" in result["message"].lower()


# ---------------------------------------------------------------------------
# Decision Pipeline Freshness endpoint tests (P17-P19)
# ---------------------------------------------------------------------------


def test_decision_freshness_verdict_computation():
    """Verdict logic covers all expected states for decision pipeline."""
    today = date(2026, 4, 15)
    yesterday = date(2026, 4, 14)
    three_days_ago = date(2026, 4, 12)

    # Both fresh
    assert diag._compute_decision_freshness_verdict(
        True, True, yesterday, yesterday
    ) == "healthy"
    # Both stale
    assert diag._compute_decision_freshness_verdict(
        False, False, three_days_ago, three_days_ago
    ) == "stale"
    # One fresh, one stale
    assert diag._compute_decision_freshness_verdict(
        True, False, yesterday, three_days_ago
    ) == "partial"
    # Both missing
    assert diag._compute_decision_freshness_verdict(
        None, None, None, None
    ) == "missing"
    # One missing
    assert diag._compute_decision_freshness_verdict(
        True, None, yesterday, None
    ) == "partial"
    assert diag._compute_decision_freshness_verdict(
        None, True, None, yesterday
    ) == "partial"


def test_decision_freshness_interpretation_messages():
    """Each freshness verdict produces a non-empty human-readable string."""
    from zoneinfo import ZoneInfo

    now = datetime(2026, 4, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    today = date(2026, 4, 15)
    stale = date(2026, 4, 12)

    for verdict, dr_d, de_d in [
        ("missing", None, None),
        ("healthy", today, today),
        ("partial", today, None),
        ("partial", None, today),
        ("partial", today, stale),
        ("stale", stale, stale),
    ]:
        msg = diag._interpret_decision_freshness_verdict(verdict, dr_d, de_d, now)
        assert isinstance(msg, str) and len(msg) > 0, f"Empty message for verdict={verdict}"


def test_decision_freshness_endpoint_registered():
    """Decision pipeline freshness endpoint is registered and uses GET."""
    registered = {r.path for r in diag.router.routes}
    assert "/diagnose-decision/pipeline-freshness" in registered

    # Uses GET method
    for r in diag.router.routes:
        if r.path == "/diagnose-decision/pipeline-freshness":
            assert "GET" in r.methods


def _fake_db_with_decision_results(results):
    """
    Create a fake DB object that returns queued results for decision pipeline queries.

    The decision freshness endpoint queries in this order:
    1-2: Latest as_of_date (decision_results, decision_explanations)
    3-6: decision_results counts by type (lineup, waiver)
    7-10: decision_explanations counts by type (lineup, waiver)
    11-12: Latest computed_at (decision_results, decision_explanations)
    """
    from types import SimpleNamespace

    result_queue = list(results)

    class FakeResult:
        def fetchone(self):
            return result_queue.pop(0) if result_queue else None

        def fetchall(self):
            return []

    fake_db = SimpleNamespace(
        execute=lambda sql, params=None: FakeResult(),
        close=lambda: None,
    )
    return fake_db


def test_decision_freshness_healthy_response(monkeypatch):
    """Healthy case: both tables have fresh data."""
    from types import SimpleNamespace

    today = date(2026, 4, 15)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)
    computed = datetime(2026, 4, 15, 7, 30, 0)

    # The endpoint queries in this order:
    # 1. dr_latest, 2. de_latest
    # For each dtype in ("lineup", "waiver"):
    #   - dr query, then de query
    # So order is: dr_lineup, de_lineup, dr_waiver, de_waiver
    # Then: dr_computed, de_computed
    results = [
        # Latest as_of_date
        SimpleNamespace(latest_date=today),
        SimpleNamespace(latest_date=today),
        # lineup: dr then de
        SimpleNamespace(n=10),
        SimpleNamespace(n=10),
        # waiver: dr then de
        SimpleNamespace(n=5),
        SimpleNamespace(n=5),
        # Latest computed_at
        SimpleNamespace(latest_computed=computed),
        SimpleNamespace(latest_computed=computed),
    ]

    fake_db = _fake_db_with_decision_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_decision_pipeline_freshness()

    assert result["verdict"] == "healthy"
    assert result["decision_results"]["latest_as_of_date"] == "2026-04-15"
    assert result["decision_explanations"]["latest_as_of_date"] == "2026-04-15"
    # Total should be sum of breakdown values
    dr_breakdown = result["decision_results"]["breakdown_by_type"]
    de_breakdown = result["decision_explanations"]["breakdown_by_type"]
    assert result["decision_results"]["total_row_count"] == sum(dr_breakdown.values())
    assert result["decision_explanations"]["total_row_count"] == sum(de_breakdown.values())
    assert result["decision_results"]["breakdown_by_type"] == {"lineup": 10, "waiver": 5}
    assert result["decision_explanations"]["breakdown_by_type"] == {"lineup": 10, "waiver": 5}
    assert "checked_at" in result
    assert "schedule_expectations" in result


def test_decision_freshness_missing_data(monkeypatch):
    """Missing case: both tables are empty."""
    from types import SimpleNamespace

    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)

    results = [
        SimpleNamespace(latest_date=None),
        SimpleNamespace(latest_date=None),
    ]

    fake_db = _fake_db_with_decision_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_decision_pipeline_freshness()

    assert result["verdict"] == "missing"
    assert result["decision_results"]["latest_as_of_date"] is None
    assert result["decision_explanations"]["latest_as_of_date"] is None
    assert "decision pipeline" in result["message"].lower()


def test_decision_freshness_stale_data(monkeypatch):
    """Stale case: both tables have data but it's 2+ days old."""
    from types import SimpleNamespace

    stale_date = date(2026, 4, 12)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)
    computed = datetime(2026, 4, 12, 7, 30, 0)

    results = [
        SimpleNamespace(latest_date=stale_date),
        SimpleNamespace(latest_date=stale_date),
        # lineup: dr then de
        SimpleNamespace(n=10),
        SimpleNamespace(n=10),
        # waiver: dr then de
        SimpleNamespace(n=5),
        SimpleNamespace(n=5),
        SimpleNamespace(latest_computed=computed),
        SimpleNamespace(latest_computed=computed),
    ]

    fake_db = _fake_db_with_decision_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_decision_pipeline_freshness()

    assert result["verdict"] == "stale"
    assert result["decision_results"]["latest_as_of_date"] == "2026-04-12"
    assert result["decision_explanations"]["latest_as_of_date"] == "2026-04-12"
    assert "stale" in result["message"].lower()


def test_decision_freshness_partial_data(monkeypatch):
    """Partial case: decision_results has data, decision_explanations is empty."""
    from types import SimpleNamespace

    today = date(2026, 4, 15)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)
    computed = datetime(2026, 4, 15, 7, 30, 0)

    results = [
        SimpleNamespace(latest_date=today),
        SimpleNamespace(latest_date=None),
        SimpleNamespace(n=10),
        SimpleNamespace(n=5),
        # No explanation counts since latest_as_of_date is None
        SimpleNamespace(latest_computed=computed),
        SimpleNamespace(latest_computed=None),
    ]

    fake_db = _fake_db_with_decision_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_decision_pipeline_freshness()

    assert result["verdict"] == "partial"
    assert result["decision_results"]["latest_as_of_date"] == "2026-04-15"
    assert result["decision_explanations"]["latest_as_of_date"] is None
    assert "partial" in result["message"].lower()


def test_decision_freshness_with_only_lineup_type(monkeypatch):
    """Edge case: only lineup decisions exist (no waiver recommendations)."""
    from types import SimpleNamespace

    today = date(2026, 4, 15)
    now_et = datetime(2026, 4, 15, 10, 0, 0, tzinfo=diag._ET)
    computed = datetime(2026, 4, 15, 7, 30, 0)

    # Order: dr_latest, de_latest, dr_lineup, de_lineup, dr_waiver, de_waiver, dr_computed, de_computed
    results = [
        SimpleNamespace(latest_date=today),
        SimpleNamespace(latest_date=today),
        # lineup: dr then de (both 20)
        SimpleNamespace(n=20),
        SimpleNamespace(n=20),
        # waiver: dr then de (both 0)
        SimpleNamespace(n=0),
        SimpleNamespace(n=0),
        SimpleNamespace(latest_computed=computed),
        SimpleNamespace(latest_computed=computed),
    ]

    fake_db = _fake_db_with_decision_results(results)
    monkeypatch.setattr(diag, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(diag, "_now_et", lambda: now_et)

    result = diag.diagnose_decision_pipeline_freshness()

    assert result["verdict"] == "healthy"
    # Total should be sum of breakdown values
    dr_breakdown = result["decision_results"]["breakdown_by_type"]
    de_breakdown = result["decision_explanations"]["breakdown_by_type"]
    assert result["decision_results"]["total_row_count"] == sum(dr_breakdown.values())
    assert result["decision_explanations"]["total_row_count"] == sum(de_breakdown.values())
    assert result["decision_results"]["breakdown_by_type"] == {"lineup": 20, "waiver": 0}
    assert result["decision_explanations"]["breakdown_by_type"] == {"lineup": 20, "waiver": 0}


# ---------------------------------------------------------------------------
# Router mount check (main.py must include this router)
# ---------------------------------------------------------------------------

def test_router_is_mounted_in_main_app():
    """
    Catches the easy mistake of adding an APIRouter but forgetting to mount
    it on the FastAPI app.
    """
    from backend.main import app
    mounted_paths = {r.path for r in app.routes}
    # When mounted with prefix="/admin", diagnostic paths appear as /admin/diagnose-scoring/...
    for p in EXPECTED_PATHS:
        full = f"/admin{p}"
        assert full in mounted_paths, (
            f"Expected scoring-diagnostics endpoint {full} to be mounted on app"
        )
