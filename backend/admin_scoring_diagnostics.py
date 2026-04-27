"""
Admin diagnostic endpoints for the scoring pipeline -- P27 NSB rollout verification.

Built after the v27 migration deploy to verify that the new NSB columns
(w_caught_stealing, w_net_stolen_bases, z_nsb) actually populate on subsequent
scheduled rolling_windows + player_scores runs.

Endpoints (all read-only, all GET, all whitelisted inputs):
  /admin/diagnose-scoring/nsb-rollout              -- aggregate fill-rate + distribution
  /admin/diagnose-scoring/nsb-leaders?direction=top&limit=20
  /admin/diagnose-scoring/nsb-player?bdl_player_id=12345
  /admin/diagnose-scoring/layer3-freshness         -- Layer 3 freshness + coverage observability

REMOVE AFTER NSB ROLLOUT VALIDATED (tracked in HANDOFF.md).
"""

from datetime import date, datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from backend.models import SessionLocal

_ET = ZoneInfo("America/New_York")


def _now_et():
    """Get current time in ET. Overridable for testing."""
    return datetime.now(_ET)


router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v):
    """Safe float coercion for JSON; None stays None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    """Safe int coercion for JSON; None stays None."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _d(v):
    """Date -> ISO string or None."""
    return v.isoformat() if v is not None else None


# Windows we expose to callers. Anything else is rejected as a bad request.
_ALLOWED_WINDOWS: frozenset = frozenset({7, 14, 30})

# Histogram bin edges for z_nsb (and any Z-score). Lower-inclusive, upper-exclusive.
# Mirrors the _label_z thresholds in explainability_layer.
_Z_BIN_EDGES: list = [-3.0, -1.5, -0.5, 0.5, 1.5, 3.01]
_Z_BIN_LABELS: list = ["POOR(<-1.5)", "WEAK(-1.5..-0.5)", "AVERAGE(-0.5..0.5)",
                       "STRONG(0.5..1.5)", "ELITE(>=1.5)"]


# ---------------------------------------------------------------------------
# 1. Rollout summary -- is the v27 migration actually populating NSB fields?
# ---------------------------------------------------------------------------

@router.get("/diagnose-scoring/nsb-rollout")
def diagnose_nsb_rollout(
    window_days: int = Query(14, description="Rolling window size (7, 14, or 30)"),
    as_of_date: Optional[str] = Query(
        None,
        description="ISO date (YYYY-MM-DD). If omitted, uses the latest as_of_date in player_scores.",
    ),
):
    """
    Aggregate view of NSB pipeline rollout health.

    Answers:
      - Did the most recent rolling_windows job populate w_caught_stealing / w_net_stolen_bases?
      - Did the most recent player_scores job populate z_nsb?
      - What does the z_nsb distribution look like (histogram + summary stats)?
      - How often does z_nsb diverge materially from z_sb (the legacy field)?

    Use this FIRST after the 4 AM ET daily job run to verify NSB is flowing.
    """
    if window_days not in _ALLOWED_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"window_days must be one of: {sorted(_ALLOWED_WINDOWS)}",
        )

    db = SessionLocal()
    try:
        # Resolve as_of_date: caller-supplied or the latest populated in player_scores.
        if as_of_date:
            resolved_date_row = db.execute(
                text("SELECT :d::date AS d"),
                {"d": as_of_date},
            ).fetchone()
        else:
            resolved_date_row = db.execute(
                text(
                    "SELECT MAX(as_of_date) AS d FROM player_scores "
                    "WHERE window_days = :w"
                ),
                {"w": window_days},
            ).fetchone()

        resolved_date = resolved_date_row.d if resolved_date_row else None
        if resolved_date is None:
            return {
                "as_of_date": None,
                "window_days": window_days,
                "message": "No player_scores rows found for this window. Has the job run?",
                "player_rolling_stats": None,
                "player_scores": None,
                "z_nsb_distribution": None,
                "z_nsb_vs_z_sb_divergence": None,
            }

        # --- player_rolling_stats fill rates for NSB columns ----------------
        prs_fill = db.execute(text("""
            SELECT
                COUNT(*)                                                      AS total_rows,
                COUNT(*) FILTER (WHERE w_ab IS NOT NULL)                      AS hitter_rows,
                COUNT(*) FILTER (WHERE w_stolen_bases     IS NOT NULL)        AS sb_filled,
                COUNT(*) FILTER (WHERE w_caught_stealing  IS NOT NULL)        AS cs_filled,
                COUNT(*) FILTER (WHERE w_net_stolen_bases IS NOT NULL)        AS nsb_filled,
                SUM(w_stolen_bases)                                           AS sum_sb,
                SUM(w_caught_stealing)                                        AS sum_cs,
                SUM(w_net_stolen_bases)                                       AS sum_nsb
            FROM player_rolling_stats
            WHERE as_of_date = :d AND window_days = :w
        """), {"d": resolved_date, "w": window_days}).fetchone()

        # --- player_scores fill rates for z_nsb -----------------------------
        ps_fill = db.execute(text("""
            SELECT
                COUNT(*)                                     AS total_rows,
                COUNT(*) FILTER (WHERE player_type IN ('hitter', 'two_way')) AS hitter_rows,
                COUNT(*) FILTER (WHERE z_sb  IS NOT NULL)    AS z_sb_filled,
                COUNT(*) FILTER (WHERE z_nsb IS NOT NULL)    AS z_nsb_filled
            FROM player_scores
            WHERE as_of_date = :d AND window_days = :w
        """), {"d": resolved_date, "w": window_days}).fetchone()

        # --- z_nsb distribution stats --------------------------------------
        dist = db.execute(text("""
            SELECT
                COUNT(z_nsb)                 AS n,
                MIN(z_nsb)                   AS z_min,
                MAX(z_nsb)                   AS z_max,
                AVG(z_nsb)                   AS z_mean,
                STDDEV_POP(z_nsb)            AS z_std
            FROM player_scores
            WHERE as_of_date = :d AND window_days = :w AND z_nsb IS NOT NULL
        """), {"d": resolved_date, "w": window_days}).fetchone()

        # --- histogram of z_nsb --------------------------------------------
        # Count rows per half-open bucket using FILTER + conditional predicates.
        buckets_row = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE z_nsb <  -1.5)                           AS b_poor,
                COUNT(*) FILTER (WHERE z_nsb >= -1.5 AND z_nsb <  -0.5)         AS b_weak,
                COUNT(*) FILTER (WHERE z_nsb >= -0.5 AND z_nsb <=  0.5)         AS b_avg,
                COUNT(*) FILTER (WHERE z_nsb >   0.5 AND z_nsb <   1.5)         AS b_strong,
                COUNT(*) FILTER (WHERE z_nsb >=  1.5)                           AS b_elite
            FROM player_scores
            WHERE as_of_date = :d AND window_days = :w AND z_nsb IS NOT NULL
        """), {"d": resolved_date, "w": window_days}).fetchone()

        histogram = {
            "POOR(<-1.5)":          _i(buckets_row.b_poor),
            "WEAK(-1.5..-0.5)":     _i(buckets_row.b_weak),
            "AVERAGE(-0.5..0.5)":   _i(buckets_row.b_avg),
            "STRONG(0.5..1.5)":     _i(buckets_row.b_strong),
            "ELITE(>=1.5)":         _i(buckets_row.b_elite),
        }

        # --- divergence: how often do z_nsb and z_sb differ materially? ---
        # Threshold of 0.1 is ~3% of the Z_CAP range; covers "meaningful" shifts.
        div = db.execute(text("""
            SELECT
                COUNT(*)                                                                 AS both_filled,
                COUNT(*) FILTER (WHERE ABS(z_nsb - z_sb) >  0.1)                         AS differ_gt_0_1,
                COUNT(*) FILTER (WHERE ABS(z_nsb - z_sb) >  0.5)                         AS differ_gt_0_5,
                AVG(z_nsb - z_sb)                                                        AS mean_delta,
                MAX(ABS(z_nsb - z_sb))                                                   AS max_abs_delta
            FROM player_scores
            WHERE as_of_date = :d AND window_days = :w
              AND z_nsb IS NOT NULL AND z_sb IS NOT NULL
        """), {"d": resolved_date, "w": window_days}).fetchone()

        # Interpretation helper -- three-state verdict on rollout health.
        def _pct(num, denom):
            if not denom or denom == 0:
                return None
            return round(100.0 * (num or 0) / denom, 2)

        prs_hitter = prs_fill.hitter_rows or 0
        ps_hitter = ps_fill.hitter_rows or 0
        cs_fill_pct = _pct(prs_fill.cs_filled, prs_hitter)
        nsb_fill_pct = _pct(prs_fill.nsb_filled, prs_hitter)
        z_nsb_fill_pct = _pct(ps_fill.z_nsb_filled, ps_hitter)

        if z_nsb_fill_pct is None:
            verdict = "no_data"
        elif z_nsb_fill_pct >= 80.0:
            verdict = "healthy"
        elif z_nsb_fill_pct >= 10.0:
            verdict = "partial"
        else:
            verdict = "empty"

        return {
            "as_of_date": _d(resolved_date),
            "window_days": window_days,
            "verdict": verdict,
            "player_rolling_stats": {
                "total_rows": _i(prs_fill.total_rows),
                "hitter_rows": _i(prs_hitter),
                "w_stolen_bases_filled": _i(prs_fill.sb_filled),
                "w_caught_stealing_filled": _i(prs_fill.cs_filled),
                "w_net_stolen_bases_filled": _i(prs_fill.nsb_filled),
                "cs_fill_pct_of_hitters": cs_fill_pct,
                "nsb_fill_pct_of_hitters": nsb_fill_pct,
                "sum_w_stolen_bases": _f(prs_fill.sum_sb),
                "sum_w_caught_stealing": _f(prs_fill.sum_cs),
                "sum_w_net_stolen_bases": _f(prs_fill.sum_nsb),
            },
            "player_scores": {
                "total_rows": _i(ps_fill.total_rows),
                "hitter_rows": _i(ps_hitter),
                "z_sb_filled": _i(ps_fill.z_sb_filled),
                "z_nsb_filled": _i(ps_fill.z_nsb_filled),
                "z_nsb_fill_pct_of_hitters": z_nsb_fill_pct,
            },
            "z_nsb_distribution": {
                "n": _i(dist.n),
                "min": _f(dist.z_min),
                "max": _f(dist.z_max),
                "mean": _f(dist.z_mean),
                "std": _f(dist.z_std),
                "histogram": histogram,
            },
            "z_nsb_vs_z_sb_divergence": {
                "both_filled": _i(div.both_filled),
                "differ_gt_0_1": _i(div.differ_gt_0_1),
                "differ_gt_0_5": _i(div.differ_gt_0_5),
                "mean_delta": _f(div.mean_delta),
                "max_abs_delta": _f(div.max_abs_delta),
                "note": (
                    "Large divergence is normal only for players with many caught-stealings. "
                    "Typical MLB CS rate is ~25% of attempts, so for a player with 10 SB / 3 CS "
                    "(NSB=7 vs SB=10), expect |z_nsb - z_sb| < 0.5 league-wide."
                ),
            },
            "interpretation": _interpret_rollout(verdict, z_nsb_fill_pct),
        }
    finally:
        db.close()


def _interpret_rollout(verdict: str, fill_pct) -> str:
    """Human-readable summary for the rollout verdict."""
    if verdict == "no_data":
        return ("No player_scores rows for this window. Either the job has not "
                "run yet or the window_days filter is wrong.")
    if verdict == "healthy":
        return (f"NSB rollout HEALTHY: z_nsb populated for {fill_pct}% of hitter rows. "
                "Pipeline is writing NSB end-to-end.")
    if verdict == "partial":
        return (f"NSB rollout PARTIAL: z_nsb populated for only {fill_pct}% of hitters. "
                "Possible causes: insufficient pool size (< MIN_SAMPLE=5), degenerate "
                "variance in CS across league, or mid-deploy snapshot.")
    return ("NSB rollout EMPTY: z_nsb is null for virtually all rows. "
            "Check that v27 migration applied and next rolling_windows + "
            "player_scores job ran AFTER the deploy.")


# ---------------------------------------------------------------------------
# 2. Leaders -- who are the top/bottom N by z_nsb?
# ---------------------------------------------------------------------------

@router.get("/diagnose-scoring/nsb-leaders")
def diagnose_nsb_leaders(
    direction: str = Query("top", description="'top' or 'bottom'"),
    limit: int = Query(20, ge=1, le=100),
    window_days: int = Query(14),
    as_of_date: Optional[str] = Query(None),
):
    """
    Top-N or bottom-N players by z_nsb for a given as_of_date + window.

    Useful for spot-checking: are the top-NSB players the ones you'd expect
    (active basestealers -- Ohtani, De La Cruz, Chisholm, etc.)? Are any
    showing surprisingly poor z_nsb because of heavy CS?
    """
    if direction not in {"top", "bottom"}:
        raise HTTPException(status_code=400, detail="direction must be 'top' or 'bottom'")
    if window_days not in _ALLOWED_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"window_days must be one of: {sorted(_ALLOWED_WINDOWS)}",
        )

    # Direction controls sort order. Whitelisted strings -- safe to inline.
    order = "DESC" if direction == "top" else "ASC"

    db = SessionLocal()
    try:
        if as_of_date:
            d_row = db.execute(text("SELECT :d::date AS d"), {"d": as_of_date}).fetchone()
        else:
            d_row = db.execute(
                text("SELECT MAX(as_of_date) AS d FROM player_scores WHERE window_days = :w"),
                {"w": window_days},
            ).fetchone()

        resolved_date = d_row.d if d_row else None
        if resolved_date is None:
            return {
                "as_of_date": None,
                "window_days": window_days,
                "direction": direction,
                "rows": [],
                "message": "No player_scores rows found.",
            }

        rows = db.execute(text(f"""
            SELECT
                ps.bdl_player_id,
                pim.full_name                                AS player_name,
                ps.games_in_window,
                ps.z_sb,
                ps.z_nsb,
                ps.composite_z,
                ps.score_0_100,
                prs.w_stolen_bases,
                prs.w_caught_stealing,
                prs.w_net_stolen_bases
            FROM player_scores ps
            LEFT JOIN player_id_mapping pim ON pim.bdl_id = ps.bdl_player_id
            LEFT JOIN player_rolling_stats prs
                ON prs.bdl_player_id = ps.bdl_player_id
               AND prs.as_of_date    = ps.as_of_date
               AND prs.window_days   = ps.window_days
            WHERE ps.as_of_date = :d
              AND ps.window_days = :w
              AND ps.z_nsb IS NOT NULL
            ORDER BY ps.z_nsb {order} NULLS LAST
            LIMIT :limit
        """), {"d": resolved_date, "w": window_days, "limit": limit}).fetchall()

        return {
            "as_of_date": _d(resolved_date),
            "window_days": window_days,
            "direction": direction,
            "limit": limit,
            "rows": [
                {
                    "bdl_player_id": _i(r.bdl_player_id),
                    "player_name": r.player_name,
                    "games_in_window": _i(r.games_in_window),
                    "z_sb": _f(r.z_sb),
                    "z_nsb": _f(r.z_nsb),
                    "z_nsb_minus_z_sb": (
                        _f(r.z_nsb) - _f(r.z_sb)
                        if r.z_nsb is not None and r.z_sb is not None
                        else None
                    ),
                    "composite_z": _f(r.composite_z),
                    "score_0_100": _f(r.score_0_100),
                    "w_stolen_bases": _f(r.w_stolen_bases),
                    "w_caught_stealing": _f(r.w_caught_stealing),
                    "w_net_stolen_bases": _f(r.w_net_stolen_bases),
                }
                for r in rows
            ],
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Per-player detail -- NSB across all windows for one player
# ---------------------------------------------------------------------------

@router.get("/diagnose-scoring/nsb-player")
def diagnose_nsb_player(
    bdl_player_id: int = Query(..., description="BDL player id"),
    as_of_date: Optional[str] = Query(None),
):
    """
    All-windows NSB breakdown for one player, to sanity-check that SB / CS /
    NSB / z_nsb all tie out across the 7/14/30 windows.
    """
    db = SessionLocal()
    try:
        if as_of_date:
            d_row = db.execute(text("SELECT :d::date AS d"), {"d": as_of_date}).fetchone()
        else:
            d_row = db.execute(
                text("SELECT MAX(as_of_date) AS d FROM player_scores WHERE bdl_player_id = :pid"),
                {"pid": bdl_player_id},
            ).fetchone()

        resolved_date = d_row.d if d_row else None
        if resolved_date is None:
            raise HTTPException(
                status_code=404,
                detail=f"No player_scores rows for bdl_player_id={bdl_player_id}",
            )

        name_row = db.execute(
            text("SELECT full_name FROM player_id_mapping WHERE bdl_id = :pid"),
            {"pid": bdl_player_id},
        ).fetchone()

        windows = db.execute(text("""
            SELECT
                ps.window_days,
                ps.games_in_window,
                ps.player_type,
                ps.z_sb,
                ps.z_nsb,
                ps.composite_z,
                ps.score_0_100,
                prs.w_stolen_bases,
                prs.w_caught_stealing,
                prs.w_net_stolen_bases,
                prs.w_ab,
                prs.w_hits
            FROM player_scores ps
            LEFT JOIN player_rolling_stats prs
                ON prs.bdl_player_id = ps.bdl_player_id
               AND prs.as_of_date    = ps.as_of_date
               AND prs.window_days   = ps.window_days
            WHERE ps.bdl_player_id = :pid AND ps.as_of_date = :d
            ORDER BY ps.window_days ASC
        """), {"pid": bdl_player_id, "d": resolved_date}).fetchall()

        return {
            "bdl_player_id": bdl_player_id,
            "player_name": name_row.full_name if name_row else None,
            "as_of_date": _d(resolved_date),
            "windows": [
                {
                    "window_days": _i(w.window_days),
                    "player_type": w.player_type,
                    "games_in_window": _i(w.games_in_window),
                    "w_ab": _f(w.w_ab),
                    "w_hits": _f(w.w_hits),
                    "w_stolen_bases": _f(w.w_stolen_bases),
                    "w_caught_stealing": _f(w.w_caught_stealing),
                    "w_net_stolen_bases": _f(w.w_net_stolen_bases),
                    "z_sb": _f(w.z_sb),
                    "z_nsb": _f(w.z_nsb),
                    "composite_z": _f(w.composite_z),
                    "score_0_100": _f(w.score_0_100),
                    # Integrity check: w_net_stolen_bases must equal w_stolen_bases - w_caught_stealing
                    "nsb_check": _nsb_integrity_check(
                        w.w_stolen_bases, w.w_caught_stealing, w.w_net_stolen_bases,
                    ),
                }
                for w in windows
            ],
        }
    finally:
        db.close()


def _nsb_integrity_check(sb, cs, nsb) -> str:
    """
    Verify the NSB invariant: w_net_stolen_bases = w_stolen_bases - w_caught_stealing.
    Returns a short status string suitable for UI display.
    """
    if sb is None and cs is None and nsb is None:
        return "n/a (pure pitcher or no batting)"
    if nsb is None:
        return "nsb_null"
    if sb is None or cs is None:
        return "partial_source"
    expected = float(sb) - float(cs)
    if abs(float(nsb) - expected) < 1e-6:
        return "ok"
    return f"mismatch (expected {expected:.4f}, got {float(nsb):.4f})"


# ---------------------------------------------------------------------------
# 4. Layer 3 Freshness -- operator-grade observability for scoring spine
# ---------------------------------------------------------------------------

@router.get("/diagnose-scoring/layer3-freshness")
def diagnose_layer3_freshness():
    """
    Layer 3 scoring pipeline freshness and coverage observability.

    Answers:
      - What is the latest as_of_date for player_rolling_stats?
      - What is the latest as_of_date for player_scores?
      - How many rows exist per window on the latest dates?
      - What were the last audit log entries for rolling_windows/player_scores?
      - Is the pipeline healthy, stale, partial, or missing?

    Use this for operational monitoring of the Layer 3 scoring spine.
    """
    db = SessionLocal()
    try:
        now_et = _now_et()

        # --- Latest as_of_date for each table -------------------------------
        prs_latest_row = db.execute(text("""
            SELECT MAX(as_of_date) AS latest_date FROM player_rolling_stats
        """)).fetchone()

        ps_latest_row = db.execute(text("""
            SELECT MAX(as_of_date) AS latest_date FROM player_scores
        """)).fetchone()

        prs_latest = prs_latest_row.latest_date if prs_latest_row else None
        ps_latest = ps_latest_row.latest_date if ps_latest_row else None

        # --- Row counts by window for latest dates -------------------------
        prs_counts = {}
        ps_counts = {}

        for window in (7, 14, 30):
            if prs_latest:
                cnt = db.execute(text("""
                    SELECT COUNT(*) AS n FROM player_rolling_stats
                    WHERE as_of_date = :d AND window_days = :w
                """), {"d": prs_latest, "w": window}).fetchone()
                prs_counts[window] = _i(cnt.n) if cnt else 0
            else:
                prs_counts[window] = None

            if ps_latest:
                cnt = db.execute(text("""
                    SELECT COUNT(*) AS n FROM player_scores
                    WHERE as_of_date = :d AND window_days = :w
                """), {"d": ps_latest, "w": window}).fetchone()
                ps_counts[window] = _i(cnt.n) if cnt else 0
            else:
                ps_counts[window] = None

        # --- Latest audit log entries --------------------------------------
        # rolling_windows typically runs at 3 AM ET
        rw_audit = db.execute(text("""
            SELECT id, job_type, target_date, status, records_processed,
                   records_failed, processing_time_seconds, started_at, completed_at
            FROM data_ingestion_logs
            WHERE job_type = 'rolling_windows'
            ORDER BY id DESC
            LIMIT 1
        """)).fetchone()

        # player_scores typically runs at 4 AM ET
        ps_audit = db.execute(text("""
            SELECT id, job_type, target_date, status, records_processed,
                   records_failed, processing_time_seconds, started_at, completed_at
            FROM data_ingestion_logs
            WHERE job_type = 'player_scores'
            ORDER BY id DESC
            LIMIT 1
        """)).fetchone()

        # --- Freshness verdict --------------------------------------------
        # Expected schedule: rolling_windows ~3 AM ET, player_scores ~4 AM ET
        # Consider data stale if latest as_of_date is 2+ days old
        today_et = now_et.date()
        stale_threshold = today_et - timedelta(days=2)

        prs_is_fresh = prs_latest and prs_latest >= stale_threshold
        ps_is_fresh = ps_latest and ps_latest >= stale_threshold

        verdict = _compute_freshness_verdict(prs_is_fresh, ps_is_fresh, prs_latest, ps_latest)

        return {
            "checked_at": now_et.isoformat(),
            "verdict": verdict,
            "message": _interpret_freshness_verdict(verdict, prs_latest, ps_latest, now_et),
            "player_rolling_stats": {
                "latest_as_of_date": _d(prs_latest),
                "row_counts_by_window": prs_counts,
                "latest_audit": _format_audit_log(rw_audit) if rw_audit else None,
            },
            "player_scores": {
                "latest_as_of_date": _d(ps_latest),
                "row_counts_by_window": ps_counts,
                "latest_audit": _format_audit_log(ps_audit) if ps_audit else None,
            },
            "schedule_expectations": {
                "rolling_windows_expected": "03:00 AM ET (daily)",
                "player_scores_expected": "04:00 AM ET (daily)",
                "timezone": "America/New_York",
            },
        }
    finally:
        db.close()


def _compute_freshness_verdict(
    prs_fresh: Optional[bool],
    ps_fresh: Optional[bool],
    prs_latest: Optional[date],
    ps_latest: Optional[date],
) -> str:
    """
    Compute a simple freshness verdict for Layer 3 tables.

    Returns one of: healthy, stale, partial, missing
    """
    if prs_latest is None and ps_latest is None:
        return "missing"
    if prs_latest and ps_latest is None:
        return "partial"
    if prs_latest is None and ps_latest:
        return "partial"
    if prs_fresh and ps_fresh:
        return "healthy"
    if prs_fresh or ps_fresh:
        return "partial"
    return "stale"


def _interpret_freshness_verdict(
    verdict: str,
    prs_latest: Optional[date],
    ps_latest: Optional[date],
    now_et: datetime,
) -> str:
    """Human-readable summary for the freshness verdict."""
    if verdict == "missing":
        return ("No Layer 3 data found. Both player_rolling_stats and player_scores "
                "are empty. Has the daily ingestion job run?")
    if verdict == "healthy":
        return (f"Layer 3 pipeline is healthy. Latest rolling_stats: {_d(prs_latest)}, "
                f"latest scores: {_d(ps_latest)}. Both tables are fresh.")
    if verdict == "partial":
        missing = []
        if prs_latest is None:
            missing.append("player_rolling_stats")
        if ps_latest is None:
            missing.append("player_scores")
        if missing:
            return (f"Layer 3 pipeline is partial. Missing data from: {', '.join(missing)}. "
                    "Check if scheduled jobs completed.")
        # One is stale, one is fresh
        return (f"Layer 3 pipeline shows partial freshness. One table is stale. "
                f"rolling_stats: {_d(prs_latest)}, scores: {_d(ps_latest)}.")
    return (f"Layer 3 pipeline is STALE. Latest rolling_stats: {_d(prs_latest)}, "
            f"latest scores: {_d(ps_latest)}. Data is 2+ days old. Check scheduled jobs.")


def _format_audit_log(row) -> dict:
    """Format a data_ingestion_logs row for JSON response."""
    return {
        "id": _i(row.id),
        "job_type": row.job_type,
        "target_date": _d(row.target_date),
        "status": row.status,
        "records_processed": _i(row.records_processed),
        "records_failed": _i(row.records_failed),
        "processing_time_seconds": _f(row.processing_time_seconds),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


# ---------------------------------------------------------------------------
# 5. Decision Pipeline Freshness -- P17-P19 observability
# ---------------------------------------------------------------------------

@router.get("/diagnose-decision/pipeline-freshness")
def diagnose_decision_pipeline_freshness():
    """
    Decision pipeline freshness and coverage observability (P17-P19).

    Answers:
      - What is the latest as_of_date for decision_results (P17)?
      - What is the latest as_of_date for decision_explanations (P19)?
      - How many rows exist per decision_type on the latest dates?
      - Is the pipeline healthy, stale, partial, or missing?

    Use this for operational monitoring of the backend decision pipeline.
    """
    db = SessionLocal()
    try:
        now_et = _now_et()

        # --- Latest as_of_date for each table -------------------------------
        dr_latest_row = db.execute(text("""
            SELECT MAX(as_of_date) AS latest_date FROM decision_results
        """)).fetchone()

        de_latest_row = db.execute(text("""
            SELECT MAX(as_of_date) AS latest_date FROM decision_explanations
        """)).fetchone()

        dr_latest = dr_latest_row.latest_date if dr_latest_row else None
        de_latest = de_latest_row.latest_date if de_latest_row else None

        # --- Row counts by decision_type for latest dates ------------------
        dr_counts = {}
        de_counts = {}

        for dtype in ("lineup", "waiver"):
            if dr_latest:
                cnt = db.execute(text("""
                    SELECT COUNT(*) AS n FROM decision_results
                    WHERE as_of_date = :d AND decision_type = :dt
                """), {"d": dr_latest, "dt": dtype}).fetchone()
                dr_counts[dtype] = _i(cnt.n) if cnt else 0
            else:
                dr_counts[dtype] = None

            if de_latest:
                cnt = db.execute(text("""
                    SELECT COUNT(*) AS n FROM decision_explanations
                    WHERE as_of_date = :d AND decision_type = :dt
                """), {"d": de_latest, "dt": dtype}).fetchone()
                de_counts[dtype] = _i(cnt.n) if cnt else 0
            else:
                de_counts[dtype] = None

        # --- Total row counts for latest dates ------------------------------
        dr_total = _i(sum(v for v in dr_counts.values() if v is not None)) if dr_latest else None
        de_total = _i(sum(v for v in de_counts.values() if v is not None)) if de_latest else None

        # --- Latest computed_at timestamps ---------------------------------
        dr_computed_row = db.execute(text("""
            SELECT MAX(computed_at) AS latest_computed FROM decision_results
        """)).fetchone()

        de_computed_row = db.execute(text("""
            SELECT MAX(computed_at) AS latest_computed FROM decision_explanations
        """)).fetchone()

        dr_computed = dr_computed_row.latest_computed if dr_computed_row else None
        de_computed = de_computed_row.latest_computed if de_computed_row else None

        # --- Freshness verdict --------------------------------------------
        # Expected schedule: decision_results ~7 AM ET, decision_explanations ~9 AM ET
        # Consider data stale if latest as_of_date is 2+ days old
        today_et = now_et.date()
        stale_threshold = today_et - timedelta(days=2)

        dr_is_fresh = dr_latest and dr_latest >= stale_threshold
        de_is_fresh = de_latest and de_latest >= stale_threshold

        verdict = _compute_decision_freshness_verdict(
            dr_is_fresh, de_is_fresh, dr_latest, de_latest
        )

        return {
            "checked_at": now_et.isoformat(),
            "verdict": verdict,
            "message": _interpret_decision_freshness_verdict(
                verdict, dr_latest, de_latest, now_et
            ),
            "decision_results": {
                "latest_as_of_date": _d(dr_latest),
                "latest_computed_at": dr_computed.isoformat() if dr_computed else None,
                "total_row_count": dr_total,
                "breakdown_by_type": dr_counts,
            },
            "decision_explanations": {
                "latest_as_of_date": _d(de_latest),
                "latest_computed_at": de_computed.isoformat() if de_computed else None,
                "total_row_count": de_total,
                "breakdown_by_type": de_counts,
            },
            "schedule_expectations": {
                "decision_results_expected": "07:00 AM ET (daily)",
                "decision_explanations_expected": "09:00 AM ET (daily)",
                "timezone": "America/New_York",
            },
        }
    finally:
        db.close()


def _compute_decision_freshness_verdict(
    dr_fresh: Optional[bool],
    de_fresh: Optional[bool],
    dr_latest: Optional[date],
    de_latest: Optional[date],
) -> str:
    """
    Compute a freshness verdict for the decision pipeline (P17-P19).

    Returns one of: healthy, stale, partial, missing
    """
    if dr_latest is None and de_latest is None:
        return "missing"
    if dr_latest and de_latest is None:
        return "partial"
    if dr_latest is None and de_latest:
        return "partial"
    if dr_fresh and de_fresh:
        return "healthy"
    if dr_fresh or de_fresh:
        return "partial"
    return "stale"


def _interpret_decision_freshness_verdict(
    verdict: str,
    dr_latest: Optional[date],
    de_latest: Optional[date],
    now_et: datetime,
) -> str:
    """Human-readable summary for the decision pipeline freshness verdict."""
    if verdict == "missing":
        return ("No decision pipeline data found. Both decision_results and "
                "decision_explanations are empty. Has the 7 AM/9 AM job run?")
    if verdict == "healthy":
        return (f"Decision pipeline is healthy. Latest results: {_d(dr_latest)}, "
                f"latest explanations: {_d(de_latest)}. Both tables are fresh.")
    if verdict == "partial":
        missing = []
        if dr_latest is None:
            missing.append("decision_results")
        if de_latest is None:
            missing.append("decision_explanations")
        if missing:
            return (f"Decision pipeline is partial. Missing data from: {', '.join(missing)}. "
                    "Check if scheduled jobs completed.")
        # One is stale, one is fresh
        return (f"Decision pipeline shows partial freshness. One table is stale. "
                f"results: {_d(dr_latest)}, explanations: {_d(de_latest)}.")
    return (f"Decision pipeline is STALE. Latest results: {_d(dr_latest)}, "
            f"latest explanations: {_d(de_latest)}. Data is 2+ days old. Check scheduled jobs.")
