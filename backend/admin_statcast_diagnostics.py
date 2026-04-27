"""
Admin diagnostic endpoints for statcast_performances.

Built after the April 14 aggregation fix + re-backfill to verify that data is
actually populating correctly. Use these to answer questions like "why are
AB totals so low?" without guessing at which view/query the operator is
looking at.

Endpoints (all read-only, all GET):
  /admin/diagnose-statcast/summary
  /admin/diagnose-statcast/by-date
  /admin/diagnose-statcast/leaderboard?metric=ab&limit=20
  /admin/diagnose-statcast/player?name=Judge
  /admin/diagnose-statcast/player?player_id=592450
  /admin/diagnose-statcast/raw-sample?limit=5
  /admin/diagnose-statcast/sanity-check

REMOVE AFTER STATCAST VALIDATION COMPLETE.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from backend.models import SessionLocal

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_METRICS = {
    "ab", "pa", "h", "hr", "r", "rbi", "bb", "so", "hbp", "sb", "cs",
    "doubles", "triples", "ip", "er", "k_pit", "bb_pit", "pitches",
}


def _f(v):
    """Safe float coercion for JSON; None stays None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _d(v):
    """Date -> ISO string or None."""
    return v.isoformat() if v is not None else None


# ---------------------------------------------------------------------------
# 1. Summary — is the table populated at all?
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/summary")
def diagnose_statcast_summary():
    """
    Top-level health summary for statcast_performances.

    Answers: does the table have data? how fresh? how many distinct players?
    what's the row-count spread across dates?

    Use this FIRST when investigating "why are my queries returning nothing".
    """
    db = SessionLocal()
    try:
        totals = db.execute(text("""
            SELECT
                COUNT(*)                              AS total_rows,
                COUNT(DISTINCT player_id)             AS distinct_player_ids,
                COUNT(DISTINCT player_name)           AS distinct_player_names,
                COUNT(DISTINCT game_date)             AS distinct_game_dates,
                MIN(game_date)                        AS min_game_date,
                MAX(game_date)                        AS max_game_date
            FROM statcast_performances
        """)).fetchone()

        zero_metric_rows = db.execute(text("""
            SELECT COUNT(*) FROM statcast_performances
            WHERE COALESCE(exit_velocity_avg, 0) = 0
              AND COALESCE(xwoba, 0) = 0
              AND COALESCE(hard_hit_pct, 0) = 0
              AND COALESCE(barrel_pct, 0) = 0
        """)).scalar()

        zero_ab_rows = db.execute(text(
            "SELECT COUNT(*) FROM statcast_performances WHERE COALESCE(ab, 0) = 0"
        )).scalar()

        pitcher_rows = db.execute(text(
            "SELECT COUNT(*) FROM statcast_performances WHERE COALESCE(ip, 0) > 0"
        )).scalar()

        # player_id format distribution — are we storing numeric mlbam_ids or text names?
        pid_formats = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE player_id ~ '^[0-9]+$')      AS numeric_ids,
                COUNT(*) FILTER (WHERE player_id !~ '^[0-9]+$' AND player_id <> '') AS non_numeric_ids,
                COUNT(*) FILTER (WHERE player_id IS NULL OR player_id = '')         AS empty_ids
            FROM statcast_performances
        """)).fetchone()

        # Sample a few player_id values so Gemini can eyeball the format
        pid_samples = db.execute(text("""
            SELECT DISTINCT player_id, player_name
            FROM statcast_performances
            ORDER BY player_id
            LIMIT 10
        """)).fetchall()

        total = totals.total_rows or 0
        return {
            "totals": {
                "total_rows": total,
                "distinct_player_ids": totals.distinct_player_ids,
                "distinct_player_names": totals.distinct_player_names,
                "distinct_game_dates": totals.distinct_game_dates,
                "min_game_date": _d(totals.min_game_date),
                "max_game_date": _d(totals.max_game_date),
            },
            "row_categories": {
                "zero_quality_metric_rows": zero_metric_rows,
                "zero_quality_metric_pct": round(100 * zero_metric_rows / total, 2) if total else None,
                "zero_ab_rows": zero_ab_rows,
                "pitcher_rows_ip_gt_0": pitcher_rows,
            },
            "player_id_format": {
                "numeric_ids": pid_formats.numeric_ids,
                "non_numeric_ids": pid_formats.non_numeric_ids,
                "empty_ids": pid_formats.empty_ids,
            },
            "player_id_samples": [
                {"player_id": r.player_id, "player_name": r.player_name}
                for r in pid_samples
            ],
            "interpretation": (
                "If total_rows==0 the backfill did not populate. "
                "If numeric_ids is high and non_numeric_ids is 0, ids are Statcast mlbam_ids. "
                "If distinct_player_names >> distinct_player_ids there is name-resolution collision."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"summary failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 2. By-date — row counts per game_date, reveals aggregation health
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/by-date")
def diagnose_statcast_by_date(limit: int = Query(30, ge=1, le=365)):
    """
    Rows-per-game_date histogram. After the aggregation fix, each date should
    have ~500-900 rows (1 per player-game). If a date has 5,000+ rows, the
    per-pitch overwrite bug is back.
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                game_date,
                COUNT(*)                        AS row_count,
                COUNT(DISTINCT player_id)       AS unique_players,
                SUM(ab)                         AS total_ab,
                SUM(pa)                         AS total_pa,
                SUM(h)                          AS total_h,
                SUM(sb)                         AS total_sb,
                SUM(cs)                         AS total_cs,
                SUM(CASE WHEN ip > 0 THEN 1 ELSE 0 END) AS pitcher_rows,
                MAX(ab)                         AS max_single_game_ab,
                MAX(pa)                         AS max_single_game_pa
            FROM statcast_performances
            GROUP BY game_date
            ORDER BY game_date DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()

        return {
            "row_count": len(rows),
            "dates": [
                {
                    "game_date": _d(r.game_date),
                    "row_count": r.row_count,
                    "unique_players": r.unique_players,
                    "total_ab": r.total_ab,
                    "total_pa": r.total_pa,
                    "total_h": r.total_h,
                    "total_sb": r.total_sb,
                    "total_cs": r.total_cs,
                    "pitcher_rows": r.pitcher_rows,
                    "max_single_game_ab": r.max_single_game_ab,
                    "max_single_game_pa": r.max_single_game_pa,
                }
                for r in rows
            ],
            "interpretation": (
                "Normal full-slate MLB day: 500-900 rows, 500-900 unique players, "
                "max_single_game_ab in 5-7. row_count >> unique_players means "
                "per-pitch rows snuck through aggregation. max_single_game_ab > 10 "
                "means the same player collided across fetches."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"by-date failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Leaderboard — cumulative stats across game_dates
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/leaderboard")
def diagnose_statcast_leaderboard(
    metric: str = Query("ab", description=f"One of: {sorted(_ALLOWED_METRICS)}"),
    limit: int = Query(20, ge=1, le=200),
):
    """
    Top-N players by cumulative metric across all game_dates.

    Use this to answer "is AB accurate?" — if the top player has ~50-65 AB
    through mid-April, data is correct. If the top is 6, aggregation is broken.
    """
    if metric not in _ALLOWED_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"metric must be one of: {sorted(_ALLOWED_METRICS)}",
        )
    db = SessionLocal()
    try:
        # Safe: metric is validated against _ALLOWED_METRICS whitelist
        rows = db.execute(text(f"""
            SELECT
                player_id,
                MAX(player_name)                AS player_name,
                MAX(team)                       AS team,
                COUNT(*)                        AS games,
                SUM({metric})                   AS total_{metric},
                SUM(ab)                         AS total_ab,
                SUM(h)                          AS total_h,
                SUM(pa)                         AS total_pa,
                SUM(sb)                         AS total_sb,
                SUM(cs)                         AS total_cs,
                MIN(game_date)                  AS first_game,
                MAX(game_date)                  AS last_game
            FROM statcast_performances
            GROUP BY player_id
            ORDER BY SUM({metric}) DESC NULLS LAST
            LIMIT :limit
        """), {"limit": limit}).fetchall()

        return {
            "metric": metric,
            "limit": limit,
            "row_count": len(rows),
            "leaderboard": [
                {
                    "player_id": r.player_id,
                    "player_name": r.player_name,
                    "team": r.team,
                    "games": r.games,
                    f"total_{metric}": getattr(r, f"total_{metric}"),
                    "total_ab": r.total_ab,
                    "total_h": r.total_h,
                    "total_pa": r.total_pa,
                    "total_sb": r.total_sb,
                    "total_cs": r.total_cs,
                    "first_game": _d(r.first_game),
                    "last_game": _d(r.last_game),
                }
                for r in rows
            ],
            "interpretation": (
                "Top batters through ~3 weeks of MLB season should have ~50-70 AB "
                "over 15-20 games. Top-of-leaderboard AB under 20 with games<5 "
                "means only a few dates ingested. Top AB under 10 with games>=10 "
                "means per-game aggregation silently dropped counts."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"leaderboard failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. Player deep-dive — all rows + cumulative for one player
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/player")
def diagnose_statcast_player(
    name: Optional[str] = Query(None, description="Substring match on player_name (case-insensitive)"),
    player_id: Optional[str] = Query(None, description="Exact match on player_id"),
):
    """
    Full per-game history + cumulative totals for a single player.

    Supply either ?name=Judge (ILIKE match) or ?player_id=592450 (exact).
    Returns every stored row plus a cumulative summary so you can eyeball
    whether individual game rows look sane AND whether totals make sense.
    """
    if not name and not player_id:
        raise HTTPException(status_code=400, detail="Supply ?name= or ?player_id=")

    db = SessionLocal()
    try:
        if player_id:
            where = "player_id = :key"
            params = {"key": player_id}
        else:
            where = "player_name ILIKE :key"
            params = {"key": f"%{name}%"}

        matches = db.execute(text(f"""
            SELECT DISTINCT player_id, player_name, team
            FROM statcast_performances
            WHERE {where}
            ORDER BY player_name
            LIMIT 20
        """), params).fetchall()

        if not matches:
            return {
                "query": {"name": name, "player_id": player_id},
                "matches": [],
                "message": "No rows matched. Try /admin/diagnose-statcast/summary first.",
            }

        results = []
        for m in matches:
            per_game = db.execute(text("""
                SELECT
                    game_date, team, pa, ab, h, hr, r, rbi, bb, so, sb, cs,
                    ip, er, k_pit, bb_pit,
                    exit_velocity_avg, hard_hit_pct, barrel_pct, xwoba
                FROM statcast_performances
                WHERE player_id = :pid
                ORDER BY game_date
            """), {"pid": m.player_id}).fetchall()

            cum = db.execute(text("""
                SELECT
                    COUNT(*)        AS games,
                    SUM(pa)         AS pa,
                    SUM(ab)         AS ab,
                    SUM(h)          AS h,
                    SUM(hr)         AS hr,
                    SUM(r)          AS r,
                    SUM(rbi)        AS rbi,
                    SUM(bb)         AS bb,
                    SUM(so)         AS so,
                    SUM(sb)         AS sb,
                    SUM(cs)         AS cs,
                    SUM(ip)         AS ip,
                    SUM(er)         AS er,
                    SUM(k_pit)      AS k_pit,
                    SUM(bb_pit)     AS bb_pit,
                    AVG(NULLIF(xwoba, 0))           AS avg_xwoba,
                    AVG(NULLIF(exit_velocity_avg, 0)) AS avg_exit_velo,
                    MIN(game_date)  AS first_game,
                    MAX(game_date)  AS last_game
                FROM statcast_performances
                WHERE player_id = :pid
            """), {"pid": m.player_id}).fetchone()

            results.append({
                "player_id": m.player_id,
                "player_name": m.player_name,
                "team": m.team,
                "cumulative": {
                    "games": cum.games,
                    "pa": cum.pa,
                    "ab": cum.ab,
                    "h": cum.h,
                    "hr": cum.hr,
                    "r": cum.r,
                    "rbi": cum.rbi,
                    "bb": cum.bb,
                    "so": cum.so,
                    "sb": cum.sb,
                    "cs": cum.cs,
                    "ip": _f(cum.ip),
                    "er": cum.er,
                    "k_pit": cum.k_pit,
                    "bb_pit": cum.bb_pit,
                    "avg_xwoba": _f(cum.avg_xwoba),
                    "avg_exit_velo": _f(cum.avg_exit_velo),
                    "first_game": _d(cum.first_game),
                    "last_game": _d(cum.last_game),
                    "computed_avg": (cum.h / cum.ab) if cum.ab else None,
                },
                "per_game": [
                    {
                        "game_date": _d(g.game_date),
                        "team": g.team,
                        "pa": g.pa, "ab": g.ab, "h": g.h, "hr": g.hr,
                        "r": g.r, "rbi": g.rbi, "bb": g.bb, "so": g.so,
                        "sb": g.sb, "cs": g.cs,
                        "ip": _f(g.ip), "er": g.er, "k_pit": g.k_pit, "bb_pit": g.bb_pit,
                        "exit_velocity_avg": _f(g.exit_velocity_avg),
                        "hard_hit_pct": _f(g.hard_hit_pct),
                        "barrel_pct": _f(g.barrel_pct),
                        "xwoba": _f(g.xwoba),
                    }
                    for g in per_game
                ],
            })

        return {
            "query": {"name": name, "player_id": player_id},
            "match_count": len(matches),
            "players": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"player lookup failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. Raw sample — return N recent rows verbatim for eyeball inspection
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/raw-sample")
def diagnose_statcast_raw_sample(limit: int = Query(5, ge=1, le=50)):
    """
    Return the most recent N rows in full so Gemini can eyeball what a
    row actually looks like. Includes every column.
    """
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT
                id, player_id, player_name, team, game_date,
                pa, ab, h, doubles, triples, hr, r, rbi, bb, so, hbp, sb, cs,
                exit_velocity_avg, launch_angle_avg, hard_hit_pct, barrel_pct,
                xba, xslg, xwoba, woba, avg, obp, slg, ops,
                ip, er, k_pit, bb_pit, pitches,
                created_at
            FROM statcast_performances
            ORDER BY game_date DESC, id DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()

        return {
            "row_count": len(rows),
            "rows": [
                {
                    "id": r.id,
                    "player_id": r.player_id,
                    "player_name": r.player_name,
                    "team": r.team,
                    "game_date": _d(r.game_date),
                    "pa": r.pa, "ab": r.ab, "h": r.h, "doubles": r.doubles,
                    "triples": r.triples, "hr": r.hr, "r": r.r, "rbi": r.rbi,
                    "bb": r.bb, "so": r.so, "hbp": r.hbp, "sb": r.sb, "cs": r.cs,
                    "exit_velocity_avg": _f(r.exit_velocity_avg),
                    "launch_angle_avg": _f(r.launch_angle_avg),
                    "hard_hit_pct": _f(r.hard_hit_pct),
                    "barrel_pct": _f(r.barrel_pct),
                    "xba": _f(r.xba), "xslg": _f(r.xslg), "xwoba": _f(r.xwoba),
                    "woba": _f(r.woba), "avg": _f(r.avg), "obp": _f(r.obp),
                    "slg": _f(r.slg), "ops": _f(r.ops),
                    "ip": _f(r.ip), "er": r.er, "k_pit": r.k_pit,
                    "bb_pit": r.bb_pit, "pitches": r.pitches,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"raw-sample failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 6. Sanity check — cross-table reality check
# ---------------------------------------------------------------------------

@router.get("/diagnose-statcast/sanity-check")
def diagnose_statcast_sanity_check():
    """
    Cross-table cross-check: does statcast_performances line up with
    mlb_player_stats (BDL) on the same date range?

    If BDL shows 60 cumulative AB for a common player but Statcast shows 6,
    Statcast aggregation is broken (or partial). If both show similar totals,
    data is accurate.
    """
    db = SessionLocal()
    try:
        sp_range = db.execute(text("""
            SELECT MIN(game_date) AS min_d, MAX(game_date) AS max_d,
                   COUNT(*) AS total_rows
            FROM statcast_performances
        """)).fetchone()

        mps_range = db.execute(text("""
            SELECT MIN(game_date) AS min_d, MAX(game_date) AS max_d,
                   COUNT(*) AS total_rows
            FROM mlb_player_stats
        """)).fetchone()

        # Pick a few well-known players by name substring to compare.
        # Names are approximate — we just want any hit in both tables to
        # run a side-by-side AB comparison.
        comparison_names = ["Judge", "Ohtani", "Betts", "Acuna", "Trout", "Soto"]
        comparisons = []
        for name in comparison_names:
            sp = db.execute(text("""
                SELECT player_name, COUNT(*) AS games, SUM(ab) AS ab, SUM(h) AS h
                FROM statcast_performances
                WHERE player_name ILIKE :n
                GROUP BY player_name
                ORDER BY SUM(ab) DESC NULLS LAST
                LIMIT 1
            """), {"n": f"%{name}%"}).fetchone()

            mps = db.execute(text("""
                SELECT COUNT(*) AS games, SUM(mps.ab) AS ab, SUM(mps.hits) AS h
                FROM mlb_player_stats mps
                JOIN player_id_mapping pim ON pim.bdl_id = mps.bdl_player_id
                WHERE pim.full_name ILIKE :n
            """), {"n": f"%{name}%"}).fetchone()

            comparisons.append({
                "search_name": name,
                "statcast": {
                    "player_name": sp.player_name if sp else None,
                    "games": sp.games if sp else 0,
                    "ab": sp.ab if sp else 0,
                    "h": sp.h if sp else 0,
                },
                "mlb_player_stats_bdl": {
                    "games": mps.games if mps else 0,
                    "ab": mps.ab if mps else 0,
                    "h": mps.h if mps else 0,
                },
                "ab_delta": (
                    (sp.ab or 0) - (mps.ab or 0)
                    if sp and mps else None
                ),
            })

        return {
            "statcast_performances": {
                "total_rows": sp_range.total_rows,
                "min_game_date": _d(sp_range.min_d),
                "max_game_date": _d(sp_range.max_d),
            },
            "mlb_player_stats": {
                "total_rows": mps_range.total_rows,
                "min_game_date": _d(mps_range.min_d),
                "max_game_date": _d(mps_range.max_d),
            },
            "comparisons": comparisons,
            "interpretation": (
                "BDL (mlb_player_stats) is the trusted primary for AB/H. "
                "If statcast AB == BDL AB within 1-2 per player, Statcast is healthy. "
                "If Statcast is systematically lower, aggregation is dropping rows. "
                "Empty Statcast column with populated BDL column means the "
                "statcast_performances table is not populated for those players/dates."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sanity-check failed: {e}")
    finally:
        db.close()
