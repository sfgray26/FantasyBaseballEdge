"""
rolling_window_engine.py -- P13 Derived Stats: Exponential Decay Rolling Windows

Pure computation module. Zero I/O.
All functions are stateless transforms: ORM rows -> RollingWindowResult objects.

Decay formula:
    weight = decay_lambda ** days_back
    where days_back = (as_of_date - game_date).days

Innings pitched parsing:
    BDL stores "6.2" meaning 6 innings + 2 outs = 6.667 decimal (NOT 6.2).
    The fractional part represents outs (0, 1, or 2), not a decimal fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# IP parser
# ---------------------------------------------------------------------------

def parse_ip(ip_str: Optional[str]) -> Optional[float]:
    """
    Convert BDL innings pitched string to decimal float.

    BDL uses "baseball notation" where the digit after the decimal point
    represents OUTS (0, 1, or 2), not a decimal fraction.

    Examples:
        "6.2" -> 6.667  (6 innings + 2 outs = 6 + 2/3)
        "0.1" -> 0.333  (0 innings + 1 out  = 1/3)
        "9.0" -> 9.0    (9 full innings, 0 outs)
        "9"   -> 9.0    (no decimal -> full innings)
        "0.0" -> 0.0
        None  -> None

    Returns None if ip_str is None or unparseable.
    """
    if ip_str is None:
        return None

    ip_str = ip_str.strip()
    if not ip_str:
        return None

    try:
        if "." in ip_str:
            parts = ip_str.split(".", 1)
            innings = int(parts[0])
            outs = int(parts[1])
            # outs must be 0, 1, or 2 in baseball notation
            return innings + outs / 3.0
        else:
            return float(int(ip_str))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RollingWindowResult:
    """
    Decay-weighted rolling window metrics for one player over one window.

    Batting fields are None for pure pitchers (no at-bats in window).
    Pitching fields are None for pure hitters (no innings pitched in window).
    Two-way players (e.g. Ohtani) will have both sets populated.
    """
    bdl_player_id: int
    as_of_date: date
    window_days: int
    games_in_window: int
    w_games: float = 0.0  # M3 fix: sum of decay weights (effective weighted game count)

    # Batting -- decay-weighted sums
    w_ab: Optional[float] = None
    w_hits: Optional[float] = None
    w_doubles: Optional[float] = None
    w_triples: Optional[float] = None
    w_home_runs: Optional[float] = None
    w_rbi: Optional[float] = None
    w_walks: Optional[float] = None
    w_strikeouts_bat: Optional[float] = None
    w_stolen_bases: Optional[float] = None
    w_caught_stealing: Optional[float] = None       # P27 NSB support
    w_net_stolen_bases: Optional[float] = None      # P27 w_stolen_bases - w_caught_stealing

    # Batting derived rates (from weighted sums)
    w_avg: Optional[float] = None       # w_hits / w_ab
    w_obp: Optional[float] = None       # (w_hits + w_walks) / (w_ab + w_walks)
    w_slg: Optional[float] = None       # weighted TB / w_ab
    w_ops: Optional[float] = None       # w_obp + w_slg

    # Pitching -- decay-weighted sums
    w_ip: Optional[float] = None            # sum of parse_ip(ip) * weight
    w_earned_runs: Optional[float] = None
    w_hits_allowed: Optional[float] = None
    w_walks_allowed: Optional[float] = None
    w_strikeouts_pit: Optional[float] = None

    # Pitching derived rates
    w_era: Optional[float] = None       # 9 * w_earned_runs / w_ip
    w_whip: Optional[float] = None      # (w_hits_allowed + w_walks_allowed) / w_ip
    w_k_per_9: Optional[float] = None   # 9 * w_strikeouts_pit / w_ip

    # Statcast advanced metrics (P28 Phase 1)
    w_exit_velocity_avg: Optional[float] = None  # Avg exit velocity (mph)
    w_launch_angle_avg: Optional[float] = None   # Avg launch angle (degrees)
    w_hard_hit_pct: Optional[float] = None       # % batted balls >= 95 mph
    w_barrel_pct: Optional[float] = None         # % ideal EV + LA combinations
    w_xwoba: Optional[float] = None              # Expected wOBA
    w_xba: Optional[float] = None                # Expected batting average
    w_xslg: Optional[float] = None               # Expected slugging
    w_xwoba_minus_woba: Optional[float] = None   # Luck differential (xwOBA - wOBA)


# ---------------------------------------------------------------------------
# Core rolling window computation
# ---------------------------------------------------------------------------

def compute_rolling_window(
    stat_rows: list,
    as_of_date: date,
    window_days: int,
    decay_lambda: float = 0.95,
) -> Optional[RollingWindowResult]:
    """
    Compute decay-weighted rolling window metrics for one player.

    Args:
        stat_rows:    ORM rows for ONE player, any date order.
        as_of_date:   Window ends on this date (inclusive).
        window_days:  Look back N days from as_of_date.
        decay_lambda: Exponential decay per day (0.95 = 5% decay per day).

    Returns:
        RollingWindowResult if player has >= 1 game in window, else None.

    Weight formula:
        days_back = (as_of_date - game_date).days
        weight    = decay_lambda ** days_back

    Games on as_of_date: days_back=0, weight=1.0
    Games 1 day back:    weight=0.95
    Games 7 days back:   weight=0.95^7 ~= 0.698

    Rate stat derivation (batting):
        w_avg = sum(w*hits)  / sum(w*ab)              if sum(w*ab) > 0
        singles = hits - doubles - triples - hr
        TB      = singles + 2*doubles + 3*triples + 4*hr
        w_slg   = sum(w*TB) / sum(w*ab)               if sum(w*ab) > 0
        approx PA = ab + walks (no HBP/SF data)
        w_obp   = sum(w*(hits+walks)) / sum(w*(ab+walks))
        w_ops   = w_obp + w_slg

    Rate stat derivation (pitching):
        w_era    = 9 * sum(w*er)                / sum(w*ip_decimal)   if ip > 0
        w_whip   = sum(w*(h_allowed+bb_allowed)) / sum(w*ip_decimal)  if ip > 0
        w_k_per9 = 9 * sum(w*strikeouts_pit)    / sum(w*ip_decimal)  if ip > 0
    """
    # Filter to window: include games where days_back in [0, window_days)
    # days_back = 0 means the game is on as_of_date (weight = 1.0)
    # days_back = window_days would be excluded (too old)
    window_rows = []
    for row in stat_rows:
        gd = row.game_date
        if gd is None:
            continue
        days_back = (as_of_date - gd).days
        if 0 <= days_back < window_days:
            window_rows.append((days_back, row))

    if not window_rows:
        return None

    # Batting accumulators
    sum_w_ab = 0.0
    sum_w_hits = 0.0
    sum_w_doubles = 0.0
    sum_w_triples = 0.0
    sum_w_hr = 0.0
    sum_w_rbi = 0.0
    sum_w_walks = 0.0
    sum_w_so_bat = 0.0
    sum_w_sb = 0.0
    sum_w_cs = 0.0               # P27 decay-weighted caught stealing
    sum_w_tb = 0.0               # total bases (for slg)
    sum_w_obp_num = 0.0          # hits + walks (numerator)
    sum_w_obp_den = 0.0          # ab + walks (denominator)
    has_batting = False

    # Pitching accumulators
    sum_w_ip = 0.0
    sum_w_er = 0.0
    sum_w_h_allowed = 0.0
    sum_w_bb_allowed = 0.0
    sum_w_k_pit = 0.0
    has_pitching = False

    games_in_window = len(window_rows)

    # M3 fix: sum of decay weights for consistent rate computation
    sum_weights = 0.0

    for days_back, row in window_rows:
        w = decay_lambda ** days_back
        sum_weights += w

        # Batting
        ab = row.ab
        hits = row.hits
        doubles = row.doubles
        triples = row.triples
        hr = row.home_runs
        rbi = row.rbi
        walks = row.walks
        so_bat = row.strikeouts_bat
        sb = row.stolen_bases
        # P27 NSB: caught_stealing may be missing on older rows; coerce None -> 0.
        # The statcast CS backfill defaults CS to 0 for rows without CS events,
        # so absent data is treated as "no caught stealings in window", which is
        # correct for the BDL feed (which does populate CS when present).
        cs = getattr(row, "caught_stealing", None)

        if ab is not None:
            has_batting = True
            _ab = float(ab)
            _hits = float(hits) if hits is not None else 0.0
            _doubles = float(doubles) if doubles is not None else 0.0
            _triples = float(triples) if triples is not None else 0.0
            _hr = float(hr) if hr is not None else 0.0
            _rbi = float(rbi) if rbi is not None else 0.0
            _walks = float(walks) if walks is not None else 0.0
            _so_bat = float(so_bat) if so_bat is not None else 0.0
            _sb = float(sb) if sb is not None else 0.0
            _cs = float(cs) if cs is not None else 0.0

            singles = _hits - _doubles - _triples - _hr
            if singles < 0:
                singles = 0.0
            tb = singles + 2.0 * _doubles + 3.0 * _triples + 4.0 * _hr

            sum_w_ab += w * _ab
            sum_w_hits += w * _hits
            sum_w_doubles += w * _doubles
            sum_w_triples += w * _triples
            sum_w_hr += w * _hr
            sum_w_rbi += w * _rbi
            sum_w_walks += w * _walks
            sum_w_so_bat += w * _so_bat
            sum_w_sb += w * _sb
            sum_w_cs += w * _cs
            sum_w_tb += w * tb
            sum_w_obp_num += w * (_hits + _walks)
            sum_w_obp_den += w * (_ab + _walks)

        # Pitching
        ip_str = row.innings_pitched
        er = row.earned_runs
        h_allowed = row.hits_allowed
        bb_allowed = row.walks_allowed
        k_pit = row.strikeouts_pit

        if ip_str is not None:
            ip_decimal = parse_ip(ip_str)
            if ip_decimal is not None and ip_decimal >= 0:
                has_pitching = True
                _er = float(er) if er is not None else 0.0
                _h_allowed = float(h_allowed) if h_allowed is not None else 0.0
                _bb_allowed = float(bb_allowed) if bb_allowed is not None else 0.0
                _k_pit = float(k_pit) if k_pit is not None else 0.0

                sum_w_ip += w * ip_decimal
                sum_w_er += w * _er
                sum_w_h_allowed += w * _h_allowed
                sum_w_bb_allowed += w * _bb_allowed
                sum_w_k_pit += w * _k_pit

    result = RollingWindowResult(
        bdl_player_id=window_rows[0][1].bdl_player_id,
        as_of_date=as_of_date,
        window_days=window_days,
        games_in_window=games_in_window,
        w_games=sum_weights,
    )

    # Batting weighted sums
    if has_batting:
        result.w_ab = sum_w_ab
        result.w_hits = sum_w_hits
        result.w_doubles = sum_w_doubles
        result.w_triples = sum_w_triples
        result.w_home_runs = sum_w_hr
        result.w_rbi = sum_w_rbi
        result.w_walks = sum_w_walks
        result.w_strikeouts_bat = sum_w_so_bat
        result.w_stolen_bases = sum_w_sb
        # P27 NSB: always populate CS (defaults to 0 when not present) and
        # derive w_net_stolen_bases. Both fields travel with the batter-profile
        # half of the result so pure pitchers remain None for both.
        result.w_caught_stealing = sum_w_cs
        result.w_net_stolen_bases = sum_w_sb - sum_w_cs

        # Batting derived rates
        if sum_w_ab > 0:
            result.w_avg = sum_w_hits / sum_w_ab
            result.w_slg = sum_w_tb / sum_w_ab
        # else: w_avg and w_slg stay None (zero AB -- can't compute rate)

        if sum_w_obp_den > 0:
            result.w_obp = sum_w_obp_num / sum_w_obp_den
        # else: w_obp stays None

        if result.w_obp is not None and result.w_slg is not None:
            result.w_ops = result.w_obp + result.w_slg

    # Pitching weighted sums
    if has_pitching:
        result.w_ip = sum_w_ip
        result.w_earned_runs = sum_w_er
        result.w_hits_allowed = sum_w_h_allowed
        result.w_walks_allowed = sum_w_bb_allowed
        result.w_strikeouts_pit = sum_w_k_pit

        # Pitching derived rates
        if sum_w_ip > 0:
            result.w_era = 9.0 * sum_w_er / sum_w_ip
            result.w_whip = (sum_w_h_allowed + sum_w_bb_allowed) / sum_w_ip
            result.w_k_per_9 = 9.0 * sum_w_k_pit / sum_w_ip

    return result


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all_rolling_windows(
    all_stat_rows: list,
    as_of_date: date,
    window_sizes: list = None,
    decay_lambda: float = 0.95,
) -> list:
    """
    Compute rolling windows for all players x all window sizes.

    Groups stat_rows by bdl_player_id, then calls compute_rolling_window
    for each (player, window_size) pair.

    Returns a flat list of RollingWindowResult objects.
    Skips None results (players with 0 games in any given window).

    Args:
        all_stat_rows: Mixed players, any date order (ORM rows).
        as_of_date:    Compute windows ending on this date.
        window_sizes:  List of window day counts (default: [7, 14, 30]).
        decay_lambda:  Exponential decay per day (default: 0.95).
    """
    if window_sizes is None:
        window_sizes = [7, 14, 30]

    # Group by player
    players: dict[int, list] = {}
    for row in all_stat_rows:
        pid = row.bdl_player_id
        players.setdefault(pid, []).append(row)

    results: list[RollingWindowResult] = []
    for player_rows in players.values():
        for window in window_sizes:
            result = compute_rolling_window(
                player_rows,
                as_of_date=as_of_date,
                window_days=window,
                decay_lambda=decay_lambda,
            )
            if result is not None:
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Statcast-enhanced rolling windows (P28 Phase 1)
# ---------------------------------------------------------------------------

@dataclass
class StatcastDailyRow:
    """
    Lightweight dataclass for one player's Statcast data on one day.

    Used by compute_all_rolling_windows_with_statcast to merge Statcast
    metrics into rolling window computation without requiring a full ORM
    import at module top level (keeping the engine pure).
    """
    player_id: str          # e.g. "mlbam:12345" or BDL player ID string
    game_date: date
    exit_velocity_avg: Optional[float] = None
    launch_angle_avg: Optional[float] = None
    hard_hit_pct: Optional[float] = None
    barrel_pct: Optional[float] = None
    xwoba: Optional[float] = None
    xba: Optional[float] = None
    xslg: Optional[float] = None
    woba: Optional[float] = None


def compute_rolling_window_with_statcast(
    stat_rows: list,
    statcast_rows: list[StatcastDailyRow],
    as_of_date: date,
    window_days: int,
    decay_lambda: float = 0.95,
) -> Optional[RollingWindowResult]:
    """
    Compute decay-weighted rolling window WITH Statcast advanced metrics.

    This is a thin wrapper around compute_rolling_window that additionally
    merges Statcast data (exit velocity, barrel%, xwOBA, etc.) into the
    result. Statcast rows are matched by (player_id, game_date) and
    decay-weighted using the same exponential formula.

    Args:
        stat_rows:       Traditional BDL box stats (same as compute_rolling_window).
        statcast_rows:   StatcastDailyRow objects for the same player/date range.
        as_of_date:      Window end date.
        window_days:     Look-back window.
        decay_lambda:    Exponential decay factor.

    Returns:
        RollingWindowResult with Statcast fields populated, or None if no
        games in window.
    """
    # Compute base rolling window from traditional stats
    result = compute_rolling_window(
        stat_rows,
        as_of_date=as_of_date,
        window_days=window_days,
        decay_lambda=decay_lambda,
    )
    if result is None:
        return None

    # Build lookup: (player_id, game_date) -> StatcastDailyRow
    statcast_lookup: dict[tuple[str, date], StatcastDailyRow] = {}
    for sc_row in statcast_rows:
        key = (sc_row.player_id, sc_row.game_date)
        statcast_lookup[key] = sc_row

    # Determine the player's ID string from BDL rows first
    player_id_str = None
    if stat_rows:
        first_row = stat_rows[0]
        for attr in ("bdl_player_id", "player_id", "mlbam_id"):
            val = getattr(first_row, attr, None)
            if val is not None:
                player_id_str = str(val)
                break

    if player_id_str is None:
        # No player ID available
        return result

    # Check if any Statcast rows actually match this player
    # Try multiple ID formats that might appear in Statcast data
    matching_statcast_rows = []
    for sc in statcast_rows:
        sc_pid = sc.player_id
        # Match exact string, or prefixed versions
        if sc_pid == player_id_str:
            matching_statcast_rows.append(sc)
        elif sc_pid.endswith(f":{player_id_str}"):
            matching_statcast_rows.append(sc)
        elif sc_pid == f"mlbam:{player_id_str}":
            matching_statcast_rows.append(sc)

    if not matching_statcast_rows:
        # No Statcast data for this specific player
        return result

    # Use the matching statcast rows going forward
    statcast_rows = matching_statcast_rows

    # Rebuild lookup with filtered rows — use normalized player_id_str as key
    statcast_lookup = {}
    for sc_row in statcast_rows:
        key = (player_id_str, sc_row.game_date)
        statcast_lookup[key] = sc_row

    # Accumulate decay-weighted Statcast metrics
    sum_w_ev = 0.0
    sum_w_la = 0.0
    sum_w_hh = 0.0
    sum_w_br = 0.0
    sum_w_xw = 0.0
    sum_w_xb = 0.0
    sum_w_xs = 0.0
    sum_w_wb = 0.0  # wOBA (for luck differential)
    sum_weights_statcast = 0.0
    n_statcast_games = 0

    for row in stat_rows:
        gd = getattr(row, "game_date", None)
        if gd is None:
            continue
        days_back = (as_of_date - gd).days
        if not (0 <= days_back < window_days):
            continue

        key = (player_id_str, gd)
        sc = statcast_lookup.get(key)
        if sc is None:
            continue

        # Only count Statcast rows that have at least one meaningful metric.
        # Treat all-None or all-zero as "no Statcast data for this game".
        vals = [sc.exit_velocity_avg, sc.launch_angle_avg, sc.hard_hit_pct,
                sc.barrel_pct, sc.xwoba, sc.xba, sc.xslg, sc.woba]
        if all(v is None or v == 0 for v in vals):
            continue

        w = decay_lambda ** days_back
        sum_weights_statcast += w
        n_statcast_games += 1

        if sc.exit_velocity_avg is not None:
            sum_w_ev += w * sc.exit_velocity_avg
        if sc.launch_angle_avg is not None:
            sum_w_la += w * sc.launch_angle_avg
        if sc.hard_hit_pct is not None:
            sum_w_hh += w * sc.hard_hit_pct
        if sc.barrel_pct is not None:
            sum_w_br += w * sc.barrel_pct
        if sc.xwoba is not None:
            sum_w_xw += w * sc.xwoba
        if sc.xba is not None:
            sum_w_xb += w * sc.xba
        if sc.xslg is not None:
            sum_w_xs += w * sc.xslg
        if sc.woba is not None:
            sum_w_wb += w * sc.woba

    if n_statcast_games == 0:
        # No Statcast data in window — return base result
        return result

    # Populate Statcast fields
    if sum_weights_statcast > 0:
        result.w_exit_velocity_avg = sum_w_ev / sum_weights_statcast if sum_w_ev > 0 else None
        result.w_launch_angle_avg = sum_w_la / sum_weights_statcast if sum_w_la > 0 else None
        result.w_hard_hit_pct = sum_w_hh / sum_weights_statcast if sum_w_hh > 0 else None
        result.w_barrel_pct = sum_w_br / sum_weights_statcast if sum_w_br > 0 else None
        result.w_xwoba = sum_w_xw / sum_weights_statcast if sum_w_xw > 0 else None
        result.w_xba = sum_w_xb / sum_weights_statcast if sum_w_xb > 0 else None
        result.w_xslg = sum_w_xs / sum_weights_statcast if sum_w_xs > 0 else None

        # Luck differential: xwOBA - wOBA (positive = unlucky, negative = lucky)
        if result.w_xwoba is not None and sum_w_wb > 0:
            w_woba = sum_w_wb / sum_weights_statcast
            result.w_xwoba_minus_woba = result.w_xwoba - w_woba

    return result


def compute_all_rolling_windows_with_statcast(
    all_stat_rows: list,
    all_statcast_rows: list[StatcastDailyRow],
    as_of_date: date,
    window_sizes: list = None,
    decay_lambda: float = 0.95,
) -> list:
    """
    Compute Statcast-enhanced rolling windows for all players x all window sizes.

    Groups stat_rows by bdl_player_id, then matches Statcast rows by the
    same player identifier. Calls compute_rolling_window_with_statcast for
    each (player, window_size) pair.

    Returns a flat list of RollingWindowResult objects with Statcast fields
    populated where data is available.

    Args:
        all_stat_rows:       Traditional BDL box stats (mixed players).
        all_statcast_rows:   StatcastDailyRow objects (mixed players).
        as_of_date:          Compute windows ending on this date.
        window_sizes:        List of window day counts (default: [7, 14, 30]).
        decay_lambda:        Exponential decay per day (default: 0.95).
    """
    if window_sizes is None:
        window_sizes = [7, 14, 30]

    # Group BDL rows by player
    players_bdl: dict[int, list] = {}
    for row in all_stat_rows:
        pid = row.bdl_player_id
        players_bdl.setdefault(pid, []).append(row)

    # Group Statcast rows by player_id string
    players_statcast: dict[str, list[StatcastDailyRow]] = {}
    for sc_row in all_statcast_rows:
        players_statcast.setdefault(sc_row.player_id, []).append(sc_row)

    results: list[RollingWindowResult] = []

    for pid, bdl_rows in players_bdl.items():
        # Resolve Statcast rows for this player
        # Try multiple ID formats: raw bdl_player_id, "mlbam:{pid}", "{pid}"
        sc_rows: list[StatcastDailyRow] = []
        for key in [str(pid), f"mlbam:{pid}", f"bdl:{pid}"]:
            if key in players_statcast:
                sc_rows = players_statcast[key]
                break

        for window in window_sizes:
            result = compute_rolling_window_with_statcast(
                bdl_rows,
                sc_rows,
                as_of_date=as_of_date,
                window_days=window,
                decay_lambda=decay_lambda,
            )
            if result is not None:
                results.append(result)

    return results
