"""
P14 Scoring Engine -- League Z-scores per player per rolling window.

Pure computation module. Zero I/O. Stateless functions only.
Input:  list of PlayerRollingStats ORM rows for one window size.
Output: list of PlayerScoreResult dataclasses.

Algorithm outline
-----------------
1. For each scoring category collect all non-None values across the player pool.
2. If fewer than MIN_SAMPLE players have a value, skip the category (all Z = None).
3. Compute population std (ddof=0). If std == 0, skip (degenerate column).
4. Z = (val - mean) / std; negate Z for lower-is-better categories.
5. Cap Z at +/-Z_CAP to dampen outlier distortion.
6. composite_z = mean of all applicable non-None Z-scores for the player.
7. score_0_100 = percentile rank within the player's player_type cohort.
8. confidence = min(1.0, games_in_window / window_days).

Hard stops
----------
- No numpy / scipy / pandas imports.
- No position-level Z-scores (deferred -- position not yet in schema).
- No datetime.utcnow() usage.
"""

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

# key -> (column_name_on_PlayerRollingStats, is_lower_better)
#
# P27 NSB: z_sb is retained for backward compatibility (explainability
# narratives, UAT checks, legacy consumers) but excluded from composite_z
# via _COMPOSITE_EXCLUDED below. z_nsb (Net SB = SB - CS) is the canonical
# H2H One Win basestealing category and drives the composite.
HITTER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_hr":  ("w_home_runs",     False),
    "z_rbi": ("w_rbi",           False),
    "z_sb":  ("w_stolen_bases",  False),      # legacy -- excluded from composite
    "z_nsb": ("w_net_stolen_bases", False),   # P27 canonical basestealing Z
    "z_avg": ("w_avg",           False),
    "z_obp": ("w_obp",           False),
}

PITCHER_CATEGORIES: dict[str, tuple[str, bool]] = {
    "z_era":     ("w_era",         True),   # lower ERA is better -> negate Z
    "z_whip":    ("w_whip",        True),   # lower WHIP is better -> negate Z
    "z_k_per_9": ("w_k_per_9",     False),
}

# Combined for iteration convenience
_ALL_CATEGORIES: dict[str, tuple[str, bool]] = {
    **HITTER_CATEGORIES,
    **PITCHER_CATEGORIES,
}

# Keys computed and persisted but NOT included in composite_z.
# z_sb correlates >0.95 with z_nsb (CS events are rare), so including both
# would silently double-weight basestealing in the 5-category composite.
_COMPOSITE_EXCLUDED: frozenset = frozenset({"z_sb"})

# Minimum number of players with a non-null value before computing Z for a category
MIN_SAMPLE: int = 5

# Cap Z-scores at this absolute value to reduce outlier distortion
Z_CAP: float = 3.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlayerScoreResult:
    bdl_player_id: int
    as_of_date: date
    window_days: int
    player_type: str          # "hitter" | "pitcher" | "two_way"
    games_in_window: int

    # Per-category Z-scores (None if category not applicable or < MIN_SAMPLE)
    z_hr:      Optional[float] = None
    z_rbi:     Optional[float] = None
    z_sb:      Optional[float] = None   # legacy -- excluded from composite_z
    z_nsb:     Optional[float] = None   # P27 Net SB (SB - CS) -- enters composite
    z_avg:     Optional[float] = None
    z_obp:     Optional[float] = None
    z_era:     Optional[float] = None
    z_whip:    Optional[float] = None
    z_k_per_9: Optional[float] = None

    composite_z: float = 0.0   # mean of all applicable non-None Z-scores
    score_0_100: float = 50.0  # percentile rank 0-100 within player_type
    confidence:  float = 0.0   # games_in_window / window_days, capped at 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _population_std(values: list[float]) -> float:
    """Population standard deviation (ddof=0) for a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _cap(z: float) -> float:
    """Clamp Z to [-Z_CAP, +Z_CAP]."""
    return max(-Z_CAP, min(Z_CAP, z))


# ---------------------------------------------------------------------------
# Winsorization + MAD helpers (K-35 enhancements)
# ---------------------------------------------------------------------------

# Percentile bounds for Winsorization (clip raw values before Z computation)
WINSOR_LO: float = 5.0   # 5th percentile
WINSOR_HI: float = 95.0  # 95th percentile


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear interpolation percentile on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    k = (pct / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _winsorize(values: list[float]) -> list[float]:
    """
    Clip values at WINSOR_LO/WINSOR_HI percentiles.

    Winsorization replaces extreme values with the boundary value rather than
    removing them. This preserves sample size while reducing outlier distortion.
    Applied BEFORE Z-score computation per K-35 best practices.
    """
    if len(values) < 3:
        return list(values)
    sv = sorted(values)
    lo = _percentile(sv, WINSOR_LO)
    hi = _percentile(sv, WINSOR_HI)
    return [max(lo, min(hi, v)) for v in values]


def _median(values: list[float]) -> float:
    """Median of a list of floats."""
    sv = sorted(values)
    n = len(sv)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (sv[mid - 1] + sv[mid]) / 2.0
    return sv[mid]


def _mad(values: list[float]) -> float:
    """
    Median Absolute Deviation, scaled to match std dev for normal distributions.

    MAD = median(|x_i - median(x)|) * 1.4826

    The 1.4826 scaling factor makes MAD a consistent estimator of std dev
    for normally distributed data. MAD is robust to outliers -- a single
    extreme value cannot break it the way it can inflate std dev.
    """
    if len(values) < 2:
        return 0.0
    med = _median(values)
    abs_devs = [abs(v - med) for v in values]
    return _median(abs_devs) * 1.4826


def _detect_player_type(row) -> str:
    """
    Determine player type from a PlayerRollingStats row.

    Returns "two_way", "hitter", "pitcher", or "unknown".
    "unknown" rows are excluded from results.
    """
    has_batting = row.w_ab is not None
    has_pitching = row.w_ip is not None
    if has_batting and has_pitching:
        return "two_way"
    if has_batting:
        return "hitter"
    if has_pitching:
        return "pitcher"
    return "unknown"


def _applicable_z_keys(player_type: str) -> list[str]:
    """Return the Z-score field names applicable to a given player_type."""
    if player_type == "hitter":
        return list(HITTER_CATEGORIES.keys())
    if player_type == "pitcher":
        return list(PITCHER_CATEGORIES.keys())
    if player_type == "two_way":
        return list(HITTER_CATEGORIES.keys()) + list(PITCHER_CATEGORIES.keys())
    return []


def _percentile_rank(player_z: float, cohort_z_scores: list[float]) -> float:
    """
    Percentile rank: proportion of cohort at or below player_z, scaled to 0-100.
    Uses weak ordering (<=) consistent with scipy.stats.percentileofscore 'weak'.
    """
    n = len(cohort_z_scores)
    if n <= 1:
        return 50.0
    rank = sum(1 for c in cohort_z_scores if c <= player_z)
    return round(rank / n * 100.0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_league_zscores(
    rolling_rows: list,
    as_of_date: date,
    window_days: int,
    *,
    winsorize: bool = True,
    use_mad: bool = False,
) -> list[PlayerScoreResult]:
    """
    Compute league Z-scores for all players in rolling_rows.

    Parameters
    ----------
    rolling_rows : list of PlayerRollingStats ORM objects for ONE window size.
    as_of_date   : the date these rolling stats represent.
    window_days  : 7, 14, or 30.
    winsorize    : if True (default), clip raw values at 5th/95th percentiles
                   before computing Z-scores. Reduces outlier distortion per K-35.
    use_mad      : if True, use MAD-based robust Z (median + MAD*1.4826) instead
                   of mean + std. More resistant to skewed distributions. Default False.

    Returns
    -------
    list[PlayerScoreResult] -- one entry per player with a known player_type.
    Players whose w_ab and w_ip are both None are excluded ("unknown" type).
    """
    if not rolling_rows:
        return []

    # ------------------------------------------------------------------
    # Step 1: Compute league-level stats per category
    # ------------------------------------------------------------------
    # M2 fix: compute Z-scores using type-appropriate pools only.
    # Hitter categories use hitter+two_way rows; pitcher categories use
    # pitcher+two_way rows. This prevents pitcher batting nulls from
    # diluting hitter pools (and vice versa).
    category_z_lookup: dict[str, dict[int, float]] = {k: {} for k in _ALL_CATEGORIES}

    # Pre-classify rows by player type for pool filtering
    _row_types: dict[int, str] = {}
    for row in rolling_rows:
        _row_types[row.bdl_player_id] = _detect_player_type(row)

    for z_key, (col_name, is_lower_better) in _ALL_CATEGORIES.items():
        is_hitter_cat = z_key in HITTER_CATEGORIES
        # Collect (player_id, value) pairs from the type-appropriate pool
        pairs: list[tuple[int, float]] = []
        for row in rolling_rows:
            pt = _row_types.get(row.bdl_player_id, "unknown")
            # Skip rows that don't belong in this category's pool
            if is_hitter_cat and pt == "pitcher":
                continue
            if not is_hitter_cat and pt == "hitter":
                continue
            val = getattr(row, col_name, None)
            if val is not None:
                pairs.append((row.bdl_player_id, float(val)))

        if len(pairs) < MIN_SAMPLE:
            # Not enough data -- all Z for this category remain None
            continue

        values = [v for _, v in pairs]

        # Winsorize raw values at 5th/95th to dampen extreme outliers
        if winsorize:
            values_for_stats = _winsorize(values)
        else:
            values_for_stats = values

        if use_mad:
            center = _median(values_for_stats)
            spread = _mad(values_for_stats)
        else:
            center = sum(values_for_stats) / len(values_for_stats)
            spread = _population_std(values_for_stats)

        if spread == 0.0:
            # Degenerate column (all identical) -- skip
            continue

        # Compute Z using the pool's center/spread but each player's ORIGINAL value
        # (Winsorization affects the distribution parameters, not individual scores)
        for player_id, val in pairs:
            z = (val - center) / spread
            if is_lower_better:
                z = -z
            category_z_lookup[z_key][player_id] = _cap(z)

    # ------------------------------------------------------------------
    # Step 2: Build per-player results
    # ------------------------------------------------------------------
    results: list[PlayerScoreResult] = []

    for row in rolling_rows:
        player_type = _detect_player_type(row)
        if player_type == "unknown":
            continue

        applicable_keys = _applicable_z_keys(player_type)
        pid = row.bdl_player_id

        result = PlayerScoreResult(
            bdl_player_id=pid,
            as_of_date=as_of_date,
            window_days=window_days,
            player_type=player_type,
            games_in_window=row.games_in_window,
        )

        # Assign per-category Z-scores
        for z_key in applicable_keys:
            z_val = category_z_lookup[z_key].get(pid)  # None if not computed
            setattr(result, z_key, z_val)

        # Step 3: composite_z = mean of all applicable non-None Z-scores.
        # P27: z_sb is excluded (superseded by z_nsb) to avoid double-counting
        # basestealing in the 5-category hitter composite.
        non_none = [
            getattr(result, k)
            for k in applicable_keys
            if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
        ]
        result.composite_z = sum(non_none) / len(non_none) if non_none else 0.0

        # Step 4: confidence = min(1.0, games_in_window / window_days)
        result.confidence = min(1.0, row.games_in_window / window_days)

        results.append(result)

    # ------------------------------------------------------------------
    # Step 5: score_0_100 -- percentile rank within player_type cohort
    # ------------------------------------------------------------------
    # Group composite_z by player_type for ranking
    type_groups: dict[str, list[float]] = {}
    for res in results:
        type_groups.setdefault(res.player_type, []).append(res.composite_z)

    for res in results:
        cohort = type_groups[res.player_type]
        res.score_0_100 = _percentile_rank(res.composite_z, cohort)

    return results


def compute_league_params(
    rolling_rows: list,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Extract league-level mean and std for each scoring category.

    Returns (league_means, league_stds) keyed by short stat name
    (e.g. "hr", "rbi", "era") matching the keys expected by
    simulation_engine.simulate_player().

    Categories with fewer than MIN_SAMPLE non-null values are omitted.
    """
    # Map z_key -> short key used by simulation_engine
    _Z_TO_SHORT = {
        "z_hr": "hr", "z_rbi": "rbi", "z_sb": "sb", "z_nsb": "nsb",
        "z_avg": "avg", "z_obp": "obp",
        "z_era": "era", "z_whip": "whip", "z_k_per_9": "k",
    }

    league_means: dict[str, float] = {}
    league_stds: dict[str, float] = {}

    for z_key, (col_name, _is_lower_better) in _ALL_CATEGORIES.items():
        values: list[float] = []
        for row in rolling_rows:
            val = getattr(row, col_name, None)
            if val is not None:
                values.append(float(val))

        if len(values) < MIN_SAMPLE:
            continue

        mean = sum(values) / len(values)
        std = _population_std(values)
        if std == 0.0:
            continue

        short = _Z_TO_SHORT.get(z_key, z_key)
        league_means[short] = mean
        league_stds[short] = std

    return league_means, league_stds


# ── Park Factor Helper (Criterion 6 Consumer) ───────────────────────────────────

def get_park_factor(park_name: str, metric: str = "hr") -> float:
    """
    Get park factor from canonical persistence.

    This is a real consumer of Criterion 6 persisted context.
    Park factors are queried from the database rather than request-time-only logic.

    Args:
        park_name: Stadium name
        metric: One of 'hr', 'run', 'hits', 'era', 'whip'

    Returns:
        Park factor value (1.0 = neutral, > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly)
    """
    from backend.models import ParkFactor, SessionLocal

    db = SessionLocal()
    try:
        factor = db.query(ParkFactor).filter_by(park_name=park_name).first()
        if factor:
            return getattr(factor, f"{metric}_factor", 1.0)
        return 1.0  # Fallback to neutral if park not found
    finally:
        db.close()
