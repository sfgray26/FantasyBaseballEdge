"""
P17 -- Decision Engine: Lineup Optimization and Waiver Intelligence.

Pure-computation module (no DB imports, no side effects).
All imports are at module top level -- no imports inside functions.

Algorithm:
  - Lineup optimizer: greedy slot-filling ranked by composite lineup_score
  - Waiver intelligence: world-with vs world-without simulation using proj_p50 composites
  - Score formula: 0.6 * score_0_100 + 0.3 * momentum_bonus + 0.1 * proj_bonus
  - Momentum bonuses: SURGING=10, HOT=5, STABLE=0, COLD=-5, COLLAPSING=-10
  - Proj bonus: normalized proj_hr_p50 + proj_rbi_p50 (hitters) or proj_k_p50 (pitchers), scale 0-10

Output: LineupDecision and WaiverDecision dataclasses, plus a flat list of DecisionResult
        dataclasses suitable for upsert into decision_results.

ADR-004: Never import betting_model or analysis.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Roster slot definitions (Yahoo H2H standard)
# ---------------------------------------------------------------------------

ROSTER_SLOTS = {
    "C":    1,
    "1B":   1,
    "2B":   1,
    "3B":   1,
    "SS":   1,
    "OF":   3,
    "Util": 1,   # any hitter
    "SP":   2,
    "RP":   2,
    "P":    1,   # any pitcher
    "BN":   5,
}

HITTER_SLOTS = {"C", "1B", "2B", "3B", "SS", "OF", "Util"}
PITCHER_SLOTS = {"SP", "RP", "P"}
BENCH_SLOT = "BN"

# Momentum bonus table (additive, 0-10 scale)
_MOMENTUM_BONUS = {
    "SURGING":    10.0,
    "HOT":         5.0,
    "STABLE":      0.0,
    "COLD":       -5.0,
    "COLLAPSING": -10.0,
}

# Proj bonus scale cap -- raw rates are rescaled to 0-10 within each call
_PROJ_BONUS_MAX = 10.0

# Upper bounds used for normalization (typical full-season medians, not maxima)
_HR_NORM   = 30.0   # 30 HR projected ROS
_RBI_NORM  = 100.0  # 100 RBI projected ROS
_K_NORM    = 200.0  # 200 Ks projected ROS


# ---------------------------------------------------------------------------
# PlayerDecisionInput dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlayerDecisionInput:
    """
    Input record for one player entering the decision engine.

    Built by daily_ingestion._run_decision_optimization() from the join of
    player_scores + player_momentum + simulation_results for a single date.
    """
    bdl_player_id: int
    name: str
    player_type: str              # "hitter" | "pitcher" | "two_way" | "unknown"
    eligible_positions: list      # e.g. ["1B", "OF"]
    score_0_100: float
    composite_z: float
    momentum_signal: str          # SURGING / HOT / STABLE / COLD / COLLAPSING
    delta_z: float
    proj_hr_p50:   Optional[float] = None
    proj_rbi_p50:  Optional[float] = None
    proj_sb_p50:   Optional[float] = None
    proj_avg_p50:  Optional[float] = None
    proj_k_p50:    Optional[float] = None
    proj_era_p50:  Optional[float] = None
    proj_whip_p50: Optional[float] = None
    downside_p25:  Optional[float] = None
    upside_p75:    Optional[float] = None


# ---------------------------------------------------------------------------
# Output dataclasses (pure -- NOT the ORM)
# ---------------------------------------------------------------------------

@dataclass
class LineupDecision:
    """
    Greedy lineup optimization result.

    selected: {slot_name -> bdl_player_id}  (one entry per filled slot)
    bench:    [bdl_player_id, ...]           (up to ROSTER_SLOTS["BN"] players)
    score:    weighted sum of lineup_scores for selected players
    unrostered: players that could not be placed due to slot exhaustion
    """
    selected: dict = field(default_factory=dict)
    bench: list = field(default_factory=list)
    score: float = 0.0
    unrostered: list = field(default_factory=list)


@dataclass
class WaiverRecommendation:
    """Single add/drop pair with supporting metrics."""
    add_player_id:  int
    drop_player_id: int
    value_gain:     float
    confidence:     float
    reasoning:      str


@dataclass
class WaiverDecision:
    """
    Waiver intelligence result: ranked list of add/drop recommendations.

    recommendations: list of WaiverRecommendation, sorted by value_gain desc
    """
    recommendations: list = field(default_factory=list)


@dataclass
class DecisionResult:
    """
    Flat, per-player output row written to decision_results by daily_ingestion.

    decision_type: "lineup" | "waiver"
    bdl_player_id: primary player (started/added)
    target_slot:   roster slot for lineup decisions; None for waiver
    drop_player_id: waiver drop target; None for lineup
    lineup_score:  composite score used in lineup ranking
    value_gain:    value delta for waiver decisions
    confidence:    [0, 1] normalized certainty of recommendation
    reasoning:     ASCII one-liner explaining the recommendation
    """
    as_of_date:     date
    decision_type:  str
    bdl_player_id:  int
    target_slot:    Optional[str]
    drop_player_id: Optional[int]
    lineup_score:   Optional[float]
    value_gain:     Optional[float]
    confidence:     float
    reasoning:      str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _momentum_bonus(signal: str) -> float:
    """Return additive momentum bonus for a momentum signal string."""
    return _MOMENTUM_BONUS.get(signal.upper() if signal else "STABLE", 0.0)


def _proj_bonus(player: PlayerDecisionInput) -> float:
    """
    Return normalized projection bonus in [0, _PROJ_BONUS_MAX].

    Hitters: (proj_hr_p50 / _HR_NORM + proj_rbi_p50 / _RBI_NORM) * 5
    Pitchers: (proj_k_p50 / _K_NORM) * _PROJ_BONUS_MAX
    Two-way: average of both
    Unknown: 0
    """
    pt = (player.player_type or "unknown").lower()

    if pt == "hitter":
        hr_norm  = min((player.proj_hr_p50  or 0.0) / _HR_NORM,  1.0)
        rbi_norm = min((player.proj_rbi_p50 or 0.0) / _RBI_NORM, 1.0)
        return ((hr_norm + rbi_norm) / 2.0) * _PROJ_BONUS_MAX

    if pt == "pitcher":
        k_norm = min((player.proj_k_p50 or 0.0) / _K_NORM, 1.0)
        return k_norm * _PROJ_BONUS_MAX

    if pt == "two_way":
        hr_norm  = min((player.proj_hr_p50  or 0.0) / _HR_NORM,  1.0)
        rbi_norm = min((player.proj_rbi_p50 or 0.0) / _RBI_NORM, 1.0)
        k_norm   = min((player.proj_k_p50   or 0.0) / _K_NORM,   1.0)
        hitter_bonus = ((hr_norm + rbi_norm) / 2.0) * _PROJ_BONUS_MAX
        pitcher_bonus = k_norm * _PROJ_BONUS_MAX
        return (hitter_bonus + pitcher_bonus) / 2.0

    return 0.0


def _lineup_score(player: PlayerDecisionInput) -> float:
    """
    Composite lineup score:
        0.6 * score_0_100 + 0.3 * momentum_bonus_normalized + 0.1 * proj_bonus

    momentum_bonus is in [-10, 10]; we normalize to [0, 10] before applying the weight
    so the three components are all on similar 0-10 scales before weighting.
    """
    mb_raw = _momentum_bonus(player.momentum_signal)       # [-10, 10]
    mb_norm = (mb_raw + 10.0) / 2.0                        # -> [0, 10]
    pb = _proj_bonus(player)                               # [0, 10]
    score_component = player.score_0_100 / 10.0            # [0, 10]
    return 0.6 * score_component + 0.3 * mb_norm + 0.1 * pb


def _can_fill_slot(player: PlayerDecisionInput, slot: str) -> bool:
    """
    Return True if this player is eligible to fill the given roster slot.

    Util accepts any hitter; P accepts any pitcher.
    """
    positions = [p.upper() for p in (player.eligible_positions or [])]
    pt = (player.player_type or "unknown").lower()

    if slot in positions:
        return True
    if slot == "Util" and pt in ("hitter", "two_way"):
        return True
    if slot == "P" and pt in ("pitcher", "two_way"):
        return True
    return False


def _composite_value(player: PlayerDecisionInput) -> float:
    """
    Simple composite value metric for waiver world-with/world-without comparisons.

    Hitters:  HR + RBI + SB (all normalized to 0-1 then summed)
    Pitchers: K (normalized) - ERA_penalty - WHIP_penalty
    Two-way:  average of both

    Returns a value in approximately [0, 3].
    """
    pt = (player.player_type or "unknown").lower()

    if pt == "hitter":
        hr  = min((player.proj_hr_p50  or 0.0) / _HR_NORM,  1.0)
        rbi = min((player.proj_rbi_p50 or 0.0) / _RBI_NORM, 1.0)
        sb  = min((player.proj_sb_p50  or 0.0) / 50.0,      1.0)
        return hr + rbi + sb

    if pt == "pitcher":
        k    = min((player.proj_k_p50   or 0.0) / _K_NORM, 1.0)
        era  = (player.proj_era_p50  or 4.50) / 9.0   # 9 ERA -> 1.0 penalty
        whip = (player.proj_whip_p50 or 1.30) / 2.0   # 2.0 WHIP -> 1.0 penalty
        return max(0.0, k - era + (1.0 - whip))

    if pt == "two_way":
        hr  = min((player.proj_hr_p50  or 0.0) / _HR_NORM,  1.0)
        rbi = min((player.proj_rbi_p50 or 0.0) / _RBI_NORM, 1.0)
        sb  = min((player.proj_sb_p50  or 0.0) / 50.0,      1.0)
        k    = min((player.proj_k_p50   or 0.0) / _K_NORM, 1.0)
        era  = (player.proj_era_p50  or 4.50) / 9.0
        whip = (player.proj_whip_p50 or 1.30) / 2.0
        hitter_val  = hr + rbi + sb
        pitcher_val = max(0.0, k - era + (1.0 - whip))
        return (hitter_val + pitcher_val) / 2.0

    # unknown player type -- fall back to score_0_100 normalized to [0, 3]
    return (player.score_0_100 / 100.0) * 3.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_lineup(
    players: list,
    as_of_date: Optional[date] = None,
) -> tuple:
    """
    Greedy lineup optimizer.

    Parameters
    ----------
    players:    list of PlayerDecisionInput -- all available roster candidates
    as_of_date: the date these decisions apply to (defaults to today)

    Returns
    -------
    (LineupDecision, list[DecisionResult])
        LineupDecision holds the structured slot mapping.
        list[DecisionResult] is the flat per-player output ready for DB upsert.
    """
    if as_of_date is None:
        as_of_date = date.today()

    if not players:
        return LineupDecision(), []

    # Rank all players by composite lineup score (descending)
    scored = sorted(players, key=_lineup_score, reverse=True)

    # Slot fill state
    slot_capacity = {slot: count for slot, count in ROSTER_SLOTS.items() if slot != BENCH_SLOT}
    filled_slots: dict = {}      # slot_key -> bdl_player_id  (for multi-slot: "OF_0", "OF_1" etc)
    placed_ids: set = set()
    slot_fill_count: dict = {slot: 0 for slot in slot_capacity}

    # Attempt to fill each prioritized slot (ordered: C, infield, OF, Util, pitchers)
    slot_priority = ["C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P"]

    for player in scored:
        if player.bdl_player_id in placed_ids:
            continue
        for slot in slot_priority:
            capacity = slot_capacity.get(slot, 0)
            if slot_fill_count[slot] >= capacity:
                continue
            if _can_fill_slot(player, slot):
                idx = slot_fill_count[slot]
                key = slot if capacity == 1 else f"{slot}_{idx}"
                filled_slots[key] = player.bdl_player_id
                slot_fill_count[slot] += 1
                placed_ids.add(player.bdl_player_id)
                break

    # Fill bench with remaining players
    bench: list = []
    unrostered: list = []
    bench_cap = ROSTER_SLOTS[BENCH_SLOT]
    for player in scored:
        if player.bdl_player_id in placed_ids:
            continue
        if len(bench) < bench_cap:
            bench.append(player.bdl_player_id)
            placed_ids.add(player.bdl_player_id)
        else:
            unrostered.append(player.bdl_player_id)

    # Compute aggregate lineup score
    id_to_player = {p.bdl_player_id: p for p in players}
    total_score = sum(
        _lineup_score(id_to_player[pid])
        for pid in filled_slots.values()
        if pid in id_to_player
    )

    lineup = LineupDecision(
        selected=filled_slots,
        bench=bench,
        score=total_score,
        unrostered=unrostered,
    )

    # Build flat DecisionResult rows
    results: list = []
    slot_lookup: dict = {}  # bdl_player_id -> slot_key
    for slot_key, pid in filled_slots.items():
        slot_lookup[pid] = slot_key

    for player in players:
        pid = player.bdl_player_id
        if pid not in placed_ids:
            continue
        if pid in bench:
            continue   # bench players get no lineup DecisionResult row
        raw_slot = slot_lookup.get(pid)
        # Normalize multi-slot keys ("OF_0" -> "OF")
        display_slot = raw_slot.split("_")[0] if raw_slot else None
        ls = _lineup_score(player)
        mb = _momentum_bonus(player.momentum_signal)
        reasoning = (
            f"Slot {display_slot}: score={player.score_0_100:.1f}, "
            f"momentum={player.momentum_signal}({mb:+.0f}), z={player.composite_z:.2f}"
        )
        results.append(DecisionResult(
            as_of_date=as_of_date,
            decision_type="lineup",
            bdl_player_id=pid,
            target_slot=display_slot,
            drop_player_id=None,
            lineup_score=round(ls, 4),
            value_gain=None,
            confidence=min(1.0, max(0.0, player.score_0_100 / 100.0)),
            reasoning=reasoning,
        ))

    return lineup, results


def optimize_waivers(
    roster: list,
    waiver_pool: list,
    as_of_date: Optional[date] = None,
) -> tuple:
    """
    Waiver intelligence: world-with vs world-without.

    For each waiver candidate, computes the value gain of adding the candidate
    while dropping the weakest bench player on the current roster.

    Parameters
    ----------
    roster:      list of PlayerDecisionInput -- current roster (active + bench)
    waiver_pool: list of PlayerDecisionInput -- available waiver candidates
    as_of_date:  date for DecisionResult rows (defaults to today)

    Returns
    -------
    (WaiverDecision, list[DecisionResult])
    """
    if as_of_date is None:
        as_of_date = date.today()

    if not waiver_pool:
        return WaiverDecision(), []

    # Identify weakest bench player as potential drop target
    # (bench = all players not slotted as starters)
    roster_sorted = sorted(roster, key=_composite_value)
    drop_candidate = roster_sorted[0] if roster_sorted else None

    if drop_candidate is None:
        return WaiverDecision(), []

    world_without_value = _composite_value(drop_candidate)

    recommendations: list = []
    for candidate in waiver_pool:
        world_with_value = _composite_value(candidate)
        gain = world_with_value - world_without_value

        # Confidence scales with score_0_100 and momentum signal
        mb = _momentum_bonus(candidate.momentum_signal)
        raw_confidence = (candidate.score_0_100 / 100.0) * 0.7 + ((mb + 10.0) / 20.0) * 0.3
        confidence = min(1.0, max(0.0, raw_confidence))

        reasoning = (
            f"Add {candidate.name}: value_gain={gain:+.3f}, "
            f"score={candidate.score_0_100:.1f}, momentum={candidate.momentum_signal}; "
            f"drop {drop_candidate.name}: value={world_without_value:.3f}"
        )
        recommendations.append(WaiverRecommendation(
            add_player_id=candidate.bdl_player_id,
            drop_player_id=drop_candidate.bdl_player_id,
            value_gain=gain,
            confidence=confidence,
            reasoning=reasoning,
        ))

    # Sort by value_gain descending
    recommendations.sort(key=lambda r: r.value_gain, reverse=True)
    waiver_decision = WaiverDecision(recommendations=recommendations)

    # Build DecisionResult rows for each recommendation
    results: list = []
    for rec in recommendations:
        results.append(DecisionResult(
            as_of_date=as_of_date,
            decision_type="waiver",
            bdl_player_id=rec.add_player_id,
            target_slot=None,
            drop_player_id=rec.drop_player_id,
            lineup_score=None,
            value_gain=round(rec.value_gain, 4),
            confidence=round(rec.confidence, 4),
            reasoning=rec.reasoning,
        ))

    return waiver_decision, results
