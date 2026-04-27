"""
MCMC Weekly Matchup Simulator

Monte Carlo simulation of H2H fantasy baseball weekly matchup outcomes.
Uses numpy for fast vectorized sampling — 1000 simulations in <50ms.

Public API:
  simulate_weekly_matchup(my_roster, opponent_roster, ...) -> dict
  simulate_roster_move(my_roster, opponent_roster, add_player, drop_player_name, ...) -> dict

Each player dict must contain:
  cat_scores: dict[str, float]   — z-score per fantasy category (higher = better for all)
  positions:  list[str] | str    — position(s)
  starts_this_week: int          — pitcher starts this week (default 1)
  name: str

Category keys (player_board convention):
  Batting: hr, r, rbi, nsb, avg, ops, tb, h
  Pitching: k_pit, era, whip, w, nsv, qs, k9

Note: All cat_scores are z-scores where HIGHER = BETTER.
ERA and WHIP z-scores are already inverted in the player board.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-player weekly standard deviation in z-score units
# ---------------------------------------------------------------------------
# These represent realistic week-to-week noise around each player's projection.
# Volatile positions and sparse stat categories get higher values.

_PLAYER_WEEKLY_STD: dict[str, float] = {
    # Batting — counting
    "hr": 0.65,
    "r": 0.70,
    "rbi": 0.70,
    "nsb": 0.90,   # stolen bases: volatile
    "h": 0.55,
    "tb": 0.65,
    # Batting — rate
    "avg": 0.40,
    "ops": 0.40,
    # Pitching — counting
    "k_pit": 0.75,
    "w": 0.85,
    "nsv": 1.00,   # saves: binary/volatile
    "qs": 0.80,
    "k9": 0.40,
    # Pitching — rate
    "era": 0.65,
    "whip": 0.55,
}

_DEFAULT_STD = 0.60  # fallback for unknown categories

# Position-based variance multiplier: role players / relievers more volatile
_POSITION_MULT: dict[str, float] = {
    "C": 1.30, "1B": 1.00, "2B": 1.10, "3B": 1.10, "SS": 1.10,
    "OF": 1.00, "LF": 1.00, "CF": 1.00, "RF": 1.00, "DH": 0.90,
    "SP": 1.20, "RP": 1.50, "P": 1.30,
}

# Counting pitcher categories that scale with starts
_STARTS_SCALE_CATS = frozenset({"k_pit", "w", "qs"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _primary_position(player: dict) -> str:
    pos = player.get("positions") or player.get("position") or []
    if isinstance(pos, list):
        return pos[0] if pos else "?"
    return str(pos) if pos else "?"


def _player_std(player: dict, cat: str) -> float:
    base = _PLAYER_WEEKLY_STD.get(cat, _DEFAULT_STD)
    mult = _POSITION_MULT.get(_primary_position(player), 1.0)
    return max(0.05, base * mult)


def _roster_means_stds(roster: list[dict], cats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (means, stds) arrays of shape (n_players, n_cats) for numpy sampling.
    """
    n = len(roster)
    if n == 0:
        return np.zeros((0, len(cats))), np.zeros((0, len(cats)))

    means = np.zeros((n, len(cats)))
    stds = np.zeros((n, len(cats)))

    for i, player in enumerate(roster):
        cat_scores = player.get("cat_scores") or {}
        starts = max(1, int(player.get("starts_this_week", 1)))
        pos = _primary_position(player)
        is_pitcher = pos in ("SP", "RP", "P")

        for j, cat in enumerate(cats):
            base_mean = float(cat_scores.get(cat, 0.0))

            # Two-start pitchers: scale counting cats proportionally
            if is_pitcher and cat in _STARTS_SCALE_CATS and starts >= 2:
                base_mean *= min(starts, 2) * 0.85  # 0.85 avoids double-counting rest effects

            means[i, j] = base_mean
            stds[i, j] = _player_std(player, cat)

    return means, stds


def _detect_categories(rosters: list[list[dict]]) -> list[str]:
    """Auto-detect categories present across all rosters."""
    all_cats: set[str] = set()
    for roster in rosters:
        for p in roster:
            all_cats.update((p.get("cat_scores") or {}).keys())
    # Remove noise keys
    all_cats.discard("l")  # losses — infrequent Yahoo cat, skip
    return sorted(all_cats)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_weekly_matchup(
    my_roster: list[dict],
    opponent_roster: list[dict],
    categories: Optional[list[str]] = None,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Monte Carlo simulation of one week's H2H matchup.

    Parameters
    ----------
    my_roster / opponent_roster:
        Lists of player dicts. Each dict needs cat_scores, positions, starts_this_week.
        Pass an empty list [] for opponent to compare against a league-average opponent
        (all cat z-scores = 0, representing the statistical mean).
    categories:
        Category keys to simulate. Auto-detected from rosters if None.
    n_sims:
        Monte Carlo iterations. 1000 is fast (<50ms) and stable.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        win_prob              float       fraction of sims where my team wins
        category_win_probs    dict        per-category win fraction
        expected_cats_won     float       expected categories won per matchup
        n_sims                int
        elapsed_ms            float
        categories_simulated  list[str]
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if categories is None:
        categories = _detect_categories([my_roster, opponent_roster])

    if not categories:
        # No category data — return 50/50
        return {
            "win_prob": 0.5,
            "category_win_probs": {},
            "expected_cats_won": 0.0,
            "n_sims": 0,
            "elapsed_ms": 0.0,
            "categories_simulated": [],
        }

    my_means, my_stds = _roster_means_stds(my_roster, categories)
    opp_means, opp_stds = _roster_means_stds(opponent_roster, categories)

    # Vectorized sampling: (n_sims, n_players, n_cats)
    n_cats = len(categories)

    if my_means.shape[0] > 0:
        my_noise = rng.normal(0.0, my_stds, size=(n_sims,) + my_means.shape)
        my_totals = (my_means + my_noise).sum(axis=1)   # (n_sims, n_cats)
    else:
        my_totals = np.zeros((n_sims, n_cats))

    if opp_means.shape[0] > 0:
        opp_noise = rng.normal(0.0, opp_stds, size=(n_sims,) + opp_means.shape)
        opp_totals = (opp_means + opp_noise).sum(axis=1)  # (n_sims, n_cats)
    else:
        # Empty opponent = league average (z=0), still has week-level noise
        avg_std = np.full(n_cats, 2.0)  # team-level noise for 12-player average opponent
        opp_totals = rng.normal(0.0, avg_std, size=(n_sims, n_cats))

    # All cats: higher is better (z-scores already inverted for ERA/WHIP)
    cat_wins = (my_totals > opp_totals).astype(float)   # (n_sims, n_cats)

    cat_win_probs = {
        cat: round(float(cat_wins[:, j].mean()), 4)
        for j, cat in enumerate(categories)
    }
    total_cat_wins = cat_wins.sum(axis=1)   # (n_sims,)
    matchup_wins = (total_cat_wins > n_cats / 2.0).astype(float)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "win_prob": round(float(matchup_wins.mean()), 4),
        "category_win_probs": cat_win_probs,
        "expected_cats_won": round(float(total_cat_wins.mean()), 2),
        "n_sims": n_sims,
        "elapsed_ms": round(elapsed_ms, 1),
        "categories_simulated": categories,
    }


def simulate_roster_move(
    my_roster: list[dict],
    opponent_roster: list[dict],
    add_player: dict,
    drop_player_name: str,
    categories: Optional[list[str]] = None,
    n_sims: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate the win-probability impact of a single ADD/DROP roster move.

    The before/after simulations use the same RNG seed offset so results
    are directly comparable (variance is minimized).

    Returns
    -------
    dict with keys:
        win_prob_before         float
        win_prob_after          float
        win_prob_gain           float   positive = move improves win chances
        category_win_probs_before  dict
        category_win_probs_after   dict
        mcmc_enabled            True
        n_sims                  int
        elapsed_ms              float
    """
    t0 = time.perf_counter()
    _seed = seed if seed is not None else 42

    # Auto-detect cats from all players (including add_player)
    if categories is None:
        categories = _detect_categories([my_roster, opponent_roster, [add_player]])

    before = simulate_weekly_matchup(
        my_roster, opponent_roster,
        categories=categories, n_sims=n_sims, seed=_seed,
    )

    # Build modified roster: drop the named player, add the new one
    drop_key = drop_player_name.strip().lower()
    new_roster = [p for p in my_roster if p.get("name", "").strip().lower() != drop_key]
    new_roster.append(add_player)

    after = simulate_weekly_matchup(
        new_roster, opponent_roster,
        categories=categories, n_sims=n_sims, seed=_seed + 1,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "win_prob_before": before["win_prob"],
        "win_prob_after": after["win_prob"],
        "win_prob_gain": round(after["win_prob"] - before["win_prob"], 4),
        "category_win_probs_before": before["category_win_probs"],
        "category_win_probs_after": after["category_win_probs"],
        "expected_cats_won_before": before["expected_cats_won"],
        "expected_cats_won_after": after["expected_cats_won"],
        "mcmc_enabled": True,
        "n_sims": n_sims,
        "elapsed_ms": round(elapsed_ms, 1),
    }


_MCMC_DISABLED: dict = {
    "win_prob_before": 0.5,
    "win_prob_after": 0.5,
    "win_prob_gain": 0.0,
    "category_win_probs_before": {},
    "category_win_probs_after": {},
    "expected_cats_won_before": 0.0,
    "expected_cats_won_after": 0.0,
    "mcmc_enabled": False,
    "n_sims": 0,
    "elapsed_ms": 0.0,
}
