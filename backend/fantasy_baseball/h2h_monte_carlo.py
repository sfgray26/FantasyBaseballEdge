"""
H2H One Win Monte Carlo Simulator

Category-by-category win probability simulation for H2H One Win fantasy format.
Returns P(win 6+ categories) instead of projected points.

Performance target: 10,000 sims <200ms via NumPy vectorization.

Usage:
    sim = H2HOneWinSimulator()
    result = sim.simulate_week(my_roster, opponent_roster, n_sims=10000)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import date


@dataclass
class H2HWinResult:
    """Result of H2H One Win Monte Carlo simulation."""

    win_probability: float  # P(win 6+ categories) [0.0, 1.0]

    locked_categories: List[str]  # >85% win probability
    swing_categories: List[str]  # 40-60% win probability (key matchups)
    vulnerable_categories: List[str]  # <30% win probability (risk zones)

    category_win_probs: Dict[str, float]  # Full breakdown: e.g. {"R": 0.72, "HR": 0.45}

    mean_categories_won: float  # Expected categories won (e.g., 5.8 / 10)
    std_categories_won: float  # Std dev (measure of volatility)

    n_simulations: int
    as_of_date: date


@dataclass
class SwingCategory:
    """A category with close win probability (40-60%) — key matchup decision point."""

    category: str
    my_win_prob: float
    opponent_win_prob: float
    recommendation: str  # "STREAM_HITTER", "STREAM_PITCHER", "RIDE_OR_DIE"


class H2HOneWinSimulator:
    """
    Monte Carlo simulation for H2H One Win fantasy format.

    Simulates N iterations of weekly stats for both rosters, comparing
    category-by-category to determine win probability (6+ = win).

    Categories (10 total):
    Hitting: R, HR, RBI, SB, AVG, OPS
    Pitching: W, QS, K, K/9 (or ERA/WHIP depending on league settings)

    Performance: NumPy vectorization targets <200ms for 10,000 sims.
    """

    # H2H One Win categories (default 10-cat format)
    HITTING_CATS = ["R", "HR", "RBI", "SB", "NSB", "AVG", "OPS"]
    PITCHING_CATS = ["W", "QS", "K", "K/9", "ERA", "WHIP"]

    # Standard deviation for stat projection (game-to-game variance)
    # Conservative: CV=0.35 for counting stats, 0.15 for rate stats
    STAT_CV = {
        # Counting stats
        "R": 0.35,
        "HR": 0.40,
        "RBI": 0.35,
        "SB": 0.50,
        "NSB": 0.50,
        "W": 0.30,
        "QS": 0.25,
        "K": 0.25,
        # Rate stats (lower variance)
        "AVG": 0.08,
        "OPS": 0.10,
        "K/9": 0.12,
        "ERA": 0.15,
        "WHIP": 0.12,
    }

    def simulate_week(
        self,
        my_roster: List[Dict[str, Any]],
        opponent_roster: List[Dict[str, Any]],
        n_sims: int = 10000,
        as_of_date: date = None,
    ) -> H2HWinResult:
        """
        Run N Monte Carlo simulations of the weekly matchup.

        Args:
            my_roster: List of player dicts with projected stats
                Example: [{"name": "Ohtani", "R": 15, "HR": 4, ...}, ...]
            opponent_roster: List of player dicts with projected stats
            n_sims: Number of simulations (default: 10000)
            as_of_date: Date for the simulation week

        Returns:
            H2HWinResult with win probability and category breakdown
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Aggregate projected stats for both rosters
        my_proj = self._aggregate_roster(my_roster)
        opp_proj = self._aggregate_roster(opponent_roster)

        # Run Monte Carlo simulation (returns both total wins and per-category matrix)
        categories_won, category_win_matrix = self._run_simulation(my_proj, opp_proj, n_sims)

        # Analyze results
        win_prob = np.mean(categories_won >= 6)  # 6+ categories = win
        locked, swing, vulnerable = self._classify_categories(category_win_matrix)

        return H2HWinResult(
            win_probability=float(win_prob),
            locked_categories=locked,
            swing_categories=swing,
            vulnerable_categories=vulnerable,
            category_win_probs=self._compute_category_probs(category_win_matrix),
            mean_categories_won=float(np.mean(categories_won)),
            std_categories_won=float(np.std(categories_won)),
            n_simulations=n_sims,
            as_of_date=as_of_date,
        )

    def _aggregate_roster(self, roster: List[Dict[str, Any]]) -> Dict[str, float]:
        """Sum projected stats across all players in roster."""
        aggregates = {cat: 0.0 for cat in self.HITTING_CATS + self.PITCHING_CATS}

        for player in roster:
            for cat in aggregates:
                aggregates[cat] += player.get(cat, 0.0)

        return aggregates

    def _run_simulation(
        self,
        my_proj: Dict[str, float],
        opp_proj: Dict[str, float],
        n_sims: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulation using NumPy vectorization.

        For each category:
          1. Sample stats from normal distribution (mean=proj, std=CV*mean)
          2. Compare my vs opponent
          3. Count categories won per simulation

        Returns:
            categories_won: Array of shape (n_sims,) with categories won per sim
            category_win_matrix: Matrix of shape (n_sims, n_categories) with per-category results
        """
        categories = self.HITTING_CATS + self.PITCHING_CATS
        n_categories = len(categories)

        # Pre-allocate results matrix: n_sims x n_categories
        category_win_matrix = np.zeros((n_sims, n_categories))

        for i, cat in enumerate(categories):
            # Get projections
            my_mean = my_proj.get(cat, 0.0)
            opp_mean = opp_proj.get(cat, 0.0)

            # Skip if both zeros (no data)
            if my_mean == 0 and opp_mean == 0:
                category_win_matrix[:, i] = 0.5  # Tie = half win
                continue

            # Compute std from CV
            cv = self.STAT_CV.get(cat, 0.35)
            my_std = max(my_mean * cv, 0.1)  # Minimum std to avoid zero variance
            opp_std = max(opp_mean * cv, 0.1)

            # Sample from normal distribution (vectorized)
            my_samples = np.random.normal(my_mean, my_std, n_sims)
            opp_samples = np.random.normal(opp_mean, opp_std, n_sims)

            # For rate stats, handle differently (lower is better for ERA/WHIP)
            if cat in ["ERA", "WHIP"]:
                # Lower is better
                category_win_matrix[:, i] = (my_samples < opp_samples).astype(float)
            else:
                # Higher is better (counting stats, AVG, OPS, K/9)
                category_win_matrix[:, i] = (my_samples > opp_samples).astype(float)

        # Sum categories won per simulation
        categories_won = np.sum(category_win_matrix, axis=1)

        return categories_won, category_win_matrix

    def _classify_categories(
        self, category_win_matrix: np.ndarray
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Classify categories by win probability.

        Locked: >85% win (safe)
        Swing: 40-60% win (key matchups)
        Vulnerable: <30% win (risk zones)
        """
        locked = []
        swing = []
        vulnerable = []

        # Compute category probabilities from simulation matrix
        category_probs = self._compute_category_probs(category_win_matrix)

        for cat, prob in category_probs.items():
            if prob > 0.85:
                locked.append(cat)
            elif prob < 0.30:
                vulnerable.append(cat)
            elif 0.40 <= prob <= 0.60:
                swing.append(cat)

        return locked, swing, vulnerable

    def _compute_category_probs(
        self, category_win_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute win probability per category from simulation matrix.

        Args:
            category_win_matrix: Matrix of shape (n_sims, n_categories)

        Returns:
            Dict mapping category name to win probability [0.0, 1.0]
        """
        categories = self.HITTING_CATS + self.PITCHING_CATS
        category_probs = {}

        for i, cat in enumerate(categories):
            # Mean win rate for this category across all simulations
            win_rate = np.mean(category_win_matrix[:, i])
            category_probs[cat] = float(win_rate)

        return category_probs
