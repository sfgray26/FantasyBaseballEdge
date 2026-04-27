"""
Tests for matchup engine play-style analysis
Run with: pytest tests/test_matchup_engine.py -v
"""

import pytest
import numpy as np
from backend.services.matchup_engine import (
    MatchupEngine,
    TeamPlayStyle,
    TeamProfileCache,
)
from backend.betting_model import CBBEdgeModel


class TestMatchupAdjustments:
    """Test that matchup-specific adjustments are computed correctly"""

    def test_identical_teams_no_adjustment(self):
        engine = MatchupEngine()
        home = TeamPlayStyle(team="TeamA")
        away = TeamPlayStyle(team="TeamB")

        adj = engine.analyze_matchup(home, away)

        # Identical default profiles should produce near-zero adjustments
        assert abs(adj.margin_adj) < 0.5
        assert adj.sd_adj < 0.5

    def test_pace_mismatch_increases_sd(self):
        engine = MatchupEngine()
        fast = TeamPlayStyle(team="Fast", pace=78.0)
        slow = TeamPlayStyle(team="Slow", pace=60.0)

        adj = engine.analyze_matchup(fast, slow)

        assert adj.sd_adj > 0
        # pace_mismatch is a variance factor — it must NOT appear in adj.factors
        # (which drives margin_adj) to prevent it from being summed into the margin.
        assert 'pace_mismatch_sd' not in adj.factors

    def test_three_point_vs_drop_creates_margin(self):
        engine = MatchupEngine()
        shooter = TeamPlayStyle(team="Shooter", three_par=0.45, three_fg_pct=0.38)
        dropper = TeamPlayStyle(team="Dropper", drop_coverage_pct=0.50)

        # Home = shooter, away = dropper (drop defence)
        adj = engine.analyze_matchup(shooter, dropper)

        assert 'home_3_vs_drop' in adj.factors
        assert adj.factors['home_3_vs_drop'] > 0  # Shooter gets margin boost

    def test_zone_vs_three_penalty(self):
        engine = MatchupEngine()
        zoner = TeamPlayStyle(team="Zone", zone_pct=0.50)
        sniper = TeamPlayStyle(team="Sniper", three_fg_pct=0.40)

        # Home = zone, away = sniper
        adj = engine.analyze_matchup(zoner, sniper)

        assert 'home_zone_vs_away_3' in adj.factors
        assert adj.factors['home_zone_vs_away_3'] < 0  # Zone gets punished

    def test_transition_gap(self):
        engine = MatchupEngine()
        runner = TeamPlayStyle(team="Runner", transition_freq=0.22, transition_ppp=1.15)
        halfcourt = TeamPlayStyle(team="Halfcourt", transition_freq=0.10, transition_ppp=1.00)

        adj = engine.analyze_matchup(runner, halfcourt)

        assert 'transition_gap' in adj.factors
        assert adj.factors['transition_gap'] > 0  # Runner gets advantage


class TestTeamProfileCache:
    """Test the team profile cache"""

    def test_set_and_get(self):
        cache = TeamProfileCache()
        profile = TeamPlayStyle(team="Duke", pace=72.0, three_par=0.38)

        cache.set("Duke", profile)

        assert cache.get("Duke") is not None
        assert cache.get("Duke").pace == 72.0

    def test_missing_team_returns_none(self):
        cache = TeamProfileCache()

        assert cache.get("Nonexistent") is None

    def test_has_profiles(self):
        cache = TeamProfileCache()
        assert not cache.has_profiles()

        cache.set("Duke", TeamPlayStyle(team="Duke"))
        assert cache.has_profiles()

    def test_len(self):
        cache = TeamProfileCache()
        cache.set("Duke", TeamPlayStyle(team="Duke"))
        cache.set("UNC", TeamPlayStyle(team="UNC"))

        assert len(cache) == 2


class TestCategoricalDampening:
    """Test that category-based split-sign RSS dampening prevents factor stuffing."""

    def test_possession_factors_dampened_via_rss(self):
        """Multiple possession-generating factors should be compressed by RSS."""
        engine = MatchupEngine()
        # transition_gap + turnover_battle + rebounding are all "possession_generating"
        factors = {"transition_gap": 2.0, "turnover_battle": 2.0, "rebounding": 2.0}
        result = engine._apply_diminishing_returns(factors)
        raw_sum = 6.0  # raw linear sum
        # RSS of [2,2,2] = sqrt(12) ≈ 3.46 → capped at 3.0
        # Then tanh further compresses
        assert result < raw_sum * 0.8  # Significant compression
        assert result > 0  # Still positive

    def test_single_factor_passes_through(self):
        """A single factor in a category should equal the raw value (before tanh)."""
        engine = MatchupEngine()
        # Small single factor: tanh(0.5/4) ≈ 0.5/4 → approximately linear
        factors = {"transition_gap": 0.5}
        result = engine._apply_diminishing_returns(factors)
        assert abs(result - 0.5) < 0.05

    def test_cross_category_factors_stack(self):
        """Factors in different categories should stack before global tanh."""
        engine = MatchupEngine()
        # One possession factor + one shot quality factor
        factors = {"transition_gap": 1.5, "home_3_vs_drop": 1.5}
        result = engine._apply_diminishing_returns(factors)
        # Both in different categories → sum = 3.0, then tanh(3/4)*4 ≈ 2.85
        assert result > 2.0  # Stacks meaningfully
        assert result < 3.0  # tanh compresses

    def test_category_cap_prevents_stuffing(self):
        """Within a single category, raw sum is capped at CATEGORY_CAP."""
        engine = MatchupEngine()
        # All possession-generating, very large values
        factors = {"transition_gap": 4.0, "turnover_battle": 4.0, "rebounding": 4.0}
        result = engine._apply_diminishing_returns(factors)
        # RSS of [4,4,4] = sqrt(48) ≈ 6.93 → capped at 3.0 → tanh(3/4)*4 ≈ 2.55
        assert result < engine.CATEGORY_CAP + 0.1  # Can't exceed category cap
        assert result > 0

    def test_mixed_sign_no_amplification(self):
        """Mixed-sign factors in same category must NOT amplify beyond net sum.

        Before the fix: [+2.0, -1.0] → sign(1)*sqrt(5) = 2.24 (WRONG)
        After the fix:  [+2.0, -1.0] → min(2.0,CAP) - min(1.0,CAP) = 1.0
        """
        engine = MatchupEngine()
        factors = {"transition_gap": 2.0, "rebounding": -1.0}
        result = engine._apply_diminishing_returns(factors)
        raw_net = 1.0  # 2.0 - 1.0
        # With split-sign RSS: rss_pos=2.0, rss_neg=1.0 → net=1.0
        # Then tanh(1.0/4)*4 ≈ 0.97
        assert result < raw_net + 0.1  # Must NOT exceed raw net
        assert result > 0.8  # tanh(1/4)*4 ≈ 0.97

    def test_all_negative_same_category(self):
        """All-negative factors in one category should produce a negative result."""
        engine = MatchupEngine()
        factors = {"transition_gap": -2.0, "turnover_battle": -1.5}
        result = engine._apply_diminishing_returns(factors)
        # rss_neg = sqrt(4 + 2.25) = 2.5, rss_pos = 0 → net = -2.5
        assert result < 0
        # tanh(-2.5/4)*4 ≈ -2.30
        assert result > -2.6

    def test_pace_mismatch_not_in_margin_factors(self):
        engine = MatchupEngine()
        fast = TeamPlayStyle(team="Fast", pace=80.0)
        slow = TeamPlayStyle(team="Slow", pace=58.0)

        adj = engine.analyze_matchup(fast, slow)

        assert "pace_mismatch_sd" not in adj.factors
        assert adj.sd_adj > 0  # SD still gets the adjustment

    def test_factor_categories_cover_known_factors(self):
        """All production factor names should map to a category."""
        engine = MatchupEngine()
        known = [
            "rebounding", "turnover_battle", "transition_gap",
            "home_3_vs_drop", "away_3_vs_drop",
            "home_zone_vs_away_3", "away_zone_vs_home_3",
        ]
        for name in known:
            assert name in engine.FACTOR_CATEGORIES, f"{name} not in FACTOR_CATEGORIES"


class TestDiminishingReturns:
    """Test that RSS + tanh caps total margin adjustment."""

    def test_small_adjustment_is_approximately_linear(self):
        """For small inputs, tanh(x) ≈ x, so small adjustments pass through."""
        engine = MatchupEngine()
        # Single uncategorised factor — passes through RSS unchanged
        factors = {"uncategorised_factor": 0.5}
        result = engine._apply_diminishing_returns(factors)
        assert abs(result - 0.5) < 0.05

    def test_large_adjustment_is_saturated(self):
        """Large raw adjustments should be compressed below MAX_TOTAL_ADJ."""
        engine = MatchupEngine()
        # Uncategorised factors stack linearly, then tanh compresses
        factors = {"a": 3.0, "b": 3.0, "c": 2.0, "d": 2.0}
        result = engine._apply_diminishing_returns(factors)
        assert result < engine.MAX_TOTAL_ADJ
        assert result > 3.5  # Still significant

    def test_total_never_exceeds_cap(self):
        """No matter how many factors, the adjustment must stay below MAX_TOTAL_ADJ."""
        engine = MatchupEngine()
        factors = {f"unc_{i}": 2.0 for i in range(10)}  # 20 pts raw
        result = engine._apply_diminishing_returns(factors)
        assert abs(result) <= engine.MAX_TOTAL_ADJ + 0.001

    def test_negative_adjustments_also_saturate(self):
        """Negative (away-favoring) adjustments should saturate symmetrically."""
        engine = MatchupEngine()
        factors = {"a": -3.0, "b": -3.0, "c": -2.0}
        result = engine._apply_diminishing_returns(factors)
        assert result > -engine.MAX_TOTAL_ADJ
        assert result < -2.5  # Compressed but still significant

    def test_empty_factors_returns_zero(self):
        engine = MatchupEngine()
        assert engine._apply_diminishing_returns({}) == 0.0

    def test_tanh_preserves_sign(self):
        """Positive inputs stay positive, negative stay negative."""
        engine = MatchupEngine()
        assert engine._apply_diminishing_returns({"a": 1.0}) > 0
        assert engine._apply_diminishing_returns({"a": -1.0}) < 0


class TestDynamicVolatility:
    """Test that market_volatility and hours_to_tipoff affect adjusted_sd."""

    def test_market_volatility_increases_sd(self):
        """Positive market_volatility should increase adjusted SD."""
        model = CBBEdgeModel(base_sd=11.0)

        sd_no_vol = model.adjusted_sd({})
        sd_with_vol = model.adjusted_sd({}, market_volatility=1.5)

        assert sd_with_vol > sd_no_vol

    def test_zero_market_volatility_no_change(self):
        """market_volatility=0 should not change SD."""
        model = CBBEdgeModel(base_sd=11.0)

        sd_none = model.adjusted_sd({})
        sd_zero = model.adjusted_sd({}, market_volatility=0.0)

        assert sd_none == sd_zero

    def test_volatility_saturates_via_tanh(self):
        """Extreme market_volatility should not produce runaway SD."""
        model = CBBEdgeModel(base_sd=11.0)

        sd_moderate = model.adjusted_sd({}, market_volatility=1.0)
        sd_extreme = model.adjusted_sd({}, market_volatility=100.0)

        # tanh(100) ≈ 1.0, so extreme should be barely above moderate
        # max multiplier is 1.15 regardless of input
        assert sd_extreme < 11.0 * 1.16
        assert sd_extreme > sd_moderate

    def test_hours_to_tipoff_decays_injury_penalties(self):
        """Injury penalties should shrink as tipoff approaches."""
        model = CBBEdgeModel(base_sd=11.0)

        penalties = {"star_injury": 3.0, "stale_lines": 0.5}

        sd_24h = model.adjusted_sd(dict(penalties), hours_to_tipoff=24.0)
        sd_1h = model.adjusted_sd(dict(penalties), hours_to_tipoff=1.0)
        sd_near = model.adjusted_sd(dict(penalties), hours_to_tipoff=0.1)

        # Far from tipoff → full injury penalty → higher SD
        assert sd_24h > sd_1h
        assert sd_1h > sd_near

    def test_time_decay_does_not_affect_non_injury_penalties(self):
        """Only penalties with 'injury' in the key should be decayed."""
        model = CBBEdgeModel(base_sd=11.0)

        penalties_only_stale = {"stale_lines": 2.0}
        sd_no_time = model.adjusted_sd(dict(penalties_only_stale))
        sd_with_time = model.adjusted_sd(dict(penalties_only_stale), hours_to_tipoff=1.0)

        # No injury penalties → time decay has no effect
        assert abs(sd_no_time - sd_with_time) < 0.001

    def test_volatility_and_time_in_analyze_game(self):
        """market_volatility and hours_to_tipoff should flow through analyze_game."""
        model = CBBEdgeModel(seed=42)

        game = {'home_team': 'Duke', 'away_team': 'UNC', 'is_neutral': True}
        ratings = {
            'kenpom': {'home': 25.0, 'away': 20.0},
            'barttorvik': {'home': 25.0, 'away': 20.0},
            'evanmiya': {'home': 25.0, 'away': 20.0},
        }
        odds = {'spread': -5.0, 'spread_odds': -110, 'spread_away_odds': -110}

        result_base = model.analyze_game(game, odds, ratings)
        model2 = CBBEdgeModel(seed=42)
        result_vol = model2.analyze_game(
            game, odds, ratings,
            market_volatility=2.0, hours_to_tipoff=6.0,
        )

        # With market volatility, SD should be higher
        assert result_vol.adjusted_sd > result_base.adjusted_sd
        # New fields recorded in full_analysis
        calcs = result_vol.full_analysis['calculations']
        assert calcs['market_volatility'] == 2.0
        assert calcs['hours_to_tipoff'] == 6.0


class TestSimultaneousKelly:
    """Test the simultaneous Kelly covariance penalty from analysis.py."""

    def _make_bet(self, idx, conf=None, spread=-5.0, side="home", units=1.0, edge=0.05):
        return {
            "index": idx,
            "conference": conf,
            "spread": spread,
            "bet_side": side,
            "recommended_units": units,
            "kelly_fractional": 0.05,
            "edge_conservative": edge,
        }

    def test_single_bet_unchanged(self):
        """A single bet should not be penalised."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [self._make_bet(0, conf="ACC")]
        result = _apply_simultaneous_kelly(bets)
        assert result[0]["recommended_units"] == 1.0

    def test_same_conference_penalty(self):
        """Multiple bets in the same conference should be penalised."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, conf="ACC", units=1.0),
            self._make_bet(1, conf="ACC", units=1.0),
            self._make_bet(2, conf="ACC", units=1.0),
        ]
        result = _apply_simultaneous_kelly(bets)
        # First bet unchanged, second/third penalised
        assert result[0]["recommended_units"] == 1.0
        assert result[1]["recommended_units"] < 1.0
        assert result[2]["recommended_units"] < result[1]["recommended_units"]

    def test_different_conferences_no_penalty(self):
        """Bets in different conferences should not be penalised for correlation."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, conf="ACC", units=1.0),
            self._make_bet(1, conf="Big Ten", units=1.0),
            self._make_bet(2, conf="SEC", units=1.0),
        ]
        result = _apply_simultaneous_kelly(bets, max_total_exposure_pct=15.0)
        # No conference penalty applied — total is 3.0 which is under 15%
        assert all(b["recommended_units"] == 1.0 for b in result)

    def test_exposure_cap_greedy_allocation(self):
        """If total exposure exceeds cap, greedy allocation should prioritize highest edge."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, units=3.0, edge=0.02),  # lowest edge
            self._make_bet(1, units=3.0, edge=0.08),  # highest edge
            self._make_bet(2, units=3.0, edge=0.05),  # middle edge
        ]
        result = _apply_simultaneous_kelly(bets, max_total_exposure_pct=5.0)
        total = sum(b["recommended_units"] for b in result)
        assert total <= 5.01  # Within cap

        # Greedy: highest-edge bet (idx=1, edge=0.08) should be fully allocated
        bet_by_idx = {b["index"]: b for b in result}
        assert bet_by_idx[1]["recommended_units"] == 3.0  # Full allocation
        # Lowest-edge bet (idx=0, edge=0.02) should be dropped or reduced
        assert bet_by_idx[0]["recommended_units"] < 3.0

    def test_fade_favourite_clustering(self):
        """Multiple bets fading heavy favourites should be penalised."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            # Fading heavy favourite: away side with spread < -5
            self._make_bet(0, spread=-8.0, side="away", units=1.0),
            self._make_bet(1, spread=-10.0, side="away", units=1.0),
            self._make_bet(2, spread=-7.0, side="away", units=1.0),
        ]
        result = _apply_simultaneous_kelly(bets, max_total_exposure_pct=15.0)
        # First fade unchanged, subsequent should be penalised
        assert result[0]["recommended_units"] == 1.0
        assert result[1]["recommended_units"] < 1.0

    def test_greedy_drops_marginal_bets(self):
        """Bets that don't fit in the cap should be zeroed out, not proportionally scaled."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, units=4.0, edge=0.10),  # Fully fills cap
            self._make_bet(1, units=4.0, edge=0.05),  # Should be dropped
            self._make_bet(2, units=4.0, edge=0.02),  # Should be dropped
        ]
        result = _apply_simultaneous_kelly(bets, max_total_exposure_pct=4.0)

        bet_by_idx = {b["index"]: b for b in result}
        # Top-edge bet fills the entire cap
        assert bet_by_idx[0]["recommended_units"] == 4.0
        # Remaining bets should be zeroed
        assert bet_by_idx[1]["recommended_units"] == 0.0
        assert bet_by_idx[2]["recommended_units"] == 0.0
        # Dropped bets should have a reason
        assert "greedy_drop" in bet_by_idx[1]["slate_adjustment_reason"]

    def test_greedy_partial_fill(self):
        """When a bet partially fits, it should get the remaining capacity."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, units=3.0, edge=0.10),  # Takes 3.0 of 5.0 cap
            self._make_bet(1, units=3.0, edge=0.05),  # Should get remaining 2.0
            self._make_bet(2, units=3.0, edge=0.02),  # Should be dropped
        ]
        result = _apply_simultaneous_kelly(bets, max_total_exposure_pct=5.0)

        bet_by_idx = {b["index"]: b for b in result}
        assert bet_by_idx[0]["recommended_units"] == 3.0
        assert abs(bet_by_idx[1]["recommended_units"] - 2.0) < 0.01
        assert bet_by_idx[2]["recommended_units"] == 0.0

    def test_empty_list_returns_empty(self):
        from backend.services.analysis import _apply_simultaneous_kelly
        assert _apply_simultaneous_kelly([]) == []

    def test_adjustment_reason_populated(self):
        """Adjusted bets should have a reason string."""
        from backend.services.analysis import _apply_simultaneous_kelly
        bets = [
            self._make_bet(0, conf="ACC", units=1.0),
            self._make_bet(1, conf="ACC", units=1.0),
        ]
        result = _apply_simultaneous_kelly(bets)
        # Second bet should have a conference correlation reason
        assert "conf_corr" in result[1]["slate_adjustment_reason"]


class TestEfgPressureGap:
    """Test _efg_pressure_gap matchup factor (Task 4.3)."""

    def test_home_offense_exploits_weak_away_defense(self):
        """Home has elite offense vs poor away defense → positive margin adj."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        # home_off_edge = 0.560 - 0.510 = 0.050
        # away_off_edge = 0.500 - 0.505 = -0.005
        # net = 0.055 → 0.055 * 15 = 0.825 > threshold
        home = TeamPlayStyle(team="Home", efg_pct=0.560, def_efg_pct=0.505)
        away = TeamPlayStyle(team="Away", efg_pct=0.500, def_efg_pct=0.510)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert adj.factors.get("efg_pressure", 0) > 0

    def test_away_offense_exploits_weak_home_defense(self):
        """Away dominant offense vs poor home defense → negative margin adj."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        home = TeamPlayStyle(team="Home", efg_pct=0.480, def_efg_pct=0.530)
        away = TeamPlayStyle(team="Away", efg_pct=0.560, def_efg_pct=0.505)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert adj.factors.get("efg_pressure", 0) < 0

    def test_below_threshold_no_adjustment(self):
        """Net eFG edge < 2pp → no factor added."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        home = TeamPlayStyle(team="Home", efg_pct=0.505, def_efg_pct=0.505)
        away = TeamPlayStyle(team="Away", efg_pct=0.505, def_efg_pct=0.505)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert "efg_pressure" not in adj.factors

    def test_none_efg_skips_factor(self):
        """When efg_pct is None (BartTorvik miss), factor is skipped."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        home = TeamPlayStyle(team="Home", efg_pct=None)
        away = TeamPlayStyle(team="Away", efg_pct=0.500)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert "efg_pressure" not in adj.factors

    def test_none_away_efg_skips_factor(self):
        """When away efg_pct is None, factor is also skipped."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        home = TeamPlayStyle(team="Home", efg_pct=0.520)
        away = TeamPlayStyle(team="Away", efg_pct=None)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert "efg_pressure" not in adj.factors

    def test_note_appended_when_factor_fires(self):
        """When factor fires, a descriptive note is added to adj.notes."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        home = TeamPlayStyle(team="Home", efg_pct=0.570, def_efg_pct=0.505)
        away = TeamPlayStyle(team="Away", efg_pct=0.490, def_efg_pct=0.505)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        assert any("eFG pressure" in note for note in adj.notes)

    def test_scale_is_correct(self):
        """Factor value equals round(net * 15, 2)."""
        engine = MatchupEngine()
        from backend.services.matchup_engine import MatchupAdjustment
        # home_off_edge = 0.550 - 0.500 = 0.050
        # away_off_edge = 0.500 - 0.505 = -0.005
        # net = 0.055 → 0.055 * 15 = 0.825
        home = TeamPlayStyle(team="Home", efg_pct=0.550, def_efg_pct=0.505)
        away = TeamPlayStyle(team="Away", efg_pct=0.500, def_efg_pct=0.500)
        adj = MatchupAdjustment()
        engine._efg_pressure_gap(home, away, adj)
        expected = round((0.550 - 0.500 - (0.500 - 0.505)) * 15.0, 2)
        assert adj.factors["efg_pressure"] == pytest.approx(expected)


class TestTurnoverPressureGap:
    """Test _turnover_pressure_gap matchup factor (P1)."""

    def test_home_pressure_advantage_positive(self):
        """Home D forces more TOs than away D → positive to_pressure."""
        from backend.services.matchup_engine import MatchupAdjustment
        engine = MatchupEngine()
        # home forces: 0.210 - 0.175 = +0.035; away forces: 0.175 - 0.175 = 0; net=0.035
        home = TeamPlayStyle(team="Home", def_to_pct=0.210, to_pct=0.175)
        away = TeamPlayStyle(team="Away", def_to_pct=0.175, to_pct=0.175)
        adj = MatchupAdjustment()
        engine._turnover_pressure_gap(home, away, adj)
        assert adj.factors.get("to_pressure", 0) > 0

    def test_away_pressure_advantage_negative(self):
        """Away D forces more TOs than home D → negative to_pressure."""
        from backend.services.matchup_engine import MatchupAdjustment
        engine = MatchupEngine()
        home = TeamPlayStyle(team="Home", def_to_pct=0.175, to_pct=0.175)
        away = TeamPlayStyle(team="Away", def_to_pct=0.210, to_pct=0.175)
        adj = MatchupAdjustment()
        engine._turnover_pressure_gap(home, away, adj)
        assert adj.factors.get("to_pressure", 0) < 0

    def test_below_threshold_no_factor(self):
        """Net < 1.5pp → no factor added."""
        from backend.services.matchup_engine import MatchupAdjustment
        engine = MatchupEngine()
        # home_def=0.005, away_def=-0.005 → net=0.010 < 0.015 threshold
        home = TeamPlayStyle(team="Home", def_to_pct=0.180, to_pct=0.175)
        away = TeamPlayStyle(team="Away", def_to_pct=0.175, to_pct=0.175)
        adj = MatchupAdjustment()
        engine._turnover_pressure_gap(home, away, adj)
        assert "to_pressure" not in adj.factors

    def test_scale_is_correct(self):
        """Factor value equals round(net * 23.0, 2)."""
        from backend.services.matchup_engine import MatchupAdjustment
        engine = MatchupEngine()
        # home_def = 0.200 - 0.175 = 0.025; away_def = 0.175 - 0.175 = 0; net = 0.025
        home = TeamPlayStyle(team="Home", def_to_pct=0.200, to_pct=0.175)
        away = TeamPlayStyle(team="Away", def_to_pct=0.175, to_pct=0.175)
        adj = MatchupAdjustment()
        engine._turnover_pressure_gap(home, away, adj)
        expected = round(0.025 * 23.0, 2)
        assert adj.factors["to_pressure"] == pytest.approx(expected)

    def test_note_appended_when_fires(self):
        """A descriptive note is appended to adj.notes when factor fires."""
        from backend.services.matchup_engine import MatchupAdjustment
        engine = MatchupEngine()
        home = TeamPlayStyle(team="Home", def_to_pct=0.220, to_pct=0.175)
        away = TeamPlayStyle(team="Away", def_to_pct=0.175, to_pct=0.175)
        adj = MatchupAdjustment()
        engine._turnover_pressure_gap(home, away, adj)
        assert any("TO pressure" in note for note in adj.notes)


class TestTeamMapping:
    """Test team name normalization, especially P2 manual override additions."""

    def test_georgia_st_manual_override(self):
        """'Georgia St Panthers' maps to 'Georgia St.' via _MANUAL_OVERRIDES."""
        from backend.services.team_mapping import normalize_team_name
        result = normalize_team_name(
            "Georgia St Panthers", ["Georgia St.", "Georgia", "Georgia Tech"]
        )
        assert result == "Georgia St."

    def test_georgia_bulldogs_not_confused_with_georgia_st(self):
        """'Georgia Bulldogs' → 'Georgia', never 'Georgia St.' (collision guard)."""
        from backend.services.team_mapping import normalize_team_name
        result = normalize_team_name(
            "Georgia Bulldogs", ["Georgia", "Georgia St.", "Georgia Tech"]
        )
        assert result != "Georgia St."
        assert result == "Georgia"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
