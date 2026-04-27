"""
Tests for backend/data_contracts/ — Layer 0 validation.

Uses real captured fixtures in tests/fixtures/bdl_mlb_*.json.
These fixtures are committed snapshots from 2026-04-05 live capture.

Tests prove:
    1. 100% parse rate against live payloads
    2. Correct nullable/non-nullable field enforcement
    3. Spread/total string-type enforcement with float property access
    4. dob format pass-through (no silent coercion)
    5. Pagination cursor extraction
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.data_contracts import (
    BDLResponse,
    MLBBettingOdd,
    MLBGame,
    MLBInjury,
    MLBPlayer,
    MLBTeam,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# MLBGame — 19 games, all fields non-null in sample
# ---------------------------------------------------------------------------

class TestMLBGame:
    def test_parses_all_games(self):
        raw = load("bdl_mlb_games.json")
        resp = BDLResponse[MLBGame].model_validate(raw)
        assert len(resp.data) == 19
        assert resp.meta.next_cursor is None  # single-page response

    def test_game_fields(self):
        raw = load("bdl_mlb_games.json")
        resp = BDLResponse[MLBGame].model_validate(raw)
        game = resp.data[0]
        assert game.id == 5057892
        assert game.season_type == "regular"
        assert game.status == "STATUS_FINAL"
        assert isinstance(game.home_team, MLBTeam)
        assert game.home_team.abbreviation == "COL"
        assert game.away_team.abbreviation == "PHI"

    def test_team_game_data(self):
        raw = load("bdl_mlb_games.json")
        resp = BDLResponse[MLBGame].model_validate(raw)
        game = resp.data[0]
        assert isinstance(game.home_team_data.inning_scores, list)
        assert game.home_team_data.runs == 1
        assert game.away_team_data.runs == 2

    def test_scoring_summary_is_list(self):
        raw = load("bdl_mlb_games.json")
        resp = BDLResponse[MLBGame].model_validate(raw)
        game = resp.data[0]
        assert isinstance(game.scoring_summary, list)
        assert len(game.scoring_summary) > 0
        play = game.scoring_summary[0]
        assert play.inning in ("top", "bottom")
        assert isinstance(play.away_score, int)

    def test_mlb_team_league_division_enum(self):
        raw = load("bdl_mlb_games.json")
        resp = BDLResponse[MLBGame].model_validate(raw)
        for game in resp.data:
            assert game.home_team.league in ("National", "American")
            assert game.home_team.division in ("East", "Central", "West")

    def test_invalid_season_type_rejected(self):
        raw = load("bdl_mlb_games.json")
        raw["data"][0]["season_type"] = "exhibition"
        with pytest.raises(ValidationError):
            BDLResponse[MLBGame].model_validate(raw)


# ---------------------------------------------------------------------------
# MLBBettingOdd — 6 vendors, spread values are strings
# ---------------------------------------------------------------------------

class TestMLBBettingOdd:
    def test_parses_all_odds(self):
        raw = load("bdl_mlb_odds.json")
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        assert len(resp.data) == 6

    def test_spread_values_are_strings(self):
        raw = load("bdl_mlb_odds.json")
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        for odd in resp.data:
            assert isinstance(odd.spread_home_value, str)
            assert isinstance(odd.spread_away_value, str)
            assert isinstance(odd.total_value, str)

    def test_float_properties(self):
        raw = load("bdl_mlb_odds.json")
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        fanduel = next(o for o in resp.data if o.vendor == "fanduel")
        assert fanduel.spread_home_float == 1.5
        assert fanduel.spread_away_float == -1.5
        assert fanduel.total_float == 3.5

    def test_american_odds_are_integers(self):
        raw = load("bdl_mlb_odds.json")
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        for odd in resp.data:
            assert isinstance(odd.moneyline_home_odds, int)
            assert isinstance(odd.moneyline_away_odds, int)

    def test_non_numeric_string_rejected(self):
        raw = load("bdl_mlb_odds.json")
        raw["data"][0]["spread_home_value"] = "pick"
        with pytest.raises(ValidationError):
            BDLResponse[MLBBettingOdd].model_validate(raw)

    def test_float_spread_value_rejected(self):
        """API contract: spread values must arrive as strings, not floats."""
        raw = load("bdl_mlb_odds.json")
        raw["data"][0]["spread_home_value"] = 1.5  # wrong type
        with pytest.raises(ValidationError):
            BDLResponse[MLBBettingOdd].model_validate(raw)

    def test_none_spread_accepted(self):
        """BDL returns None when a book has not set a spread (moneyline-only row)."""
        raw = load("bdl_mlb_odds.json")
        raw["data"][0]["spread_home_value"] = None
        raw["data"][0]["spread_home_odds"] = None
        raw["data"][0]["spread_away_value"] = None
        raw["data"][0]["spread_away_odds"] = None
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        row = resp.data[0]
        assert row.spread_home_value is None
        assert row.spread_home_odds is None
        assert row.spread_home_float is None
        assert row.has_spread is False
        # Moneyline must still be captured for the row to exist.
        assert isinstance(row.moneyline_home_odds, int)

    def test_none_total_accepted(self):
        raw = load("bdl_mlb_odds.json")
        raw["data"][0]["total_value"] = None
        raw["data"][0]["total_over_odds"] = None
        raw["data"][0]["total_under_odds"] = None
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        row = resp.data[0]
        assert row.total_value is None
        assert row.total_float is None
        assert row.has_total is False

    def test_has_spread_true_for_full_line(self):
        raw = load("bdl_mlb_odds.json")
        resp = BDLResponse[MLBBettingOdd].model_validate(raw)
        assert all(o.has_spread for o in resp.data)
        assert all(o.has_total for o in resp.data)

    def test_missing_moneyline_still_rejected(self):
        """Moneyline is the defining attribute of an odds row — None is invalid."""
        raw = load("bdl_mlb_odds.json")
        raw["data"][0]["moneyline_home_odds"] = None
        with pytest.raises(ValidationError):
            BDLResponse[MLBBettingOdd].model_validate(raw)


# ---------------------------------------------------------------------------
# MLBInjury — 25 items, cursor pagination, nullable detail/side
# ---------------------------------------------------------------------------

class TestMLBInjury:
    def test_parses_all_injuries(self):
        raw = load("bdl_mlb_injuries.json")
        resp = BDLResponse[MLBInjury].model_validate(raw)
        assert len(resp.data) == 25
        assert resp.meta.next_cursor == 409031

    def test_nullable_fields_present(self):
        raw = load("bdl_mlb_injuries.json")
        resp = BDLResponse[MLBInjury].model_validate(raw)
        details = [i.detail for i in resp.data]
        sides = [i.side for i in resp.data]
        assert None in details, "Expected some null detail values"
        assert None in sides, "Expected some null side values"

    def test_player_college_mostly_null(self):
        raw = load("bdl_mlb_injuries.json")
        resp = BDLResponse[MLBInjury].model_validate(raw)
        nulls = sum(1 for i in resp.data if i.player.college is None)
        assert nulls >= 15, f"Expected mostly null college fields, got {nulls} nulls"

    def test_dob_stored_as_string(self):
        raw = load("bdl_mlb_injuries.json")
        resp = BDLResponse[MLBInjury].model_validate(raw)
        for inj in resp.data:
            if inj.player.dob is not None:
                assert isinstance(inj.player.dob, str)

    def test_dob_non_string_rejected(self):
        raw = load("bdl_mlb_injuries.json")
        raw["data"][0]["player"]["dob"] = 19940705  # int instead of string
        with pytest.raises(ValidationError):
            BDLResponse[MLBInjury].model_validate(raw)


# ---------------------------------------------------------------------------
# MLBPlayer — search endpoint (Ohtani)
# ---------------------------------------------------------------------------

class TestMLBPlayer:
    def test_parses_ohtani(self):
        raw = load("bdl_mlb_players.json")
        resp = BDLResponse[MLBPlayer].model_validate(raw)
        assert len(resp.data) == 1
        p = resp.data[0]
        assert p.full_name == "Shohei Ohtani"
        assert p.jersey == "17"
        assert p.team.abbreviation == "LAD"
        assert p.dob == "5/7/1994"
        assert p.college is None
        assert p.draft is None
