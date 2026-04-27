"""
Yahoo Fantasy Sports API client — OAuth 2.0 with auto-refresh and resilience patterns.

Unified client combining base OAuth logic with circuit breaker, fallback, and cache.

Authentication flow (one-time setup):
    python -m backend.fantasy_baseball.yahoo_client_resilient --auth

This opens a browser, you authorize, paste the redirect URL back,
and the refresh token is saved to .env automatically.

Subsequent calls use the stored refresh token to obtain fresh access tokens.

Yahoo Fantasy API base: https://fantasysports.yahooapis.com/fantasy/v2/
League key format:      mlb.l.{YAHOO_LEAGUE_ID}

Resilience features (ResilientYahooClient):
- Circuit breaker for cascading failures
- Metadata fallback when percent_owned is unavailable
- Position normalization for lineup mismatches
- Stale cache for graceful degradation
"""

import csv
import json
import logging
import os
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from dotenv import load_dotenv, set_key

from backend.core.circuit_breaker import CircuitBreaker as _CoreCircuitBreaker
from backend.fantasy_baseball.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.fantasy_baseball.cache_manager import StaleCacheManager, CacheResult, NoDataAvailableError
from backend.fantasy_baseball.position_normalizer import (
    PositionNormalizer,
    YahooRoster,
    RosterSlot,
    Player,
    ValidationResult,
    LineupValidationError,
)
from backend.fantasy_baseball.lineup_validator import (
    LineupValidator,
    OptimizedSlot,
    format_lineup_report,
)

load_dotenv()
logger = logging.getLogger(__name__)

_token_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YAHOO_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
YAHOO_API_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
YAHOO_SPORT = "469"  # 2026 MLB season game ID

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class YahooAuthError(Exception):
    pass


class YahooAPIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Base client — OAuth + HTTP + all Yahoo API methods
# ---------------------------------------------------------------------------

class YahooFantasyClient:
    """
    Thin wrapper around Yahoo Fantasy Sports API v2.

    Usage:
        client = YahooFantasyClient()
        league = client.get_league()
        roster = client.get_my_roster()
    """

    def __init__(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("API CLIENT INIT: YahooFantasyClient - Initializing...")

        self.client_id = os.getenv("YAHOO_CLIENT_ID", "")
        self.client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
        self.league_id = os.getenv("YAHOO_LEAGUE_ID", "72586")
        self.league_key = f"{YAHOO_SPORT}.l.{self.league_id}"
        self._refresh_token = os.getenv("YAHOO_REFRESH_TOKEN", "")
        self._access_token = os.getenv("YAHOO_ACCESS_TOKEN", "")
        self._token_expiry: float = 0.0
        self._session = requests.Session()
        self._cb = _CoreCircuitBreaker(failure_threshold=3, recovery_timeout=60, window_seconds=300)

        # Log credential status (masked)
        client_id_status = f"{self.client_id[:10]}..." if len(self.client_id) > 10 else "NOT_SET"
        client_secret_status = f"{self.client_secret[:5]}..." if len(self.client_secret) > 5 else "NOT_SET"
        refresh_token_status = f"{self._refresh_token[:10]}..." if len(self._refresh_token) > 10 else "NOT_SET"

        logger.info("API CLIENT INIT: YahooFantasyClient - client_id=%s, client_secret=%s, refresh_token=%s",
                   client_id_status, client_secret_status, refresh_token_status)
        logger.info("API CLIENT INIT: YahooFantasyClient - league_id=%s, league_key=%s",
                   self.league_id, self.league_key)

        if not self.client_id or not self.client_secret:
            logger.error("API CLIENT INIT FAILED: YahooFantasyClient - Missing credentials (client_id=%s, client_secret=%s)",
                        client_id_status, client_secret_status)
            raise YahooAuthError(
                "YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET must be set in .env"
            )

        logger.info("API CLIENT INIT SUCCESS: YahooFantasyClient - Initialization complete")

    # ------------------------------------------------------------------
    # OAuth 2.0 — Authorization Code Flow
    # ------------------------------------------------------------------

    def get_authorization_url(self) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": "oob",
            "response_type": "code",
            "language": "en-us",
        }
        return f"{YAHOO_AUTH_URL}?{urlencode(params)}"

    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """Exchange authorization code for access + refresh tokens."""
        response = requests.post(
            YAHOO_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": auth_code.strip(),
                "redirect_uri": "oob",
            },
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise YahooAuthError(
                f"Token exchange failed: {response.status_code} — {response.text}"
            )
        tokens = response.json()
        self._store_tokens(tokens)
        return tokens

    def _refresh_access_token(self) -> None:
        """Use refresh token to get a new access token."""
        if not self._refresh_token:
            raise YahooAuthError(
                "No refresh token stored. Run: python -m backend.fantasy_baseball.yahoo_client_resilient --auth"
            )
        response = requests.post(
            YAHOO_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise YahooAuthError(
                f"Token refresh failed: {response.status_code} — {response.text}"
            )
        tokens = response.json()
        self._store_tokens(tokens)

    def _store_tokens(self, tokens: dict) -> None:
        """Persist tokens to .env and update in-memory state.

        On Railway (no writable .env), the write fails silently —
        tokens are still live in-memory for the process lifetime.
        Set YAHOO_REFRESH_TOKEN in Railway env vars directly after
        completing the one-time auth flow locally.
        """
        self._access_token = tokens["access_token"]
        self._refresh_token = tokens.get("refresh_token", self._refresh_token)
        self._token_expiry = time.time() + tokens.get("expires_in", 3600) - 60
        # Write back to .env — best-effort; fails silently on Railway
        try:
            set_key(str(ENV_PATH), "YAHOO_ACCESS_TOKEN", self._access_token)
            set_key(str(ENV_PATH), "YAHOO_REFRESH_TOKEN", self._refresh_token)
            logger.info("Yahoo tokens refreshed and persisted to .env")
        except Exception as exc:
            logger.info("Yahoo tokens refreshed (in-memory only — .env not writable: %s)", exc)

    def _ensure_token(self) -> None:
        """Thread-safe token refresh with double-check locking."""
        # Fast path — no lock needed
        if time.time() < self._token_expiry and self._access_token:
            return
        # Slow path — acquire lock, then re-check before refreshing
        with _token_lock:
            if time.time() < self._token_expiry and self._access_token:
                return
            self._refresh_access_token()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET from Yahoo API with circuit breaker and timeout."""
        if not self._cb.should_allow_request():
            raise YahooAPIError("Yahoo API circuit breaker is OPEN — service temporarily unavailable", 503)

        self._ensure_token()
        url = f"{YAHOO_API_BASE}/{path.lstrip('/')}"
        default_params = {"format": "json"}
        if params:
            default_params.update(params)

        # Yahoo rejects URL-encoded commas in the `out` param (e.g. %2C).
        # Extract it and append as a raw query-string fragment instead.
        out_value = default_params.pop("out", None)
        if out_value:
            url = f"{url}?out={out_value}"

        for attempt in range(3):
            try:
                resp = self._session.get(
                    url,
                    params=default_params,
                    headers={"Authorization": f"Bearer {self._access_token}"},
                    timeout=10,
                )
                if resp.status_code == 401:
                    # Token may have just expired mid-request
                    self._refresh_access_token()
                    continue
                if resp.status_code == 999:
                    wait = 2 ** attempt
                    logger.warning(f"Yahoo rate limit hit, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    raise YahooAPIError(
                        f"Yahoo API error {resp.status_code}: {resp.text[:300]}",
                        resp.status_code,
                    )
                self._cb.record_success()
                return resp.json()
            except YahooAPIError:
                raise
            except Exception:
                if attempt == 2:
                    self._cb.record_failure()
                raise

        raise YahooAPIError("Yahoo API failed after 3 attempts")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_league_section(raw) -> dict:
        """
        Yahoo returns league[N] as either:
          (a) a merged dict  {"name": "...", "teams": {...}, ...}
          (b) a list of single-key dicts  [{"name": "..."}, {"teams": {...}}, ...]

        This helper normalises both shapes to a plain dict.
        """
        if isinstance(raw, list):
            out = {}
            for item in raw:
                if isinstance(item, dict):
                    out.update(item)
            return out
        return raw if isinstance(raw, dict) else {}

    def _league_section(self, data: dict, index: int) -> dict:
        """Extract and flatten league[index] from a fantasy_content response."""
        return self._flatten_league_section(
            data["fantasy_content"]["league"][index]
        )

    def _safe_get(self, obj: dict, key: str) -> dict:
        """Get key from a dict, auto-flattening the value if Yahoo returned it as a list."""
        val = obj.get(key, {}) if isinstance(obj, dict) else {}
        if isinstance(val, list):
            return self._flatten_league_section(val)
        return val if isinstance(val, dict) else {}

    def _team_section(self, data: dict) -> dict:
        """
        Flatten the entire team array from a fantasy_content response.

        Yahoo returns team as either:
          (a) [[meta_dict, ...], {"roster": {...}}]   — 2-element list
          (b) [{"team_key": ...}, {"name": ...}, ..., {"roster": {...}}]  — flat list

        We flatten ALL dict items across the outer list, skipping nested lists,
        so the result always contains keys like "roster", "name", etc.
        """
        return self._flatten_league_section(
            data["fantasy_content"]["team"]
        )

    # ------------------------------------------------------------------
    # League endpoints
    # ------------------------------------------------------------------

    def get_league(self) -> dict:
        """League metadata: name, scoring type, settings."""
        data = self._get(f"league/{self.league_key}")
        return self._league_section(data, 0)

    def get_league_settings(self) -> dict:
        data = self._get(f"league/{self.league_key}/settings")
        return self._league_section(data, 0)

    @staticmethod
    def _iter_block(block, item_key: str):
        """Yield item_key values from either an indexed dict or a list block.

        Yahoo 2025 format: {"count": N, "0": {item_key: ...}, "1": {item_key: ...}}
        Yahoo 2026 format: [{item_key: ...}, {item_key: ...}]
        Yahoo 2026 roster format: {"0": {item_key: [...]}, "1": {item_key: [...]}} (no "count" key)
        """
        if isinstance(block, list):
            for item in block:
                if isinstance(item, dict) and item_key in item:
                    yield item[item_key]
        elif isinstance(block, dict):
            count = int(block.get("count", 0))

            # If we have a count, use it
            if count > 0:
                for i in range(count):
                    entry = block.get(str(i), {})
                    if isinstance(entry, dict) and item_key in entry:
                        yield entry[item_key]
            else:
                # New Yahoo API format: numeric keys without "count"
                # Iterate through all numeric keys in sorted order
                numeric_keys = sorted([k for k in block.keys() if k.isdigit()])
                for key in numeric_keys:
                    entry = block.get(key, {})
                    if isinstance(entry, dict):
                        # For numeric key entries, the item_key value might be directly the data
                        # or nested under item_key
                        if item_key in entry:
                            yield entry[item_key]
                        elif len(entry) == 1 and item_key in str(entry):
                            # Entry might be like {"team": [...]} directly
                            if isinstance(list(entry.values())[0], list):
                                yield list(entry.values())[0]

    def get_standings(self) -> list[dict]:
        data = self._get(f"league/{self.league_key}/standings")
        sec = self._league_section(data, 1)
        teams_raw = sec.get("standings", [{}])[0].get("teams", {})
        teams = []
        count = int(teams_raw.get("count", 0))
        for i in range(count):
            team_data = teams_raw[str(i)]["team"]
            teams.append(self._parse_team(team_data))
        return teams

    def get_all_teams(self) -> list[dict]:
        data = self._get(f"league/{self.league_key}/teams")
        teams_raw = self._league_section(data, 1).get("teams", {})
        return [self._parse_team(team_data) for team_data in self._iter_block(teams_raw, "team")]

    def get_league_rosters(self, league_key: str, include_team_key: bool = True) -> list[dict]:
        """Fetch all rosters for all teams in a league."""
        url = f"league/{league_key}/teams/roster"
        logger.info("get_league_rosters: Fetching from URL=%s", url)
        data = self._get(url)
        logger.info("get_league_rosters: Raw response type=%s, keys=%s", type(data).__name__, list(data.keys()) if isinstance(data, dict) else "N/A")
        try:
            sec = self._league_section(data, 1)
            logger.info("get_league_rosters: League section type=%s, keys=%s", type(sec).__name__, list(sec.keys()) if isinstance(sec, dict) else "N/A")

            teams_raw = sec.get("teams", {})
            logger.info("get_league_rosters: teams_raw type=%s", type(teams_raw).__name__)

            # Handle case where teams is already a list (newer Yahoo API format)
            if isinstance(teams_raw, list):
                logger.info("get_league_rosters: teams_raw is list with %d elements", len(teams_raw))
                teams_raw = {"team": teams_raw}

            all_players = []
            for team_data in self._iter_block(teams_raw, "team"):
                # Handle Yahoo API structure changes - team_data[0] might be list or dict
                if len(team_data) > 0:
                    first_element = team_data[0]
                    if isinstance(first_element, list):
                        # Newer format: team_data[0] is a list, look for dict with team_key
                        team_meta = {}
                        for item in first_element:
                            if isinstance(item, dict) and "team_key" in item:
                                team_meta = item
                                break
                    elif isinstance(first_element, dict):
                        # Older format: team_data[0] is directly the metadata dict
                        team_meta = first_element
                    else:
                        team_meta = {}
                else:
                    team_meta = {}

                team_key = team_meta.get("team_key")
                logger.debug("get_league_rosters: Processing team=%s", team_key)

                roster_wrapper = {}
                for node in team_data:
                    if isinstance(node, dict) and "roster" in node:
                        roster_wrapper = node["roster"]
                        break

                logger.info("get_league_rosters: roster_wrapper type=%s, keys=%s",
                           type(roster_wrapper).__name__, list(roster_wrapper.keys()) if isinstance(roster_wrapper, dict) else "N/A")

                players_processed = 0  # Initialize for both format paths

                # Resolve players_raw from roster_wrapper.
                # Format A (old): roster_wrapper["players"] = {"0": {"player": [...]}, ...}
                # Format B (2026): roster_wrapper["0"]["players"] = {"0": {"player": [...]}, ...}
                #   i.e. a single numeric key wraps the players dict.
                players_raw = {}
                if isinstance(roster_wrapper, dict):
                    players_raw = roster_wrapper.get("players", {})
                    if not players_raw:
                        # Format B: look for a numeric key containing "players"
                        for rk, rv in roster_wrapper.items():
                            if rk.isdigit() and isinstance(rv, dict) and "players" in rv:
                                players_raw = rv["players"]
                                break

                if isinstance(players_raw, list):
                    players_raw = {"player": players_raw}

                logger.info("get_league_rosters: players_raw type=%s, n_keys=%s",
                            type(players_raw).__name__,
                            len(players_raw) if isinstance(players_raw, dict) else "N/A")

                # _iter_block handles both {"count": N, "0": {"player": ...}} and
                # plain numeric-key dicts without a count.
                players_processed = 0
                for player_list in self._iter_block(players_raw, "player"):
                    player_dict = self._parse_player(player_list)
                    if include_team_key:
                        player_dict["team_key"] = team_key
                    all_players.append(player_dict)
                    players_processed += 1

                logger.debug("get_league_rosters: Team %s processed %d players", team_key, players_processed)

            logger.info("get_league_rosters: Returning %d total players", len(all_players))
            return all_players
        except Exception as exc:
            logger.error("get_league_rosters failed: %s", exc, exc_info=True)
            return []

    def get_my_team_key(self) -> str:
        """Return the team key for the authenticated user's team."""
        data = self._get(f"league/{self.league_key}/teams")
        teams_raw = self._league_section(data, 1).get("teams", {})
        
        for team_list in self._iter_block(teams_raw, "team"):
            meta = {}
            
            # Handle deeply nested Yahoo response structures (Bugfix March 28)
            def flatten_team_data(obj, depth=0):
                """Recursively extract team metadata from nested lists/dicts."""
                if depth > 5:
                    return
                if isinstance(obj, list):
                    for item in obj:
                        flatten_team_data(item, depth + 1)
                elif isinstance(obj, dict):
                    # Check for is_owned_by_current_login at any level
                    if "is_owned_by_current_login" in obj:
                        meta["is_owned_by_current_login"] = obj["is_owned_by_current_login"]
                    # Extract key fields
                    for key in ["team_key", "team_id", "name", "is_owned_by_current_login"]:
                        if key in obj:
                            meta[key] = obj[key]
                    # Recurse into nested structures
                    for v in obj.values():
                        if isinstance(v, (list, dict)):
                            flatten_team_data(v, depth + 1)
            
            flatten_team_data(team_list)
            
            if meta.get("is_owned_by_current_login"):
                team_key = meta.get("team_key")
                if team_key:
                    return team_key
        
        # Fallback: try environment variable
        import os
        env_team_key = os.getenv("YAHOO_TEAM_KEY")
        if env_team_key:
            logger.warning(f"Using YAHOO_TEAM_KEY from env: {env_team_key}")
            return env_team_key
            
        raise YahooAPIError("Could not find your team — are you authenticated?")

    def get_faab_balance(self) -> Optional[float]:
        """Return authenticated user's remaining FAAB budget (None if not a FAAB league)."""
        try:
            data = self._get(f"league/{self.league_key}/teams")
            teams_raw = self._league_section(data, 1).get("teams", {})
            for team_list in self._iter_block(teams_raw, "team"):
                meta = {}
                entries = team_list if isinstance(team_list, list) else [team_list]
                for d in entries:
                    if isinstance(d, list):
                        for item in d:
                            if isinstance(item, dict):
                                meta.update(item)
                    elif isinstance(d, dict):
                        meta.update(d)
                if meta.get("is_owned_by_current_login"):
                    val = meta.get("faab_balance")
                    return float(val) if val is not None else None
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Roster endpoints
    # ------------------------------------------------------------------

    def get_roster_raw(self, team_key: Optional[str] = None) -> dict:
        """Return the raw fantasy_content payload for debugging Yahoo response shapes."""
        if team_key is None:
            team_key = self.get_my_team_key()
        data = self._get(f"team/{team_key}/roster/players")
        return data.get("fantasy_content", {})

    def get_roster(self, team_key: Optional[str] = None) -> list[dict]:
        """Return full roster for team_key (defaults to authenticated user's team).

        Includes selected_position field indicating Yahoo lineup slot:
        - "IL", "IL10", "IL60" = Injured List (don't count against active roster)
        - "BN" = Bench
        - "C", "1B", "2B", "3B", "SS", "OF", "Util" = Active lineup slots
        - "SP", "RP", "P" = Pitcher slots
        """
        if team_key is None:
            team_key = self.get_my_team_key()
        data = self._get(f"team/{team_key}/roster/players")
        team_data = self._team_section(data)
        roster = self._safe_get(team_data, "roster")
        slot_0 = self._safe_get(roster, "0")
        players_raw = self._safe_get(slot_0, "players")
        
        # Deduplicate by player_key to prevent roster page duplicates (Bugfix March 28)
        players_by_key: dict[str, dict] = {}
        count = int(players_raw.get("count", 0))
        
        for i in range(count):
            entry = players_raw.get(str(i), {})
            entry = self._flatten_league_section(entry) if isinstance(entry, list) else entry
            player_data = entry.get("player", entry) if isinstance(entry, dict) else entry
            p = self._parse_player(player_data)

            # Extract selected_position from roster data (indicates IL, BN, or active slot)
            selected_pos = self._extract_selected_position(player_data)
            if selected_pos:
                p["selected_position"] = selected_pos
            
            # Deduplicate by player_key (line 447-451)
            player_key_val = p.get("player_key")
            if player_key_val and player_key_val not in players_by_key:
                players_by_key[player_key_val] = p
            elif not player_key_val:
                # Fallback: use player_id if player_key is missing
                player_id = p.get("player_id") or p.get("name", f"unknown_{i}")
                if player_id not in players_by_key:
                    players_by_key[player_id] = p
        
        return list(players_by_key.values())

    @staticmethod
    def _extract_selected_position(player_data) -> Optional[str]:
        """Extract selected_position (IL, BN, C, 1B, etc.) from Yahoo player data.

        Yahoo returns this as a sibling to player metadata:
        [{player_key...}, {name...}, {selected_position: {position: "IL"}}]
        """
        if not isinstance(player_data, list):
            return None

        for item in player_data:
            if isinstance(item, dict) and "selected_position" in item:
                sp = item["selected_position"]
                if isinstance(sp, dict):
                    return sp.get("position")
                elif isinstance(sp, list):
                    for spd in sp:
                        if isinstance(spd, dict) and "position" in spd:
                            return spd["position"]
        return None

    def get_all_rosters(self) -> dict[str, list[dict]]:
        """All rosters keyed by team_key."""
        teams = self.get_all_teams()
        rosters = {}
        for team in teams:
            try:
                rosters[team["team_key"]] = self.get_roster(team["team_key"])
            except YahooAPIError as e:
                logger.warning(f"Failed to fetch roster for {team['name']}: {e}")
        return rosters

    # ------------------------------------------------------------------
    # Player endpoints
    # ------------------------------------------------------------------

    def get_player(self, player_key: str) -> dict:
        data = self._get(f"player/{player_key}")
        return self._parse_player(data["fantasy_content"]["player"][0])

    def search_players(self, name: str, status: str = "A") -> list[dict]:
        """
        Search available players by name.
        status: A=available, T=taken, W=on waivers, FA=free agent
        """
        data = self._get(
            f"league/{self.league_key}/players",
            params={"search": name, "status": status},
        )
        players_raw = self._league_section(data, 1).get("players", {})
        return self._parse_players_block(players_raw)

    def get_free_agents(self, position: str = "", start: int = 0, count: int = 25) -> list[dict]:
        """Paginated available players (free agents + waivers, status=A).

        sort=AR ranks by percent rostered — the only sort that reflects real
        pickup value. Yahoo's default sort order is opaque and unreliable.

        HOTFIX (K-24 regression): out=stats is NOT a valid subresource on
        league/.../players — Yahoo returns 400 "Invalid subresource stats
        requested". Stats are fetched separately via get_players_stats_batch()
        and merged in as a best-effort enrichment step so a stats API failure
        cannot take down the waiver surface.
        """
        params = {"status": "A", "start": start, "count": count, "sort": "AR"}
        if position:
            params["position"] = position
        data = self._get(f"league/{self.league_key}/players", params=params)
        players_raw = self._league_section(data, 1).get("players", {})
        players = self._parse_players_block(players_raw)

        # Best-effort: enrich with season stats via the supported batch endpoint.
        # If the call fails for any reason the players list is still returned
        # with stats={} for each player — the waiver endpoint must not 503.
        try:
            player_keys = [p["player_key"] for p in players if p.get("player_key")]
            if player_keys:
                stats_map = self.get_players_stats_batch(player_keys)
                for p in players:
                    if "stats" not in p:
                        p["stats"] = stats_map.get(p.get("player_key") or "", {})
        except Exception as _stats_err:
            logger.warning("get_free_agents stats batch failed (non-fatal): %s", _stats_err)

        return players

    def get_players_stats_batch(self, player_keys: list, stat_type: str = "season") -> dict:
        """Fetch season stats for a batch of players via the supported batch stats endpoint.

        Uses: league/{league_key}/players;player_keys={k1},{k2},.../stats;type={stat_type}
        Returns: {player_key: {stat_id_str: value_str, ...}}
        Max 25 player_keys per Yahoo hard limit.
        """
        if not player_keys:
            return {}
        keys_str = ",".join(player_keys[:25])
        data = self._get(
            f"league/{self.league_key}/players;player_keys={keys_str}/stats;type={stat_type}"
        )
        result: dict = {}
        players_raw = self._league_section(data, 1).get("players", {})
        for p in self._iter_block(players_raw, "player"):
            player_key: Optional[str] = None
            stats_raw: dict = {}
            if isinstance(p, list):
                for item in p:
                    if isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict) and "player_key" in sub:
                                player_key = sub["player_key"]
                    elif isinstance(item, dict):
                        if "player_stats" in item:
                            for stat_entry in item["player_stats"].get("stats", []):
                                if isinstance(stat_entry, dict):
                                    s = stat_entry.get("stat", {})
                                    sid = s.get("stat_id")
                                    if sid is not None:
                                        stats_raw[str(sid)] = s.get("value", "")
            if player_key and stats_raw:
                result[player_key] = stats_raw
        return result

    def get_player_stats(self, player_key: str, stat_type: str = "season") -> dict:
        """
        stat_type: 'season', 'average', 'projected_season'
        """
        data = self._get(f"player/{player_key}/stats;type={stat_type}")
        player = data["fantasy_content"]["player"]
        return self._parse_player_with_stats(player)

    def get_waiver_players(self, start: int = 0, count: int = 25) -> list[dict]:
        params = {"status": "W", "start": start, "count": count}
        data = self._get(f"league/{self.league_key}/players", params=params)
        players_raw = self._league_section(data, 1).get("players", {})
        return self._parse_players_block(players_raw)

    def get_adp_and_injury_feed(
        self,
        pages: int = 4,
        count_per_page: int = 25,
    ) -> list[dict]:
        """Paginated ADP + injury status snapshot for all rostered + available players.

        Fetches all players sorted by Average Draft Position (sort=DA) so that
        the caller can detect rank movement and surface new injury flags.
        Returns up to pages*count_per_page players (default 100).

        Each returned dict has:
            player_key, name, team, positions, status, injury_note,
            is_undroppable, percent_owned
        """
        results: list[dict] = []
        for page in range(pages):
            start = page * count_per_page
            try:
                params = {"start": start, "count": count_per_page, "sort": "DA"}
                data = self._get(f"league/{self.league_key}/players", params=params)
                players_raw = self._league_section(data, 1).get("players", {})
                batch = self._parse_players_block(players_raw)
                if not batch:
                    break  # Yahoo returned empty page — no more players
                results.extend(batch)
            except YahooAPIError as exc:
                logger.warning(
                    "get_adp_and_injury_feed page %d failed (%s) — returning partial results",
                    page, exc,
                )
                break
        return results

    # ------------------------------------------------------------------
    # Draft endpoints
    # ------------------------------------------------------------------

    def get_draft_results(self) -> list[dict]:
        """Return completed draft picks (empty until draft runs)."""
        data = self._get(f"league/{self.league_key}/draftresults")
        picks_raw = (
            self._league_section(data, 1)
            .get("draft_results", {})
            .get("0", {})
            .get("draft_results", {})
        )
        picks = []
        count = int(picks_raw.get("count", 0))
        for i in range(count):
            pick = picks_raw[str(i)]["draft_result"][0]
            picks.append({
                "pick": pick.get("pick"),
                "round": pick.get("round"),
                "team_key": pick.get("team_key"),
                "player_key": pick.get("player_key"),
            })
        return picks

    # ------------------------------------------------------------------
    # Lineup management
    # ------------------------------------------------------------------

    def get_lineup(self, team_key: Optional[str] = None, date: Optional[str] = None) -> list[dict]:
        """Fetch current lineup for a date (YYYY-MM-DD). Defaults to today."""
        if team_key is None:
            team_key = self.get_my_team_key()
        if date is None:
            date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        data = self._get(f"team/{team_key}/roster/players", params={"date": date})
        players_raw = (
            self._team_section(data)
            .get("roster", {})
            .get("0", {})
            .get("players", {})
        )
        players = []
        count = int(players_raw.get("count", 0))
        for i in range(count):
            player_data = players_raw[str(i)]["player"]
            p = self._parse_player(player_data)
            # selected_position may be in the player data
            if isinstance(player_data, list):
                for item in player_data:
                    if isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict) and "selected_position" in sub:
                                sp = sub["selected_position"]
                                if isinstance(sp, list):
                                    for spd in sp:
                                        if isinstance(spd, dict) and "position" in spd:
                                            p["selected_position"] = spd["position"]
            players.append(p)
        return players

    def set_lineup(self, team_key: Optional[str] = None, date: Optional[str] = None,
                   lineup: Optional[list[dict]] = None) -> dict:
        """
        Set lineup for a given date.

        lineup: list of {player_key: str, position: str}
            position: 'C','1B','2B','3B','SS','OF','Util','SP','RP','P','BN','DL'

        Returns dict: {applied: [player_keys], skipped: [player_keys], warnings: [str]}
        Raises YahooAPIError only when ALL players fail and no partial success is possible.
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        if team_key is None:
            team_key = self.get_my_team_key()
        if date is None:
            date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        if not lineup:
            return {"applied": [], "skipped": [], "warnings": []}

        self._ensure_token()
        url = f"{YAHOO_API_BASE}/team/{team_key}/roster"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/xml",
        }

        def _build_xml(players: list) -> bytes:
            player_xml = "\n".join(
                f'<player><player_key>{p["player_key"]}</player_key>'
                f'<position>{p["position"]}</position></player>'
                for p in players
            )
            return (
                f'<?xml version="1.0"?>'
                f'<fantasy_content><roster><coverage_type>date</coverage_type>'
                f'<date>{date}</date><players>{player_xml}</players></roster></fantasy_content>'
            ).encode("utf-8")

        # Attempt full batch first (fast path)
        resp = self._session.put(url, data=_build_xml(lineup), headers=headers)
        if resp.status_code in (200, 204):
            return {"applied": [p["player_key"] for p in lineup], "skipped": [], "warnings": []}

        # If Yahoo returns "game_ids don't match", fall back to player-by-player
        # so stale/traded players are skipped rather than blocking the whole lineup.
        if "game_ids" in resp.text or "game_id" in resp.text.lower():
            _log.warning("set_lineup batch rejected (game_id mismatch) — retrying per-player")
            applied, skipped, warnings = [], [], []
            for p in lineup:
                r = self._session.put(url, data=_build_xml([p]), headers=headers)
                if r.status_code in (200, 204):
                    applied.append(p["player_key"])
                else:
                    skipped.append(p["player_key"])
                    msg = f"Skipped {p['player_key']} (pos={p['position']}): {r.text[:200]}"
                    _log.warning(msg)
                    warnings.append(msg)
            if not applied:
                raise YahooAPIError(
                    f"set_lineup failed for all {len(skipped)} player(s): {resp.text[:300]}",
                    resp.status_code,
                )
            return {"applied": applied, "skipped": skipped, "warnings": warnings}

        raise YahooAPIError(f"set_lineup failed: {resp.status_code} — {resp.text[:300]}", resp.status_code)

    def get_scoreboard(self, week: Optional[int] = None) -> list[dict]:
        """Fetch matchup scoreboard for a week (defaults to current).

        Yahoo's scoreboard nesting changed between API versions:
          v1: league[1].scoreboard.0.matchups   (older format)
          v2: league[1].scoreboard.matchups      (2025+ format)
          v3: league[1].scoreboard.0.matchups.0  (nested indexed format)
        Multiple paths are tried so this survives Yahoo response shape changes.
        """
        path = f"league/{self.league_key}/scoreboard"
        params = {}
        if week:
            params["week"] = week
        data = self._get(path, params=params if params else None)
        sec = self._league_section(data, 1)
        scoreboard = sec.get("scoreboard", {})

        # Try v2 path first (scoreboard.matchups)
        matchups_raw = scoreboard.get("matchups", {})

        # Fall back to v1 path (scoreboard.0.matchups)
        if not matchups_raw:
            matchups_raw = scoreboard.get("0", {}).get("matchups", {})

        # If scoreboard itself is a list, flatten it
        if isinstance(scoreboard, list):
            flat = {}
            for item in scoreboard:
                if isinstance(item, dict):
                    flat.update(item)
            matchups_raw = flat.get("matchups", {})
        
        # Deep search for matchups structure if still not found
        if not matchups_raw and isinstance(scoreboard, dict):
            # Search recursively for matchups
            def find_matchups(obj, depth=0):
                if depth > 3:
                    return None
                if isinstance(obj, dict):
                    if "matchups" in obj:
                        return obj["matchups"]
                    for v in obj.values():
                        result = find_matchups(v, depth + 1)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_matchups(item, depth + 1)
                        if result:
                            return result
                return None
            matchups_raw = find_matchups(scoreboard) or {}

        matchups = []
        
        # Handle both dict (indexed) and list formats
        if isinstance(matchups_raw, list):
            for entry in matchups_raw:
                if isinstance(entry, dict):
                    if "matchup" in entry:
                        matchups.append(entry["matchup"])
                    else:
                        matchups.append(entry)
        elif isinstance(matchups_raw, dict):
            count = int(matchups_raw.get("count", 0))
            for i in range(count):
                entry = matchups_raw.get(str(i), {})
                if isinstance(entry, dict):
                    if "matchup" in entry:
                        matchups.append(entry["matchup"])
                    else:
                        matchups.append(entry)
        
        return matchups

    def add_drop_player(self, add_player_key: str, drop_player_key: Optional[str] = None,
                        team_key: Optional[str] = None) -> bool:
        """Add a free agent (and optionally drop a player)."""
        if team_key is None:
            team_key = self.get_my_team_key()
        self._ensure_token()
        drop_xml = (
            f'<player><player_key>{drop_player_key}</player_key>'
            f'<transaction_data><type>drop</type>'
            f'<destination_team_key>LW</destination_team_key></transaction_data></player>'
        ) if drop_player_key else ""
        xml_body = (
            f'<?xml version="1.0"?><fantasy_content><transaction>'
            f'<type>{"add/drop" if drop_player_key else "add"}</type>'
            f'<trader_team_key>{team_key}</trader_team_key>'
            f'<players>'
            f'<player><player_key>{add_player_key}</player_key>'
            f'<transaction_data><type>add</type>'
            f'<destination_team_key>{team_key}</destination_team_key></transaction_data></player>'
            f'{drop_xml}</players></transaction></fantasy_content>'
        )
        url = f"{YAHOO_API_BASE}/league/{self.league_key}/transactions"
        resp = self._session.post(
            url,
            data=xml_body.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/xml",
            },
        )
        if resp.status_code not in (200, 201):
            raise YahooAPIError(f"add/drop failed: {resp.status_code} — {resp.text[:300]}", resp.status_code)
        return True

    def get_transactions(self, t_type: str = "add,drop,trade") -> list[dict]:
        """Recent transactions for the league."""
        data = self._get(
            f"league/{self.league_key}/transactions",
            params={"type": t_type},
        )
        txns_raw = self._league_section(data, 1).get("transactions", {})
        txns = []
        count = int(txns_raw.get("count", 0))
        for i in range(count):
            txns.append(txns_raw[str(i)].get("transaction", {}))
        return txns

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_team(team_list: list) -> dict:
        """Flatten Yahoo's nested team structure."""
        meta = {}
        if isinstance(team_list[0], list):
            for item in team_list[0]:
                if isinstance(item, dict):
                    meta.update(item)
        return {
            "team_key": meta.get("team_key"),
            "team_id": meta.get("team_id"),
            "name": meta.get("name"),
            "manager": meta.get("managers", [{}])[0].get("manager", {}).get("nickname"),
        }

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        """Safely convert value to float, returning default on failure or NaN."""
        if value is None:
            return default
        try:
            result = float(value)
            # Check for NaN (NaN != NaN)
            if result != result:
                return default
            return result
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_player(player_list: list) -> dict:
        """Flatten Yahoo's nested player structure."""
        meta = {}
        
        # Defensive parsing for nested list structures (Bugfix March 28)
        def flatten_player_data(obj, depth=0):
            """Recursively extract player metadata from nested lists/dicts."""
            if depth > 5:
                return
            if isinstance(obj, list):
                for item in obj:
                    flatten_player_data(item, depth + 1)
            elif isinstance(obj, dict):
                meta.update(obj)
                # Recurse into nested values that might contain more data
                for v in obj.values():
                    if isinstance(v, (list, dict)):
                        flatten_player_data(v, depth + 1)
        
        flatten_player_data(player_list)

        # Extract eligible positions
        positions_raw = meta.get("eligible_positions", [])
        positions = []
        if isinstance(positions_raw, list):
            positions = [p.get("position") for p in positions_raw if isinstance(p, dict)]
        elif isinstance(positions_raw, dict):
            pos = positions_raw.get("position")
            positions = [pos] if pos else []

        # Extract percent_owned — Yahoo returns this inside an "ownership" sub-block
        # at the outer player list level (not inside metadata).
        owned_pct = 0.0
        
        # Search recursively for ownership data
        def find_ownership(obj, depth=0):
            nonlocal owned_pct
            if depth > 5 or owned_pct > 0:
                return
            if isinstance(obj, list):
                for item in obj:
                    find_ownership(item, depth + 1)
            elif isinstance(obj, dict):
                if "ownership" in obj:
                    own = obj["ownership"]
                    # Yahoo 2025+ returns percent_rostered; older format used percent_owned
                    for _key in ("percent_rostered", "percent_owned"):
                        pct_block = own.get(_key)
                        if pct_block is not None:
                            if isinstance(pct_block, dict):
                                raw = pct_block.get("value", 0)
                            else:
                                raw = pct_block
                            owned_pct = YahooFantasyClient._safe_float(raw, 0.0)
                            if owned_pct > 0:
                                break
                # Also check for old flat format
                elif "percent_rostered" in obj:
                    owned_pct = YahooFantasyClient._safe_float(obj["percent_rostered"], 0.0)
                elif "percent_owned" in obj and isinstance(obj.get("percent_owned"), dict):
                    owned_pct = YahooFantasyClient._safe_float(
                        obj["percent_owned"].get("value", 0), 0.0
                    )
                # Recurse
                for v in obj.values():
                    if isinstance(v, (list, dict)):
                        find_ownership(v, depth + 1)
        
        find_ownership(player_list)
        
        # Fallback: old "percent_owned" key directly in metadata (pre-2025 format)
        if owned_pct == 0.0:
            owned_raw = meta.get("percent_owned", 0)
            if isinstance(owned_raw, dict):
                owned_pct = YahooFantasyClient._safe_float(owned_raw.get("value", 0), 0.0)
            else:
                owned_pct = YahooFantasyClient._safe_float(owned_raw, 0.0)

        # Extract name with defensive handling
        name = meta.get("full_name")
        if not name and isinstance(meta.get("name"), dict):
            name = meta["name"].get("full")
        if not name:
            name = meta.get("name", "Unknown")
        # Strip injury descriptions occasionally appended by Yahoo to the name field
        # e.g. "Jason Adam Quadriceps" -> "Jason Adam"
        if isinstance(name, str):
            name = re.sub(
                r"\s+(?:Quadriceps|Hamstring|Shoulder|Elbow|Hip|Knee|Back|Wrist|Ankle|"
                r"Oblique|Forearm|Calf|Groin|Thumb|Finger|Ribs?|Concussion|"
                r"Strain|Sprain|Fracture|Tear|Surgery|Illness|Fatigue|IL|DL)\b.*$",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()
        
        return {
            "player_key": meta.get("player_key"),
            "player_id": meta.get("player_id"),
            "name": name,
            "team": meta.get("editorial_team_abbr"),
            "positions": [p for p in positions if p],
            "status": meta.get("status") or None,
            "injury_note": meta.get("injury_note") or None,
            "is_undroppable": meta.get("is_undroppable", 0) in (1, '1', True, 'true'),
            "percent_owned": owned_pct,
        }

    def _parse_player_with_stats(self, player: list) -> dict:
        parsed = self._parse_player(player[0] if isinstance(player[0], list) else player)
        stats_raw = {}
        for item in player:
            if isinstance(item, dict) and "player_stats" in item:
                stats_list = item["player_stats"].get("stats", [])
                for stat_entry in stats_list:
                    if isinstance(stat_entry, dict):
                        s = stat_entry.get("stat", {})
                        stats_raw[s.get("stat_id")] = s.get("value")
        parsed["stats"] = stats_raw
        return parsed

    def _parse_players_block(self, players_raw) -> list[dict]:
        results = []
        for p in self._iter_block(players_raw, "player"):
            parsed = self._parse_player(p)
            # K-24: extract season stats when present (out=stats on free agent calls)
            stats_raw = {}
            if isinstance(p, list):
                for item in p:
                    if isinstance(item, dict) and "player_stats" in item:
                        stats_list = item["player_stats"].get("stats", [])
                        for stat_entry in stats_list:
                            if isinstance(stat_entry, dict):
                                s = stat_entry.get("stat", {})
                                sid = s.get("stat_id")
                                if sid is not None:
                                    stats_raw[str(sid)] = s.get("value", "")
            if stats_raw:
                parsed["stats"] = stats_raw
            results.append(parsed)
        return results


# ---------------------------------------------------------------------------
# CLI: one-time auth setup
# ---------------------------------------------------------------------------

def run_auth_flow():
    """Interactive OAuth setup — run once to get refresh token."""
    load_dotenv()
    client = YahooFantasyClient()

    auth_url = client.get_authorization_url()
    print("\n" + "=" * 60)
    print("YAHOO FANTASY — ONE-TIME AUTHORIZATION")
    print("=" * 60)
    print(f"\nStep 1: Open this URL in your browser:\n\n  {auth_url}\n")
    try:
        webbrowser.open(auth_url)
        print("(Browser opened automatically)")
    except Exception:
        pass

    print("\nStep 2: Authorize the app")
    print("Step 3: Yahoo will show you a 6-digit code (or redirect to oob://)")
    code = input("\nEnter the authorization code: ").strip()

    tokens = client.exchange_code_for_tokens(code)
    print("\nAuthorization successful!")
    print(f"  Access token expires in: {tokens.get('expires_in', '?')}s")
    print("  Refresh token saved to .env")

    # Quick test
    try:
        league = client.get_league()
        print(f"\nConnected to league: {league.get('name')}")
        my_key = client.get_my_team_key()
        print(f"  Your team key: {my_key}")
    except Exception as e:
        print(f"\nAuth succeeded but test call failed: {e}")
        print("  Tokens are saved — try again after retrying.")


# ---------------------------------------------------------------------------
# Resilience layer dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaiverResponse:
    """Standardized waiver wire response."""
    players: List[Dict]
    source: str  # "yahoo_api", "cache", "projection_estimate"
    fresh: bool
    errors: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is [] or self.errors is None:
            self.errors = []


@dataclass
class LineupResult:
    """Result of lineup setting operation."""
    success: bool
    changes: Optional[Dict] = None
    errors: List[str] = None
    warnings: List[str] = None
    retry_possible: bool = False
    suggested_action: Optional[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


# ---------------------------------------------------------------------------
# Resilient client — extends YahooFantasyClient with circuit breaker + cache
# ---------------------------------------------------------------------------

class ResilientYahooClient(YahooFantasyClient):
    """
    Yahoo client with resilience patterns.

    Extends the base YahooFantasyClient with:
    - Circuit breaker to prevent cascading failures
    - Fallback to metadata-only when percent_owned fails
    - Position normalization to prevent lineup mismatches
    - Stale cache for availability during outages

    Usage:
        client = ResilientYahooClient()  # Same init as YahooFantasyClient

        # Waiver wire with automatic fallback
        result = await client.get_waiver_players("mlb.l.12345")
        if not result.fresh:
            logger.warning(f"Serving stale data: {result.errors}")

        # Lineup with validation
        result = await client.set_lineup_resilient(team_id, optimized_lineup)
        if not result.success:
            print(result.suggested_action)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize resilience components
        self.circuit = CircuitBreaker(
            name="yahoo_fantasy_api",
            failure_threshold=3,
            recovery_timeout=300,  # 5 minutes
            expected_exception=Exception,
        )

        self.cache = StaleCacheManager(
            cache_dir=os.getenv("YAHOO_CACHE_DIR", ".cache/fantasy"),
            max_age=timedelta(hours=int(os.getenv("YAHOO_CACHE_TTL_HOURS", "24"))),
            enabled=os.getenv("YAHOO_CACHE_DISABLED", "false").lower() != "true"
        )

        self.position_normalizer = PositionNormalizer()
        self.lineup_validator = LineupValidator()

        # Track ADP data path for fallback enrichment
        self.adp_data_path = os.getenv(
            "ADP_DATA_PATH",
            "/app/data/projections/adp_yahoo_2026.csv"
        )

    # ==================================================================
    # Waiver Wire Operations
    # ==================================================================

    async def get_waiver_players(self, league_id: str, **filters) -> WaiverResponse:
        """
        Get waiver wire players with full fallback chain.

        Fallback order:
        1. Try API with percent_owned (normal)
        2. Try API with metadata only + ADP enrichment
        3. Serve from cache if API unavailable
        4. Fail with clear error if no data available
        """
        cache_key = f"waiver_{league_id}_{hash(str(sorted(filters.items())))}"

        try:
            # Attempt 1: Circuit breaker wrapped API call with fallback
            players = await self.circuit.call_async(
                self._fetch_waiver_with_fallback,
                league_id,
                filters
            )

            # Cache successful result
            self.cache.write(cache_key, players, metadata={"filters": filters})

            return WaiverResponse(
                players=players,
                source="yahoo_api",
                fresh=True,
                errors=[],
                metadata={"count": len(players), "cached": False}
            )

        except CircuitOpenError:
            # Circuit open - use cache
            logger.warning("Circuit open for waiver fetch, checking cache")
            cached = self.cache.read(cache_key)

            if cached:
                return WaiverResponse(
                    players=cached.data,
                    source="cache",
                    fresh=False,
                    errors=["Yahoo API circuit open: serving stale data"],
                    metadata={
                        "count": len(cached.data),
                        "cached": True,
                        "cache_age_hours": self.cache.get_age_hours(cached)
                    }
                )

            # No cache available
            return WaiverResponse(
                players=[],
                source="unavailable",
                fresh=False,
                errors=["Yahoo API unavailable and no cache available"],
                metadata={"circuit_open": True}
            )

        except NoDataAvailableError as e:
            return WaiverResponse(
                players=[],
                source="unavailable",
                fresh=False,
                errors=[str(e)],
                metadata={"error_type": "no_data_available"}
            )

    async def _fetch_waiver_with_fallback(
        self,
        league_id: str,
        filters: Dict
    ) -> List[Dict]:
        """
        Internal: Try primary fetch, fallback to metadata-only on percent_owned error.
        """
        try:
            # Primary: Try normal fetch (assumes parent has this method)
            return await self._fetch_waiver_primary(league_id, filters)

        except Exception as e:
            error_str = str(e).lower()

            # Check if it's the percent_owned error
            if "percent_owned" in error_str or "subresource" in error_str:
                logger.warning(
                    f"percent_owned subresource failed, using metadata fallback: {e}"
                )
                return await self._fetch_waiver_metadata_only(league_id, filters)

            # Re-raise other errors
            raise

    async def _fetch_waiver_primary(self, league_id: str, filters: Dict) -> List[Dict]:
        """Primary waiver fetch via sync parent, run in thread pool."""
        start = filters.get("start", 0)
        count = filters.get("count", 25)
        return await asyncio.to_thread(
            super(ResilientYahooClient, self).get_waiver_players,
            start, count
        )

    async def _fetch_waiver_metadata_only(
        self,
        league_id: str,
        filters: Dict
    ) -> List[Dict]:
        """
        Fallback: Fetch metadata only and enrich with ADP estimates.
        """
        logger.info(f"Fetching waiver metadata only for {league_id}")

        # Fetch without percent_owned subresource
        # This assumes YahooClient has a way to specify subresources
        # Adjust the call based on your actual YahooClient API

        players = await self._fetch_with_subresources(
            league_id,
            subresources="metadata",
            **filters
        )

        # Enrich with ADP-based ownership estimates
        adp_data = self._load_adp_data()

        for player in players:
            player_name = player.get("name", "")
            estimated = self._estimate_ownership_from_adp(player_name, adp_data)
            player["percent_owned"] = estimated
            player["percent_owned_estimated"] = True
            player["percent_owned_source"] = "adp_proxy"

        return players

    async def _fetch_with_subresources(
        self,
        league_id: str,
        subresources: str,
        **filters
    ) -> List[Dict]:
        """Fetch players with specified subresources."""
        url = f"/fantasy/v2/league/{league_id}/players"
        params = {
            "out": subresources,
            "format": "json",
            **filters
        }

        # Call parent's request method
        response = await self._make_request(url, params)
        return self._parse_players_response(response)

    def _load_adp_data(self) -> Dict[str, float]:
        """Load ADP data for ownership estimation."""
        adp_map = {}
        try:
            with open(self.adp_data_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "").strip()
                    adp = row.get("ADP", "")
                    if name and adp:
                        try:
                            adp_map[name] = float(adp)
                        except ValueError:
                            pass
        except FileNotFoundError:
            logger.warning(f"ADP data not found at {self.adp_data_path}")

        return adp_map

    def _estimate_ownership_from_adp(
        self,
        player_name: str,
        adp_data: Dict[str, float]
    ) -> float:
        """
        Estimate ownership percentage from ADP.

        Lower ADP (drafted earlier) = higher ownership
        This is a rough heuristic - adjust formula as needed.
        """
        adp = adp_data.get(player_name)
        if not adp:
            return 0.0  # Unknown players = 0% owned

        # Rough estimation: ADP 1-50 ~ 90-100%, ADP 200+ ~ 0-10%
        if adp <= 50:
            return max(0, 100 - (adp - 1) * 0.2)  # 100% at 1, 90% at 50
        elif adp <= 100:
            return max(0, 90 - (adp - 50) * 1.2)  # 90% at 50, 30% at 100
        elif adp <= 200:
            return max(0, 30 - (adp - 100) * 0.2)  # 30% at 100, 10% at 200
        else:
            return max(0, 10 - (adp - 200) * 0.05)  # 10% at 200, 0% at 400

    # ==================================================================
    # Lineup Operations
    # ==================================================================

    async def set_lineup_resilient(
        self,
        team_id: str,
        optimized_lineup: Dict[str, Any],
        auto_correct: bool = True,
    ) -> LineupResult:
        """
        Set lineup with pre-validation and graceful degradation.

        Steps:
        1. Get current Yahoo roster
        2. Normalize positions between optimizer and Yahoo
        3. Game-aware validation (check players have games today)
        4. Auto-correct if enabled (swap players with no games)
        5. Execute with circuit breaker
        """
        try:
            # Step 1: Get current Yahoo roster
            yahoo_roster = await self._get_yahoo_roster(team_id)

            # Step 2: Normalize positions
            try:
                normalized_assignments = self.position_normalizer.normalize_lineup(
                    optimized_lineup,
                    yahoo_roster,
                    strict=False  # Don't fail on unmatched slots
                )
            except LineupValidationError as e:
                return LineupResult(
                    success=False,
                    errors=[str(e)],
                    retry_possible=False,
                    suggested_action="Check position eligibility in optimizer vs Yahoo roster"
                )

            # Step 3: Position validation before API call
            position_validation = self.position_normalizer.validate_lineup_before_submit(
                normalized_assignments,
                yahoo_roster
            )

            if not position_validation.valid:
                return LineupResult(
                    success=False,
                    errors=position_validation.errors,
                    warnings=position_validation.warnings,
                    retry_possible=False,
                    suggested_action="Fix position mismatches before retrying"
                )

            # Log warnings but proceed
            if position_validation.warnings:
                logger.warning(f"Lineup warnings: {position_validation.warnings}")

            # Step 4: Game-aware validation
            slot_lookup = {s.id: s for s in yahoo_roster.slots}

            optimized_slots = []
            for slot_id, player_id in normalized_assignments.items():
                slot = slot_lookup.get(slot_id)
                player = next((p for p in yahoo_roster.players if p.id == player_id), None)

                optimized_slots.append(OptimizedSlot(
                    slot_id=slot_id,
                    position=slot.position if slot else "Unknown",
                    player_id=player_id,
                    player_name=player.name if player else None
                ))

            roster_players = [
                {
                    "player_id": p.id,
                    "name": p.name,
                    "team": getattr(p, 'team', ''),
                    "positions": p.eligible_positions or p.yahoo_positions or p.positions,
                    "eligible_positions": p.eligible_positions or p.yahoo_positions or p.positions,
                }
                for p in yahoo_roster.players
            ]

            if auto_correct:
                submission = self.lineup_validator.auto_correct_lineup(
                    optimized_slots,
                    roster_players
                )

                if submission.changes_made:
                    logger.info("Lineup auto-corrected:\n" + format_lineup_report(submission))
                    normalized_assignments = submission.assignments
            else:
                game_validation = self.lineup_validator.validate_lineup(
                    optimized_slots,
                    roster_players,
                    strict=False
                )

                if game_validation.invalid_players:
                    player_names = [p.player_name for p in game_validation.invalid_players]
                    return LineupResult(
                        success=False,
                        errors=[f"Players with no games today: {', '.join(player_names)}"],
                        warnings=game_validation.warnings,
                        retry_possible=True,
                        suggested_action="Enable auto_correct or manually adjust lineup"
                    )

            # Step 5: Execute with circuit breaker
            try:
                result = await self.circuit.call_async(
                    self._execute_lineup_set,
                    team_id,
                    normalized_assignments
                )

                return LineupResult(
                    success=True,
                    changes=result,
                    warnings=position_validation.warnings,
                )

            except CircuitOpenError:
                return LineupResult(
                    success=False,
                    errors=["Yahoo API circuit is open (too many failures)"],
                    warnings=position_validation.warnings,
                    retry_possible=True,
                    suggested_action="Wait 5 minutes for circuit to reset, then retry"
                )

        except Exception as e:
            logger.exception("Unexpected error setting lineup")
            return LineupResult(
                success=False,
                errors=[f"Unexpected error: {str(e)}"],
                retry_possible=True,
                suggested_action="Check logs and retry"
            )

    async def _get_yahoo_roster(self, team_id: str) -> YahooRoster:
        """Fetch and parse Yahoo roster."""
        # Note: get_roster is synchronous, returns List[dict]
        roster_list = self.get_roster(team_id)

        slots = []
        players = []

        slot_assignments: Dict[str, str] = {}  # position -> player_id

        for player_data in roster_list:
            player_id = str(player_data.get("player_id") or player_data.get("id", ""))
            selected_pos = player_data.get("selected_position", "BN")

            if selected_pos and selected_pos != "BN":
                slot_assignments[selected_pos] = player_id

            players.append(Player(
                id=player_id,
                name=player_data.get("name", "Unknown"),
                positions=player_data.get("eligible_positions", []),
                yahoo_positions=player_data.get("eligible_positions", []),
                eligible_positions=player_data.get("eligible_positions", []),
                team=player_data.get("editorial_team_abbr") or player_data.get("team", "")
            ))

        for position, player_id in slot_assignments.items():
            slots.append(RosterSlot(
                id=f"slot_{position}",
                position=position,
                player_id=player_id
            ))

        return YahooRoster(slots=slots, players=players)

    async def _execute_lineup_set(
        self,
        team_id: str,
        assignments: Dict[str, str]
    ) -> Dict:
        """Execute the actual lineup API call via sync parent, run in thread pool."""
        lineup_list = [{"player_key": pk, "position": pos}
                       for pos, pk in assignments.items()]
        return await asyncio.to_thread(
            super(ResilientYahooClient, self).set_lineup,
            team_key=team_id,
            lineup=lineup_list
        )

    # ==================================================================
    # Health & Monitoring
    # ==================================================================

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all resilience components."""
        return {
            "circuit_breaker": self.circuit.get_stats(),
            "cache": self.cache.get_stats(),
            "client_type": "ResilientYahooClient",
        }

    def force_circuit_open(self):
        """Manually open circuit (for testing or emergency)."""
        self.circuit.force_open()
        logger.warning("Yahoo API circuit manually opened")

    def force_circuit_close(self):
        """Manually close circuit (after fixing issue)."""
        self.circuit.force_close()
        logger.info("Yahoo API circuit manually closed")

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear_all()


# ---------------------------------------------------------------------------
# Module-level singletons — use these instead of constructing directly
# ---------------------------------------------------------------------------

_client: "Optional[YahooFantasyClient]" = None
_client_lock = threading.Lock()

_resilient_client: "Optional[ResilientYahooClient]" = None
_resilient_client_lock = threading.Lock()


def get_yahoo_client() -> "YahooFantasyClient":
    """
    Return the process-level YahooFantasyClient singleton.

    Thread-safe via double-checked locking. Token refresh is handled
    internally by _ensure_token() -- callers never need to refresh manually.
    """
    global _client
    # NOTE: The outer check is safe on CPython (GIL makes reference
    # assignment atomic) but would be a data race on free-threaded
    # runtimes. If upgrading to Python 3.13+ with --disable-gil,
    # remove the outer check and rely solely on the lock.
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = YahooFantasyClient()
    return _client


def get_resilient_yahoo_client() -> "ResilientYahooClient":
    """
    Return the process-level ResilientYahooClient singleton.

    Use this for endpoints that need circuit-breaker + stale-cache behaviour.
    """
    global _resilient_client
    # NOTE: The outer check is safe on CPython (GIL makes reference
    # assignment atomic) but would be a data race on free-threaded
    # runtimes. If upgrading to Python 3.13+ with --disable-gil,
    # remove the outer check and rely solely on the lock.
    if _resilient_client is not None:
        return _resilient_client
    with _resilient_client_lock:
        if _resilient_client is None:
            _resilient_client = ResilientYahooClient()
    return _resilient_client


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--auth" in sys.argv:
        run_auth_flow()
    else:
        print("Usage: python -m backend.fantasy_baseball.yahoo_client_resilient --auth")
        print("  Runs one-time OAuth setup to get your refresh token.")
