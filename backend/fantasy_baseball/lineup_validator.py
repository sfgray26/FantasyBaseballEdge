"""
Game-aware lineup validator for Yahoo Fantasy Baseball.

Prevents game_id mismatch errors by validating player eligibility
before submitting lineups to Yahoo's API.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import requests

logger = logging.getLogger(__name__)


class GameStatus(Enum):
    """Status of a player's game today."""
    HAS_GAME = "has_game"
    NO_GAME = "no_game"
    DOUBLE_HEADER = "double_header"
    GAME_STARTED = "game_started"
    POSTPONED = "postponed"
    UNKNOWN = "unknown"


@dataclass
class PlayerGameInfo:
    """Information about a player's game today."""
    player_id: str
    player_name: str
    team: str
    opponent: Optional[str] = None
    game_time: Optional[datetime] = None
    is_home: bool = False
    status: GameStatus = GameStatus.UNKNOWN
    double_header_game: Optional[int] = None  # 1 or 2 for DH
    venue: Optional[str] = None
    weather: Optional[Dict] = None


@dataclass
class LineupValidation:
    """Result of lineup validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    invalid_players: List[PlayerGameInfo] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        lines = [f"Lineup Validation: {status}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.suggestions:
            lines.append(f"  Suggestions ({len(self.suggestions)}):")
            for s in self.suggestions:
                lines.append(f"    - {s.get('action', 'Unknown')}")
        return "\n".join(lines)


@dataclass
class OptimizedSlot:
    """A slot in the optimized lineup."""
    slot_id: str
    position: str
    player_id: Optional[str] = None
    player_name: Optional[str] = None


@dataclass
class LineupSubmission:
    """Final lineup ready for submission."""
    assignments: Dict[str, str]  # slot_id -> player_id
    bench_assignments: Dict[str, str]  # benched players
    changes_made: List[str]  # human-readable change log
    validation: LineupValidation


class ScheduleFetcher:
    """
    Fetches today's MLB schedule for game-aware validation.
    
    Uses multiple data sources:
    1. MLB Stats API (primary)
    2. ESPN API (fallback)
    3. Yahoo fantasy game data (tertiary)
    """
    
    MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
    ESPN_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    
    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Optional[Tuple[datetime, Dict]] = None
    
    def get_todays_schedule(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Fetch today's MLB schedule.
        
        Returns dict keyed by team abbreviation with game info.
        """
        if date is None:
            date = datetime.now()
        
        # Check cache
        if self._cache:
            cached_time, cached_data = self._cache
            if datetime.now() - cached_time < self.cache_ttl:
                logger.debug("Using cached schedule")
                return cached_data
        
        # Try MLB Stats API first
        try:
            schedule = self._fetch_mlb_schedule(date)
            self._cache = (datetime.now(), schedule)
            return schedule
        except Exception as e:
            logger.warning(f"MLB Stats API failed: {e}. Trying ESPN...")
        
        # Fallback to ESPN
        try:
            schedule = self._fetch_espn_schedule(date)
            self._cache = (datetime.now(), schedule)
            return schedule
        except Exception as e:
            logger.error(f"All schedule sources failed: {e}")
            return {}
    
    def _fetch_mlb_schedule(self, date: datetime) -> Dict[str, Dict]:
        """Fetch from MLB Stats API."""
        date_str = date.strftime("%Y-%m-%d")
        params = {
            "sportId": 1,  # MLB
            "date": date_str,
            "hydrate": "team,venue,game(content(summary))",
        }
        
        response = requests.get(self.MLB_SCHEDULE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        schedule = {}
        for date_info in data.get("dates", []):
            for game in date_info.get("games", []):
                game_info = self._parse_mlb_game(game)
                
                # Add both teams
                home_team = game["teams"]["home"]["team"]["abbreviation"]
                away_team = game["teams"]["away"]["team"]["abbreviation"]
                
                schedule[home_team] = {**game_info, "is_home": True, "opponent": away_team}
                schedule[away_team] = {**game_info, "is_home": False, "opponent": home_team}
        
        logger.info(f"Fetched {len(schedule)} teams from MLB Stats API")
        return schedule
    
    def _parse_mlb_game(self, game: Dict) -> Dict:
        """Parse MLB game data into standard format."""
        game_time = datetime.fromisoformat(
            game["gameDate"].replace("Z", "+00:00")
        )
        
        return {
            "game_id": game["gamePk"],
            "game_time": game_time,
            "status": game["status"]["detailedState"],
            "venue": game.get("venue", {}).get("name"),
            "double_header": game.get("doubleHeader", "N") != "N",
            "game_number": game.get("gameNumber", 1),
        }
    
    def _fetch_espn_schedule(self, date: datetime) -> Dict[str, Dict]:
        """Fetch from ESPN API as fallback."""
        date_str = date.strftime("%Y%m%d")
        params = {"dates": date_str}
        
        response = requests.get(self.ESPN_SCHEDULE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        schedule = {}
        for event in data.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            
            comp = competitions[0]
            game_time = datetime.fromisoformat(
                event["date"].replace("Z", "+00:00")
            )
            
            for team_data in comp.get("competitors", []):
                team_abbr = team_data["team"]["abbreviation"]
                is_home = team_data["homeAway"] == "home"
                opponent = next(
                    (c["team"]["abbreviation"] for c in comp["competitors"] if c != team_data),
                    None
                )
                
                schedule[team_abbr] = {
                    "game_id": event["id"],
                    "game_time": game_time,
                    "status": comp["status"]["type"]["description"],
                    "is_home": is_home,
                    "opponent": opponent,
                    "venue": team_data.get("team", {}).get("venue", {}).get("fullName"),
                }
        
        logger.info(f"Fetched {len(schedule)} teams from ESPN API")
        return schedule
    
    def get_team_game_status(self, team: str, date: Optional[datetime] = None) -> Optional[Dict]:
        """Get game status for a specific team."""
        schedule = self.get_todays_schedule(date)
        return schedule.get(team)


class LineupValidator:
    """
    Validates fantasy lineups against real-world MLB schedules.
    
    Prevents game_id mismatch errors by ensuring all starters
    have actual games today, and suggests valid replacements.
    """
    
    # Yahoo team abbreviations -> MLB team abbreviations
    YAHOO_TO_MLB = {
        "NYY": "NYY", "LAD": "LAD", "BOS": "BOS", "CHC": "CHC",
        "STL": "STL", "SFG": "SF", "NYM": "NYM", "HOU": "HOU",
        "ATL": "ATL", "PHI": "PHI", "TOR": "TOR", "SEA": "SEA",
        "TBR": "TB", "MIL": "MIL", "SDP": "SD", "CLE": "CLE",
        "MIN": "MIN", "DET": "DET", "CHW": "CWS", "LAA": "LAA",
        "ARI": "AZ", "TEX": "TEX", "BAL": "BAL", "PIT": "PIT",
        "CIN": "CIN", "COL": "COL", "KCR": "KC", "MIA": "MIA",
        "OAK": "OAK", "WSN": "WSH",
    }
    
    def __init__(self, schedule_fetcher: Optional[ScheduleFetcher] = None):
        self.schedule = schedule_fetcher or ScheduleFetcher()
    
    def validate_lineup(
        self,
        optimized_slots: List[OptimizedSlot],
        roster_players: List[Dict],  # Full roster with bench
        current_time: Optional[datetime] = None,
        strict: bool = False
    ) -> LineupValidation:
        """
        Validate that all starters have games today.
        
        Args:
            optimized_slots: Slots from the optimizer
            roster_players: All players on roster (starters + bench)
            current_time: Time to check against (default: now)
            strict: If True, fail on any player without a game
            
        Returns:
            LineupValidation with errors, warnings, and suggestions
        """
        if current_time is None:
            current_time = datetime.now()
        
        todays_schedule = self.schedule.get_todays_schedule(current_time)
        validation = LineupValidation(valid=True)
        
        for slot in optimized_slots:
            if not slot.player_id:
                continue
            
            # Find player in roster
            player = self._find_player(roster_players, slot.player_id)
            if not player:
                validation.errors.append(
                    f"Slot {slot.position}: Player {slot.player_id} not found on roster"
                )
                validation.valid = False
                continue
            
            # Check game status
            game_info = self._get_player_game_status(player, todays_schedule)
            
            if game_info.status == GameStatus.NO_GAME:
                validation.invalid_players.append(game_info)
                validation.warnings.append(
                    f"{game_info.player_name} ({slot.position}): No game today"
                )
                if strict:
                    validation.valid = False
                    
            elif game_info.status == GameStatus.GAME_STARTED:
                validation.warnings.append(
                    f"{game_info.player_name}: Game already started - changes may not apply"
                )
                
            elif game_info.status == GameStatus.POSTPONED:
                validation.warnings.append(
                    f"{game_info.player_name}: Game postponed - cannot start"
                )
                if strict:
                    validation.valid = False
                    
            elif game_info.status == GameStatus.UNKNOWN:
                validation.warnings.append(
                    f"{game_info.player_name}: Cannot verify game status"
                )
        
        # Generate suggestions for invalid players
        if validation.invalid_players:
            validation.suggestions = self._generate_suggestions(
                validation.invalid_players,
                roster_players,
                optimized_slots,
                todays_schedule
            )
        
        return validation
    
    def auto_correct_lineup(
        self,
        optimized_slots: List[OptimizedSlot],
        roster_players: List[Dict],
        current_time: Optional[datetime] = None
    ) -> LineupSubmission:
        """
        Automatically fix lineup by replacing players with no games.
        
        Args:
            optimized_slots: Original optimized slots
            roster_players: Full roster
            current_time: Time context
            
        Returns:
            LineupSubmission with corrected assignments
        """
        if current_time is None:
            current_time = datetime.now()
        
        todays_schedule = self.schedule.get_todays_schedule(current_time)
        
        assignments = {}
        bench_assignments = {}
        changes_made = []
        invalid_players = []
        
        # First pass: identify invalid starters
        for slot in optimized_slots:
            if not slot.player_id:
                continue
            
            player = self._find_player(roster_players, slot.player_id)
            if not player:
                invalid_players.append((slot, None))
                continue
            
            game_info = self._get_player_game_status(player, todays_schedule)
            
            if game_info.status in (GameStatus.NO_GAME, GameStatus.POSTPONED):
                invalid_players.append((slot, player))
            else:
                assignments[slot.slot_id] = slot.player_id
        
        # Second pass: find replacements
        for slot, invalid_player in invalid_players:
            replacement = self._find_replacement(
                slot,
                invalid_player,
                roster_players,
                assignments.values(),  # Already used players
                todays_schedule
            )
            
            if replacement:
                assignments[slot.slot_id] = replacement["player_id"]
                bench_assignments[invalid_player["player_id"]] = invalid_player["player_id"] if invalid_player else None
                
                old_name = invalid_player["name"] if invalid_player else slot.player_name or "Unknown"
                changes_made.append(
                    f"{slot.position}: {old_name} (no game) → {replacement['name']} "
                    f"(vs {replacement['opponent']}, {replacement['game_time'].strftime('%H:%M')})"
                )
            else:
                # No valid replacement found
                if invalid_player:
                    changes_made.append(
                        f"{slot.position}: {invalid_player['name']} (no game) → BENCH (no eligible replacement)"
                    )
        
        # Validate the corrected lineup
        validation = self.validate_lineup(
            [OptimizedSlot(sid, pos, pid, name) for sid, pid, pos, name in [
                (s.slot_id, assignments.get(s.slot_id), s.position, s.player_name) 
                for s in optimized_slots
            ]],
            roster_players,
            current_time,
            strict=False
        )
        
        return LineupSubmission(
            assignments=assignments,
            bench_assignments=bench_assignments,
            changes_made=changes_made,
            validation=validation
        )
    
    def _find_player(self, roster: List[Dict], player_id: str) -> Optional[Dict]:
        """Find player in roster by ID."""
        for p in roster:
            if str(p.get("player_id")) == str(player_id) or str(p.get("id")) == str(player_id):
                return p
        return None
    
    def _get_player_game_status(
        self, 
        player: Dict, 
        schedule: Dict
    ) -> PlayerGameInfo:
        """Determine a player's game status for today."""
        player_id = str(player.get("player_id") or player.get("id", ""))
        player_name = player.get("name", "Unknown")
        
        # Get team
        team = player.get("team", player.get("editorial_team_abbr", ""))
        if not team:
            return PlayerGameInfo(
                player_id=player_id,
                player_name=player_name,
                team="",
                status=GameStatus.UNKNOWN
            )
        
        # Map Yahoo team to MLB team
        mlb_team = self.YAHOO_TO_MLB.get(team.upper(), team.upper())
        
        # Look up game
        game = schedule.get(mlb_team)
        if not game:
            return PlayerGameInfo(
                player_id=player_id,
                player_name=player_name,
                team=team,
                status=GameStatus.NO_GAME
            )
        
        # Determine status
        game_time = game.get("game_time")
        status_str = game.get("status", "").lower()
        
        if "postponed" in status_str:
            game_status = GameStatus.POSTPONED
        elif game_time and datetime.now() > game_time:
            game_status = GameStatus.GAME_STARTED
        elif game.get("double_header"):
            game_status = GameStatus.DOUBLE_HEADER
        else:
            game_status = GameStatus.HAS_GAME
        
        return PlayerGameInfo(
            player_id=player_id,
            player_name=player_name,
            team=team,
            opponent=game.get("opponent"),
            game_time=game_time,
            is_home=game.get("is_home", False),
            status=game_status,
            double_header_game=game.get("game_number"),
            venue=game.get("venue")
        )
    
    def _generate_suggestions(
        self,
        invalid_players: List[PlayerGameInfo],
        roster: List[Dict],
        slots: List[OptimizedSlot],
        schedule: Dict
    ) -> List[Dict]:
        """Generate replacement suggestions for invalid players."""
        suggestions = []
        
        # Get bench players with games
        bench_options = []
        for player in roster:
            player_id = str(player.get("player_id") or player.get("id", ""))
            
            # Skip players already in lineup
            if any(str(s.player_id) == player_id for s in slots):
                continue
            
            game_info = self._get_player_game_status(player, schedule)
            if game_info.status == GameStatus.HAS_GAME:
                bench_options.append({
                    "player": player,
                    "game_info": game_info
                })
        
        # Sort by game time (earlier = better for DFS)
        bench_options.sort(key=lambda x: x["game_info"].game_time or datetime.max)
        
        for invalid in invalid_players:
            # Find slot for this player
            slot = next(
                (s for s in slots if s.player_id == invalid.player_id),
                None
            )
            
            if not slot:
                continue
            
            # Find eligible replacements
            eligible = [
                opt for opt in bench_options
                if self._is_position_eligible(opt["player"], slot.position)
            ]
            
            if eligible:
                best = eligible[0]
                suggestions.append({
                    "action": "replace",
                    "slot": slot.position,
                    "remove": invalid.player_name,
                    "add": best["game_info"].player_name,
                    "reason": f"{invalid.player_name} has no game",
                    "game_info": {
                        "opponent": best["game_info"].opponent,
                        "time": best["game_info"].game_time.strftime("%H:%M") if best["game_info"].game_time else "TBD",
                        "venue": "Home" if best["game_info"].is_home else "Away"
                    }
                })
            else:
                suggestions.append({
                    "action": "no_replacement",
                    "slot": slot.position,
                    "player": invalid.player_name,
                    "reason": "No bench players with games available for this position"
                })
        
        return suggestions
    
    def _find_replacement(
        self,
        slot: OptimizedSlot,
        invalid_player: Optional[Dict],
        roster: List[Dict],
        used_player_ids: List[str],
        schedule: Dict
    ) -> Optional[Dict]:
        """Find the best replacement for an invalid player."""
        used_set = set(str(pid) for pid in used_player_ids)
        
        candidates = []
        for player in roster:
            player_id = str(player.get("player_id") or player.get("id", ""))
            
            if player_id in used_set:
                continue
            
            if not self._is_position_eligible(player, slot.position):
                continue
            
            game_info = self._get_player_game_status(player, schedule)
            if game_info.status != GameStatus.HAS_GAME:
                continue
            
            # Score: earlier game is better (more time to accumulate stats)
            score = game_info.game_time.timestamp() if game_info.game_time else float('inf')
            candidates.append({
                "player_id": player_id,
                "name": player.get("name", "Unknown"),
                "opponent": game_info.opponent,
                "game_time": game_info.game_time,
                "score": score
            })
        
        if not candidates:
            return None
        
        # Return earliest game
        candidates.sort(key=lambda x: x["score"])
        return candidates[0]
    
    def _is_position_eligible(self, player: Dict, slot_position: str) -> bool:
        """Check if player can fill a slot position."""
        eligible = player.get("eligible_positions", player.get("positions", []))
        
        # Normalize
        slot_pos = slot_position.upper()
        player_positions = set(p.upper() for p in eligible)
        
        if slot_pos in player_positions:
            return True
        
        # Utility slot accepts any hitter
        if slot_pos == "UTIL":
            return bool(player_positions.intersection({"1B", "2B", "3B", "SS", "C", "OF", "LF", "CF", "RF", "DH"}))
        
        # OF accepts LF/CF/RF
        if slot_pos == "OF" and player_positions.intersection({"LF", "CF", "RF"}):
            return True
        
        return False


def format_lineup_report(submission: LineupSubmission) -> str:
    """Format a LineupSubmission into a human-readable report."""
    lines = [
        "=" * 50,
        "LINEUP VALIDATION REPORT",
        "=" * 50,
        ""
    ]
    
    if submission.changes_made:
        lines.append("🔄 AUTOMATIC CORRECTIONS MADE:")
        for change in submission.changes_made:
            lines.append(f"  • {change}")
        lines.append("")
    
    lines.append(f"✓ Validation Status: {'PASS' if submission.validation.valid else 'FAIL'}")
    
    if submission.validation.errors:
        lines.append("\n❌ ERRORS:")
        for error in submission.validation.errors:
            lines.append(f"  • {error}")
    
    if submission.validation.warnings:
        lines.append("\n⚠️  WARNINGS:")
        for warning in submission.validation.warnings:
            lines.append(f"  • {warning}")
    
    if submission.validation.suggestions:
        lines.append("\n💡 SUGGESTIONS:")
        for suggestion in submission.validation.suggestions:
            action = suggestion.get("action", "unknown")
            if action == "replace":
                lines.append(
                    f"  • {suggestion['slot']}: Replace {suggestion['remove']} with "
                    f"{suggestion['add']} ({suggestion['game_info']['venue']} vs "
                    f"{suggestion['game_info']['opponent']}, {suggestion['game_info']['time']})"
                )
            else:
                lines.append(f"  • {suggestion.get('reason', 'No suggestion')}")
    
    lines.append("")
    lines.append(f"📊 Final Lineup: {len(submission.assignments)} players assigned")
    
    return "\n".join(lines)
