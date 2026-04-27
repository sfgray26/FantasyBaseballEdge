"""
Position normalization for Yahoo Fantasy roster management.

Handles position eligibility mismatches between projection sources (Steamer)
and Yahoo's roster requirements.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Simplified player representation."""
    id: str
    name: str
    positions: List[str]  # From projection source
    yahoo_positions: Optional[List[str]] = None  # Yahoo's view
    eligible_positions: Optional[List[str]] = None  # Valid Yahoo slots
    team: Optional[str] = None  # Team abbreviation


@dataclass
class RosterSlot:
    """Yahoo roster slot."""
    id: str
    position: str
    player_id: Optional[str] = None


@dataclass
class YahooRoster:
    """Yahoo roster structure."""
    slots: List[RosterSlot]
    players: List[Player]


class PositionNormalizer:
    """
    Normalize position eligibility between projection sources and Yahoo.
    
    Yahoo uses specific position codes that may differ from projection
    sources like Steamer. This class handles the mapping and validation.
    """
    
    # Map common position codes to Yahoo's expected format.
    # League 72586 uses LF/CF/RF — there is NO generic "OF" slot.
    # Projection sources that emit "OF" must be resolved to a specific
    # outfield position at lineup-solve time; the normalizer preserves the
    # original value so the solver can pick the best available slot.
    POSITION_MAP = {
        # Hitters
        "C": "C",
        "1B": "1B",
        "2B": "2B",
        "3B": "3B",
        "SS": "SS",
        "LF": "LF",
        "CF": "CF",
        "RF": "RF",
        "OF": "OF",  # Kept for eligibility checks; NOT a valid roster slot
        "DH": "Util",  # Designated hitter -> Utility slot
        "UTIL": "Util",
        # Pitchers
        "SP": "SP",
        "RP": "RP",
        "P": "P",  # Generic pitcher (Yahoo accepts P)
    }

    # Positions that can fill utility slot
    UTILITY_ELIGIBLE = {"1B", "2B", "3B", "SS", "C", "LF", "CF", "RF", "OF", "DH"}

    # Valid outfield slot positions in this league (no generic OF slot)
    OUTFIELD_POSITIONS = {"LF", "CF", "RF"}
    
    @classmethod
    def normalize_position(cls, position: str) -> str:
        """Normalize a position code to Yahoo format."""
        pos_upper = position.upper().strip()
        return cls.POSITION_MAP.get(pos_upper, pos_upper)
    
    @classmethod
    def normalize_player_positions(cls, player: Player) -> Player:
        """Normalize all positions for a player."""
        normalized = [cls.normalize_position(p) for p in player.positions]
        player.yahoo_positions = list(set(normalized))  # Deduplicate
        return player
    
    @classmethod
    def is_eligible_for_slot(cls, player: Player, slot_position: str) -> bool:
        """
        Check if player is eligible for a roster slot.
        
        Args:
            player: Player with yahoo_positions populated
            slot_position: Target slot position (e.g., "2B", "Util")
        """
        if not player.yahoo_positions:
            return False
        
        slot_pos = cls.normalize_position(slot_position)
        player_positions = set(player.yahoo_positions)
        
        # Direct match
        if slot_pos in player_positions:
            return True

        # Outfield flexibility: LF/CF/RF slots can be filled by any OF-eligible player.
        # A player tagged "OF" (generic) can fill LF, CF, or RF.
        # A player tagged LF can also fill CF/RF (Yahoo allows cross-OF).
        # An "OF" slot can be filled by any LF/CF/RF/OF player.
        if slot_pos in cls.OUTFIELD_POSITIONS or slot_pos == "OF":
            if player_positions.intersection(cls.OUTFIELD_POSITIONS | {"OF"}):
                return True

        # Utility slot: Any hitter can fill Util
        if slot_pos == "Util" and player_positions.intersection(cls.UTILITY_ELIGIBLE):
            return True
        
        # Pitcher flexibility: SP/RP can fill P slot
        if slot_pos == "P" and player_positions.intersection({"SP", "RP"}):
            return True
        
        return False
    
    @classmethod
    def normalize_lineup(
        cls,
        optimized_lineup: Dict[str, Any],
        yahoo_roster: YahooRoster,
        strict: bool = False
    ) -> Dict[str, str]:
        """
        Match optimized lineup to Yahoo's actual roster slots.
        
        Args:
            optimized_lineup: Dict with 'starters' list from optimizer
            yahoo_roster: Current Yahoo roster structure
            strict: If True, fail on any unmatched slot
            
        Returns:
            Dict mapping slot_id -> player_id
            
        Raises:
            LineupValidationError: If strict=True and validation fails
        """
        assignments = {}
        used_players: Set[str] = set()
        unmatched_slots = []
        
        for slot in yahoo_roster.slots:
            slot_pos = cls.normalize_position(slot.position)
            
            # Find best matching player from optimization
            matched = False
            for opt_player_data in optimized_lineup.get("starters", []):
                player_id = str(opt_player_data.get("id") or opt_player_data.get("player_id"))
                
                if player_id in used_players:
                    continue
                
                # Build Player object
                player = Player(
                    id=player_id,
                    name=opt_player_data.get("name", "Unknown"),
                    positions=opt_player_data.get("positions", []),
                )
                cls.normalize_player_positions(player)
                
                # Check eligibility
                if cls.is_eligible_for_slot(player, slot_pos):
                    assignments[slot.id] = player_id
                    used_players.add(player_id)
                    matched = True
                    break
            
            if not matched:
                unmatched_slots.append(slot_pos)
                logger.warning(f"No eligible player for slot {slot_pos}")
        
        if strict and unmatched_slots:
            raise LineupValidationError(
                f"Could not fill slots: {unmatched_slots}. "
                f"Available players may not have correct position eligibility."
            )
        
        return assignments
    
    @classmethod
    def validate_lineup_before_submit(
        cls,
        assignments: Dict[str, str],
        yahoo_roster: YahooRoster
    ) -> "ValidationResult":
        """
        Dry-run validation before calling Yahoo's set_lineup API.
        
        Returns ValidationResult with success status and detailed errors.
        """
        errors = []
        warnings = []
        
        # Build lookup for Yahoo roster
        slot_map = {s.id: s for s in yahoo_roster.slots}
        player_map = {p.id: p for p in yahoo_roster.players}
        
        for slot_id, player_id in assignments.items():
            # Validate slot exists
            if slot_id not in slot_map:
                errors.append(f"Invalid slot ID: {slot_id}")
                continue
            
            slot = slot_map[slot_id]
            slot_pos = cls.normalize_position(slot.position)
            
            # Validate player exists
            if player_id not in player_map:
                # Player might be from optimizer, not yet on roster
                warnings.append(
                    f"Player {player_id} not found in Yahoo roster (may be waiver add)"
                )
                continue
            
            player = player_map[player_id]
            
            # Validate position eligibility
            if player.yahoo_positions is None:
                cls.normalize_player_positions(player)
            
            if not cls.is_eligible_for_slot(player, slot_pos):
                errors.append(
                    f"Position mismatch: {player.name} ({player.yahoo_positions}) "
                    f"cannot play {slot_pos}"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            assignment_count=len(assignments),
        )


@dataclass
class ValidationResult:
    """Result of lineup validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    assignment_count: int
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        lines = [f"Lineup Validation: {status} ({self.assignment_count} assignments)"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class LineupValidationError(Exception):
    """Raised when lineup validation fails in strict mode."""
    pass


class PositionEligibilityLookup:
    """
    Cache and lookup position eligibility from Yahoo.
    
    Yahoo's position eligibility can change (e.g., due to games played
    at new positions). This caches the eligibility to reduce API calls.
    """
    
    def __init__(self):
        self._cache: Dict[str, List[str]] = {}
    
    def update_from_yahoo(self, yahoo_roster: YahooRoster):
        """Update cache from Yahoo roster data."""
        for player in yahoo_roster.players:
            if player.eligible_positions:
                self._cache[player.id] = player.eligible_positions
    
    def get_eligibility(self, player_id: str) -> Optional[List[str]]:
        """Get cached eligibility for a player."""
        return self._cache.get(player_id)
    
    def is_cached(self, player_id: str) -> bool:
        """Check if player eligibility is cached."""
        return player_id in self._cache
