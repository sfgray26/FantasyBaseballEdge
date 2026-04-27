"""
ELITE LINEUP CONSTRAINT SOLVER — Integer Linear Programming Optimization

Solves the daily lineup problem as a constraint satisfaction problem:
- Fill 9 batter slots: C, 1B, 2B, 3B, SS, OF×3, Util
- Respect position eligibility
- Maximize total fantasy value
- Handle multi-position players optimally

Uses OR-Tools CP-SAT solver for optimal solutions in milliseconds.

As elite players know: It's not just picking the 9 best players—it's about
scarcity (C/SS are rare) and flexibility (multi-eligible players are gold).

Usage:
    from backend.fantasy_baseball.lineup_constraint_solver import LineupConstraintSolver
    solver = LineupConstraintSolver()
    optimal_lineup = solver.solve(roster, scores, eligibility)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logging.warning("OR-Tools not available, falling back to greedy solver")

from backend.fantasy_baseball.elite_lineup_scorer import EliteScore

logger = logging.getLogger(__name__)


class PositionSlot(str, Enum):
    """Yahoo H2H standard batting slots."""
    CATCHER = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SHORTSTOP = "SS"
    OUTFIELD_1 = "OF1"
    OUTFIELD_2 = "OF2"
    OUTFIELD_3 = "OF3"
    UTILITY = "Util"
    BENCH = "BN"


@dataclass
class PlayerSlotAssignment:
    """Assignment of player to slot."""
    player_id: str
    player_name: str
    slot: PositionSlot
    score: float
    eligibility: List[str]
    reason: str


@dataclass
class OptimizedLineup:
    """Complete optimized lineup result."""
    assignments: List[PlayerSlotAssignment]
    total_score: float
    is_optimal: bool
    solver_type: str
    unassigned_players: List[str]


class LineupConstraintSolver:
    """
    Elite constraint solver for daily lineup optimization.
    
    Uses OR-Tools CP-SAT for true optimization (not greedy approximation).
    Falls back to scarcity-first greedy if OR-Tools unavailable.
    """
    
    # Slot configuration: (slot, eligible_positions, scarcity_rank)
    # Scarcity: lower = fill first (C and SS are scarcest)
    SLOT_CONFIG = [
        (PositionSlot.CATCHER, ["C"], 1),
        (PositionSlot.SHORTSTOP, ["SS"], 2),
        (PositionSlot.SECOND_BASE, ["2B"], 3),
        (PositionSlot.THIRD_BASE, ["3B"], 4),
        (PositionSlot.FIRST_BASE, ["1B"], 5),
        (PositionSlot.OUTFIELD_1, ["OF", "LF", "CF", "RF"], 6),
        (PositionSlot.OUTFIELD_2, ["OF", "LF", "CF", "RF"], 7),
        (PositionSlot.OUTFIELD_3, ["OF", "LF", "CF", "RF"], 8),
        (PositionSlot.UTILITY, ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"], 9),
    ]
    
    def __init__(self):
        self.use_ortools = ORTOOLS_AVAILABLE
    
    def solve(
        self,
        players: List[Dict],
        player_scores: Dict[str, EliteScore],
        eligibility: Dict[str, List[str]],
        locked_slots: Optional[Dict[PositionSlot, str]] = None,
    ) -> OptimizedLineup:
        """
        Solve for optimal lineup.
        
        Args:
            players: List of player dicts with 'player_id', 'name', etc.
            player_scores: Dict mapping player_id to EliteScore
            eligibility: Dict mapping player_id to eligible positions
            locked_slots: Optional forced assignments {slot: player_id}
        
        Returns:
            OptimizedLineup with assignments
        """
        if self.use_ortools:
            return self._solve_ilp(players, player_scores, eligibility, locked_slots)
        else:
            return self._solve_greedy(players, player_scores, eligibility, locked_slots)
    
    def _solve_ilp(
        self,
        players: List[Dict],
        player_scores: Dict[str, EliteScore],
        eligibility: Dict[str, List[str]],
        locked_slots: Optional[Dict[PositionSlot, str]],
    ) -> OptimizedLineup:
        """
        Solve using Integer Linear Programming (OR-Tools CP-SAT).
        
        This finds the mathematically optimal solution, not an approximation.
        """
        model = cp_model.CpModel()
        
        player_ids = [p["player_id"] for p in players]
        slots = [slot for slot, _, _ in self.SLOT_CONFIG]
        
        # Decision variables: x[player][slot] = 0 or 1
        x = {}
        for pid in player_ids:
            for slot in slots:
                x[(pid, slot)] = model.NewBoolVar(f"x_{pid}_{slot.value}")
        
        # Constraint 1: Each slot filled by exactly 1 player
        for slot in slots:
            model.Add(sum(x[(pid, slot)] for pid in player_ids) == 1)
        
        # Constraint 2: Each player in at most 1 slot
        for pid in player_ids:
            model.Add(sum(x[(pid, slot)] for slot in slots) <= 1)
        
        # Constraint 3: Player must be eligible for assigned slot
        for pid in player_ids:
            for slot, eligible_positions, _ in self.SLOT_CONFIG:
                player_eligible = eligibility.get(pid, [])
                # Check if player is eligible for this slot
                can_fill = any(
                    pos in player_eligible or pos in ["LF", "CF", "RF"]
                    for pos in eligible_positions
                )
                if not can_fill:
                    model.Add(x[(pid, slot)] == 0)
        
        # Constraint 4: Locked slots (user overrides)
        if locked_slots:
            for slot, pid in locked_slots.items():
                model.Add(x[(pid, slot)] == 1)
        
        # Objective: Maximize total score
        # Scale scores to integers (CP-SAT works with ints)
        objective_terms = []
        for pid in player_ids:
            score = player_scores.get(pid)
            if score:
                # Scale by 1000 to preserve 3 decimal places
                scaled_score = int(score.total_score * 1000)
                for slot in slots:
                    objective_terms.append(scaled_score * x[(pid, slot)])
        
        model.Maximize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # 5 second timeout
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        # Extract solution
        assignments = []
        assigned_players = set()
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for slot, _, _ in self.SLOT_CONFIG:
                for pid in player_ids:
                    if solver.Value(x[(pid, slot)]) == 1:
                        score = player_scores.get(pid)
                        player = next((p for p in players if p["player_id"] == pid), {})
                        
                        assignments.append(PlayerSlotAssignment(
                            player_id=pid,
                            player_name=player.get("name", "Unknown"),
                            slot=slot,
                            score=score.total_score if score else 0.0,
                            eligibility=eligibility.get(pid, []),
                            reason=score.reasoning if score else "",
                        ))
                        assigned_players.add(pid)
                        break
        
        unassigned = [p["player_id"] for p in players if p["player_id"] not in assigned_players]
        total_score = sum(a.score for a in assignments)
        
        return OptimizedLineup(
            assignments=assignments,
            total_score=round(total_score, 3),
            is_optimal=(status == cp_model.OPTIMAL),
            solver_type="OR-Tools CP-SAT",
            unassigned_players=unassigned,
        )
    
    def _solve_greedy(
        self,
        players: List[Dict],
        player_scores: Dict[str, EliteScore],
        eligibility: Dict[str, List[str]],
        locked_slots: Optional[Dict[PositionSlot, str]],
    ) -> OptimizedLineup:
        """
        Fallback greedy solver (scarcity-first approach).
        
        Not mathematically optimal, but fast and doesn't require OR-Tools.
        """
        assignments = []
        assigned_players: Set[str] = set()
        
        # Handle locked slots first
        if locked_slots:
            for slot, pid in locked_slots.items():
                player = next((p for p in players if p["player_id"] == pid), None)
                if player:
                    score = player_scores.get(pid)
                    assignments.append(PlayerSlotAssignment(
                        player_id=pid,
                        player_name=player.get("name", "Unknown"),
                        slot=slot,
                        score=score.total_score if score else 0.0,
                        eligibility=eligibility.get(pid, []),
                        reason=score.reasoning if score else "User locked",
                    ))
                    assigned_players.add(pid)
        
        # Fill remaining slots by scarcity
        for slot, eligible_positions, _ in self.SLOT_CONFIG:
            if slot in (locked_slots or {}):
                continue  # Already assigned
            
            # Find eligible, unassigned players
            candidates = []
            for player in players:
                pid = player["player_id"]
                if pid in assigned_players:
                    continue
                
                player_eligible = eligibility.get(pid, [])
                can_fill = any(
                    pos in player_eligible or pos in ["LF", "CF", "RF"]
                    for pos in eligible_positions
                )
                
                if can_fill:
                    score = player_scores.get(pid)
                    candidates.append((pid, player, score.total_score if score else 0.0))
            
            if candidates:
                # Pick highest scorer
                best = max(candidates, key=lambda x: x[2])
                pid, player, score_val = best
                
                score = player_scores.get(pid)
                assignments.append(PlayerSlotAssignment(
                    player_id=pid,
                    player_name=player.get("name", "Unknown"),
                    slot=slot,
                    score=score_val,
                    eligibility=eligibility.get(pid, []),
                    reason=score.reasoning if score else "",
                ))
                assigned_players.add(pid)
            else:
                # Empty slot
                assignments.append(PlayerSlotAssignment(
                    player_id="",
                    player_name="EMPTY",
                    slot=slot,
                    score=0.0,
                    eligibility=[],
                    reason=f"No eligible player for {slot.value}",
                ))
        
        unassigned = [p["player_id"] for p in players if p["player_id"] not in assigned_players]
        total_score = sum(a.score for a in assignments if a.player_id)
        
        return OptimizedLineup(
            assignments=assignments,
            total_score=round(total_score, 3),
            is_optimal=False,  # Greedy is not guaranteed optimal
            solver_type="Greedy (Scarcity-First)",
            unassigned_players=unassigned,
        )
    
    def analyze_scarcity(
        self,
        roster: List[Dict],
        eligibility: Dict[str, List[str]],
    ) -> Dict[str, any]:
        """
        Analyze roster scarcity by position.
        
        Helps users understand positional strengths/weaknesses.
        """
        position_counts = {}
        
        for pos in ["C", "1B", "2B", "3B", "SS", "OF"]:
            eligible = [
                p for p in roster
                if pos in eligibility.get(p.get("player_id"), [])
            ]
            position_counts[pos] = {
                "count": len(eligible),
                "players": [p.get("name") for p in eligible[:3]],  # Top 3
                "is_scarce": len(eligible) <= 1,
            }
        
        # Identify multi-eligible (flexible) players
        multi_eligible = [
            {
                "name": p.get("name"),
                "positions": eligibility.get(p.get("player_id"), []),
                "count": len(eligibility.get(p.get("player_id"), [])),
            }
            for p in roster
            if len(eligibility.get(p.get("player_id", []))) >= 3
        ]
        multi_eligible.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "position_depth": position_counts,
            "multi_eligible_players": multi_eligible[:5],
            "scarcity_warnings": [
                f"{pos} is scarce (only {data['count']} eligible)"
                for pos, data in position_counts.items()
                if data["is_scarce"]
            ],
        }
    
    def suggest_lineup_improvements(
        self,
        current_lineup: OptimizedLineup,
        waiver_players: List[Dict],
        waiver_scores: Dict[str, EliteScore],
    ) -> List[Dict]:
        """
        Suggest waiver adds that would improve the lineup.
        
        Elite insight: Don't just suggest "good players"—suggest players
        who fill specific gaps or upgrade weak positions.
        """
        suggestions = []
        
        # Find weakest slots
        slot_scores = {a.slot: a.score for a in current_lineup.assignments}
        avg_score = sum(slot_scores.values()) / len(slot_scores) if slot_scores else 0
        
        weak_slots = [
            slot for slot, score in slot_scores.items()
            if score < avg_score * 0.9  # 10% below average
        ]
        
        # Check waiver for upgrades
        for player in waiver_players:
            pid = player.get("player_id")
            score = waiver_scores.get(pid)
            if not score:
                continue
            
            eligibility = player.get("positions", [])
            
            # Check if this player upgrades any weak slot
            for slot in weak_slots:
                slot_eligible = next(
                    (e for s, e, _ in self.SLOT_CONFIG if s == slot),
                    []
                )
                can_fill = any(pos in slot_eligible for pos in eligibility)
                
                if can_fill and score.total_score > slot_scores.get(slot, 0):
                    suggestions.append({
                        "player": player.get("name"),
                        "position": slot.value,
                        "current_score": slot_scores.get(slot, 0),
                        "upgrade_score": score.total_score,
                        "improvement": round(score.total_score - slot_scores.get(slot, 0), 3),
                        "reason": f"Upgrades {slot.value} by {score.total_score - slot_scores.get(slot, 0):.2f} pts",
                    })
        
        # Sort by improvement magnitude
        suggestions.sort(key=lambda x: x["improvement"], reverse=True)
        return suggestions[:5]


# Singleton
_solver: Optional[LineupConstraintSolver] = None


def get_lineup_solver() -> LineupConstraintSolver:
    """Get singleton solver."""
    global _solver
    if _solver is None:
        _solver = LineupConstraintSolver()
    return _solver
