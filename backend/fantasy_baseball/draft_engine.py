"""
Live Draft Assistant — Treemendous League

Pick order (Treemendous keeper-format, no keepers set):
  Round 1: 1→12  (linear)
  Round 2: 1→12  (SAME order — not reversed)
  Round 3: 12→1  (snake begins here)
  Round 4: 1→12
  Round 5: 12→1
  ... alternating from round 3 onward

This gives position 12 three consecutive picks (12, 24, 25) and
position 1 the first two picks but no late-round snake cushion.

Usage:
    from backend.fantasy_baseball.draft_engine import DraftState, DraftRecommender
    state = DraftState(my_draft_position=7, num_teams=12, num_rounds=23)
    recs = DraftRecommender(state, player_board).recommend(top_n=5)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

NUM_TEAMS = 12
NUM_ROUNDS = 23

# Roster slots to fill
ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "LF": 1, "CF": 1, "RF": 1, "Util": 1,
    "SP": 2, "RP": 2, "P": 3,
    "BN": 5,
}
TOTAL_PICKS = NUM_ROUNDS  # picks per team

# Positions that satisfy each slot
SLOT_ELIGIBILITY = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "LF": ["LF", "OF"],
    "CF": ["CF", "OF"],
    "RF": ["RF", "OF"],
    "Util": ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH"],
    "SP": ["SP"],
    "RP": ["RP"],
    "P": ["SP", "RP"],
    "BN": ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "SP", "RP", "DH"],
}

# Category scarcity multipliers for need-based scoring
SCARCITY = {
    "NSB": 2.0,   # Stolen bases — very scarce
    "NSV": 2.0,   # Net saves — very scarce
    "C": 1.8,     # Catcher — position scarcity
    "SS": 1.3,    # Shortstop
}


# ---------------------------------------------------------------------------
# Pick order
# ---------------------------------------------------------------------------

def get_pick_order(round_num: int, num_teams: int = NUM_TEAMS) -> list[int]:
    """
    Return ordered list of draft positions (1-indexed) for a given round.

    Round 1: [1, 2, ..., 12]  — linear
    Round 2: [1, 2, ..., 12]  — same (no reversal)
    Round 3: [12, 11, ..., 1] — reversed (snake begins)
    Round 4: [1, 2, ..., 12]
    Round 5: [12, 11, ..., 1]
    ...
    """
    positions = list(range(1, num_teams + 1))
    if round_num <= 2:
        return positions                  # Rounds 1 and 2: always linear
    # From round 3 onward: odd-numbered rounds from pick perspective
    # Round 3 is the "1st snake round" — reverses
    snake_round = round_num - 2          # 1 for round 3, 2 for round 4, ...
    if snake_round % 2 == 1:             # Odd snake rounds: reverse
        return list(reversed(positions))
    return positions                      # Even snake rounds: normal


def build_full_pick_order(num_teams: int = NUM_TEAMS, num_rounds: int = NUM_ROUNDS) -> list[tuple[int, int, int]]:
    """
    Build the complete draft order as list of (overall_pick, round, position).
    overall_pick is 1-indexed.
    """
    order = []
    pick = 1
    for rnd in range(1, num_rounds + 1):
        positions = get_pick_order(rnd, num_teams)
        for pos in positions:
            order.append((pick, rnd, pos))
            pick += 1
    return order


def picks_for_position(draft_position: int, num_teams: int = NUM_TEAMS,
                        num_rounds: int = NUM_ROUNDS) -> list[tuple[int, int]]:
    """Return list of (overall_pick, round) for a given draft position."""
    full_order = build_full_pick_order(num_teams, num_rounds)
    return [(p, r) for p, r, pos in full_order if pos == draft_position]


# ---------------------------------------------------------------------------
# Draft state
# ---------------------------------------------------------------------------

@dataclass
class DraftedPlayer:
    overall_pick: int
    round_num: int
    draft_position: int
    player_id: str
    player_name: str
    team: str
    positions: list[str]
    player_type: str


@dataclass
class DraftState:
    """
    Tracks the live draft: picks made, my roster, available players.
    """
    my_draft_position: int
    num_teams: int = NUM_TEAMS
    num_rounds: int = NUM_ROUNDS
    picks_made: list[DraftedPlayer] = field(default_factory=list)
    my_roster: list[DraftedPlayer] = field(default_factory=list)

    def __post_init__(self):
        self._full_order = build_full_pick_order(self.num_teams, self.num_rounds)
        self._my_picks = picks_for_position(self.my_draft_position, self.num_teams, self.num_rounds)
        self._drafted_ids: set[str] = set()

    @property
    def current_overall_pick(self) -> int:
        return len(self.picks_made) + 1

    @property
    def current_round(self) -> int:
        if self.current_overall_pick > len(self._full_order):
            return self.num_rounds
        _, rnd, _ = self._full_order[self.current_overall_pick - 1]
        return rnd

    @property
    def current_draft_position(self) -> int:
        if self.current_overall_pick > len(self._full_order):
            return 0
        _, _, pos = self._full_order[self.current_overall_pick - 1]
        return pos

    @property
    def is_my_pick(self) -> bool:
        return self.current_draft_position == self.my_draft_position

    @property
    def drafted_player_ids(self) -> set[str]:
        return self._drafted_ids

    def next_my_pick(self) -> Optional[tuple[int, int]]:
        """Return (overall_pick, round) for my next upcoming pick."""
        current = self.current_overall_pick
        for pick_num, rnd in self._my_picks:
            if pick_num >= current:
                return (pick_num, rnd)
        return None

    def picks_until_my_turn(self) -> int:
        nxt = self.next_my_pick()
        if nxt is None:
            return 0
        return nxt[0] - self.current_overall_pick

    def log_pick(self, player_id: str, player_name: str, team: str,
                 positions: list[str], player_type: str) -> DraftedPlayer:
        """Record a pick (any team, including mine)."""
        _, rnd, pos = self._full_order[self.current_overall_pick - 1]
        dp = DraftedPlayer(
            overall_pick=self.current_overall_pick,
            round_num=rnd,
            draft_position=pos,
            player_id=player_id,
            player_name=player_name,
            team=team,
            positions=positions,
            player_type=player_type,
        )
        self.picks_made.append(dp)
        self._drafted_ids.add(player_id)
        if pos == self.my_draft_position:
            self.my_roster.append(dp)
        return dp

    def roster_positions_filled(self) -> dict[str, int]:
        """Count how many of each position I've drafted."""
        filled: dict[str, int] = {}
        for p in self.my_roster:
            for pos in p.positions:
                filled[pos] = filled.get(pos, 0) + 1
        return filled

    def roster_needs(self) -> dict[str, int]:
        """Return remaining positional needs (target - filled)."""
        filled = self.roster_positions_filled()
        needs: dict[str, int] = {}
        for slot, count in ROSTER_SLOTS.items():
            eligible_filled = sum(
                filled.get(pos, 0)
                for pos in SLOT_ELIGIBILITY.get(slot, [])
            )
            remaining = max(0, count - eligible_filled)
            if remaining > 0:
                needs[slot] = remaining
        return needs

    def picks_remaining(self) -> int:
        return self.num_rounds - len(self.my_roster)

    def draft_complete(self) -> bool:
        return len(self.picks_made) >= self.num_teams * self.num_rounds

    def summary(self) -> str:
        nxt = self.next_my_pick()
        nxt_str = f"Pick {nxt[0]} (Round {nxt[1]})" if nxt else "Draft complete"
        return (
            f"Overall pick {self.current_overall_pick} | "
            f"Round {self.current_round} | "
            f"Position on clock: {self.current_draft_position} | "
            f"{'>>> YOUR PICK <<<' if self.is_my_pick else f'Picks until your turn: {self.picks_until_my_turn()}'} | "
            f"Next my pick: {nxt_str}"
        )


# ---------------------------------------------------------------------------
# Category balance tracker
# ---------------------------------------------------------------------------

@dataclass
class CategoryBalance:
    """Tracks projected category contributions of my current roster."""
    r: float = 0; h: float = 0; hr: float = 0; rbi: float = 0
    k_bat: float = 0; tb: float = 0; avg_total: float = 0; ops_total: float = 0
    nsb: float = 0; pa_total: float = 0
    w: float = 0; l: float = 0; hr_pit: float = 0; k_pit: float = 0
    era_sum: float = 0; whip_sum: float = 0; k9_sum: float = 0
    qs: float = 0; nsv: float = 0; ip_total: float = 0
    sp_count: int = 0; rp_count: int = 0

    def avg(self) -> float:
        return self.avg_total / self.pa_total if self.pa_total > 0 else 0.0

    def ops(self) -> float:
        return self.ops_total / self.pa_total if self.pa_total > 0 else 0.0

    def era(self) -> float:
        return (self.era_sum * 9 / self.ip_total) if self.ip_total > 0 else 0.0

    def whip(self) -> float:
        return self.whip_sum / self.ip_total if self.ip_total > 0 else 0.0

    def k9(self) -> float:
        return (self.k9_sum * 9 / self.ip_total) if self.ip_total > 0 else 0.0

    def grade(self, cat: str) -> str:
        """Quick A/B/C/D/F grade for a category vs league average."""
        thresholds = {
            "R": [(110, "A"), (95, "B"), (80, "C"), (65, "D")],
            "H": [(175, "A"), (155, "B"), (140, "C"), (125, "D")],
            "HR": [(35, "A"), (28, "B"), (22, "C"), (16, "D")],
            "RBI": [(105, "A"), (90, "B"), (75, "C"), (60, "D")],
            "K_bat": [(90, "A"), (110, "B"), (130, "C"), (150, "D")],  # lower is better
            "TB": [(320, "A"), (280, "B"), (245, "C"), (210, "D")],
            "AVG": [(0.290, "A"), (0.275, "B"), (0.260, "C"), (0.245, "D")],
            "OPS": [(0.920, "A"), (0.870, "B"), (0.830, "C"), (0.790, "D")],
            "NSB": [(40, "A"), (25, "B"), (15, "C"), (8, "D")],
            "W": [(55, "A"), (45, "B"), (35, "C"), (25, "D")],
            "L": [(30, "A"), (38, "B"), (45, "C"), (52, "D")],  # lower is better
            "ERA": [(3.20, "A"), (3.60, "B"), (4.00, "C"), (4.40, "D")],  # lower
            "WHIP": [(1.08, "A"), (1.18, "B"), (1.28, "C"), (1.38, "D")],  # lower
            "K_pit": [(220, "A"), (185, "B"), (155, "C"), (125, "D")],
            "K9": [(10.0, "A"), (9.0, "B"), (8.0, "C"), (7.0, "D")],
            "QS": [(90, "A"), (75, "B"), (60, "C"), (45, "D")],
            "NSV": [(35, "A"), (25, "B"), (15, "C"), (8, "D")],
        }
        threshs = thresholds.get(cat, [])
        reverse = cat in ("K_bat", "L", "ERA", "WHIP")  # lower is better
        val = {
            "R": self.r, "H": self.h, "HR": self.hr, "RBI": self.rbi,
            "K_bat": self.k_bat, "TB": self.tb, "AVG": self.avg(),
            "OPS": self.ops(), "NSB": self.nsb, "W": self.w, "L": self.l,
            "ERA": self.era(), "WHIP": self.whip(), "K_pit": self.k_pit,
            "K9": self.k9(), "QS": self.qs, "NSV": self.nsv,
        }.get(cat, 0)

        for threshold, grade in threshs:
            if reverse:
                if val <= threshold:
                    return grade
            else:
                if val >= threshold:
                    return grade
        return "F"


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

@dataclass
class PickRecommendation:
    player_id: str
    player_name: str
    team: str
    positions: list[str]
    player_type: str
    overall_rank: int
    tier: int
    z_score: float
    adp: float
    adp_diff: float          # positive = value (available later than expected)
    need_boost: float        # positional/category need multiplier
    composite_score: float
    top_categories: list[str]
    rationale: str
    reach_alert: bool = False


class DraftRecommender:
    """
    Generates ranked pick recommendations for the current draft state.

    Scoring formula:
      composite = z_score × positional_need_boost × tier_urgency
      - z_score: projected category value
      - positional_need_boost: 1.0–2.0x based on how urgently we need this position
      - tier_urgency: small penalty if player is last in their tier (can wait)
    """

    def __init__(self, state: DraftState, player_board: list[dict]):
        self.state = state
        self.board = player_board  # list of player dicts from player_board module

    def recommend(self, top_n: int = 5) -> list[PickRecommendation]:
        needs = self.state.roster_needs()
        picks_left = self.state.picks_remaining()
        current_round = self.state.current_round

        # Determine the pick number we're actually targeting (my turn or current pick)
        if self.state.is_my_pick:
            target_pick = self.state.current_overall_pick
        else:
            nxt = self.state.next_my_pick()
            target_pick = nxt[0] if nxt else self.state.current_overall_pick

        # ── Bug fix 1: draft-slot pre-filtering ──────────────────────────────
        # Players with ADP well below target_pick are almost certainly gone.
        # Keep them if not yet picked (might be a steal) but apply likely-gone penalty.
        available = [
            p for p in self.board
            if p["id"] not in self.state.drafted_player_ids
        ]

        scored = []
        for p in available:
            z = p.get("z_score", 0.0)
            adp = p.get("adp", 999.0)
            tier = p.get("tier", 10)
            overall_rank = p.get("rank", 999)

            # Need-based boost
            need_boost = self._compute_need_boost(p, needs, picks_left, current_round)

            # ADP diff vs MY pick number (not overall clock pick)
            # Positive = available later than expected = value
            adp_diff = adp - target_pick
            # Cap value bonus at 1.5 so a round-16 player can't jump to round 1
            # purely because of ADP spread. Meaningful within ±20 picks only.
            adp_value_bonus = min(max(0, adp_diff) * 0.04, 1.5)

            # Reach penalty: taking significantly before ADP
            reach_penalty = max(0, -adp_diff - 5) * 0.05
            reach_alert = adp_diff < -8

            # ── Bug fix 1b: "likely gone" heavy penalty ────────────────────
            # If we haven't yet reached my turn and ADP is well before my pick,
            # this player will almost certainly be gone — discount heavily.
            if not self.state.is_my_pick and adp < target_pick - 2:
                picks_not_logged = target_pick - self.state.current_overall_pick  # noqa: F841
                # How far before my pick is this player expected to go?
                already_gone_by = target_pick - adp
                # Each additional pick before mine reduces survival probability
                gone_penalty = min(already_gone_by * 0.4, z * 0.8)
                composite = (z * need_boost) + adp_value_bonus - reach_penalty - gone_penalty
            else:
                composite = (z * need_boost) + adp_value_bonus - reach_penalty

            top_cats = sorted(
                p.get("cat_scores", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            top_cat_names = [c for c, _ in top_cats if _ > 0]

            recs = PickRecommendation(
                player_id=p["id"],
                player_name=p["name"],
                team=p["team"],
                positions=p["positions"],
                player_type=p["type"],
                overall_rank=overall_rank,
                tier=tier,
                z_score=z,
                adp=adp,
                adp_diff=adp_diff,
                need_boost=need_boost,
                composite_score=composite,
                top_categories=top_cat_names,
                rationale=self._build_rationale(p, need_boost, adp_diff, needs, current_round),
                reach_alert=reach_alert,
            )
            scored.append(recs)

        scored.sort(key=lambda r: r.composite_score, reverse=True)
        return scored[:top_n]

    def look_ahead(self) -> dict:
        """
        Returns intelligence about the picks between now and my next turn.

        Answers: "What will likely be gone before I pick? What should I
        target if those are gone? Who is a must-draft vs can-wait?"
        """
        if self.state.is_my_pick:
            return {"picks_away": 0, "likely_gone": [], "targets_if_gone": []}

        nxt = self.state.next_my_pick()
        if not nxt:
            return {"picks_away": 0, "likely_gone": [], "targets_if_gone": []}

        picks_away = self.state.picks_until_my_turn()
        my_pick_num = nxt[0]
        current_pick = self.state.current_overall_pick

        available = [
            p for p in self.board
            if p["id"] not in self.state.drafted_player_ids
        ]

        # Likely gone: players whose ADP falls between now and my pick
        likely_gone = [
            p for p in available
            if current_pick <= p["adp"] < my_pick_num
        ]
        likely_gone.sort(key=lambda p: p["adp"])

        # Likely still available: players with ADP >= my_pick_num
        likely_avail = [
            p for p in available
            if p["adp"] >= my_pick_num - 3
        ]
        likely_avail.sort(key=lambda p: p.get("z_score", 0), reverse=True)

        # Best targets at my actual pick number
        top_targets = likely_avail[:8]

        # Tier breaks: are there players in a different tier who might slip?
        sleepers = [
            p for p in available
            if p["adp"] > my_pick_num + 5  # ADP says they go later
            and p.get("z_score", 0) > 3.0  # but have real value
        ][:3]

        return {
            "picks_away": picks_away,
            "my_next_pick": my_pick_num,
            "my_next_round": nxt[1],
            "likely_gone": likely_gone[:10],
            "top_targets_at_my_pick": top_targets[:5],
            "potential_sleepers": sleepers,
        }

    def _compute_need_boost(self, player: dict, needs: dict[str, int],
                             picks_left: int, current_round: int) -> float:
        """
        Scale the need boost based on:
        - How urgently we need this position
        - How many picks remain to fill it
        - Positional scarcity
        """
        positions = player.get("positions", [])
        boost = 1.0

        # Check if we have unfilled mandatory slots this player can fill
        for slot, remaining in needs.items():
            if remaining <= 0:
                continue
            eligible = SLOT_ELIGIBILITY.get(slot, [])
            can_fill = any(pos in eligible for pos in positions)
            if not can_fill:
                continue

            # Urgency: how many picks before we likely can't get this position?
            if slot == "C":
                boost = max(boost, 1.8)   # Catcher is scarce — high urgency
            elif slot in ("SS", "2B"):
                boost = max(boost, 1.3)
            elif slot in ("SP", "RP"):
                if current_round >= 4:
                    boost = max(boost, 1.2)  # Need pitching urgency in mid-rounds

        # Stolen base scarcity boost
        nsb = player.get("proj", {}).get("nsb", 0)
        if nsb > 30 and picks_left < 12:
            boost = max(boost, 1.6)   # Running out of picks, SB scarce

        # ── Bug fix 2: NSV/closer boost — round threshold ────────────────────
        # Never boost closers in rounds 1-4. Closers at ADP 23 (Clase) should
        # NOT outrank Freeman/Ramirez at pick 6. Start boosting round 5+;
        # escalate urgency round 9+ when closer supply dries up.
        nsv = player.get("proj", {}).get("nsv", 0)
        closer_slots = needs.get("RP", 0)
        if nsv > 20 and closer_slots > 0:
            if current_round >= 9:
                boost = max(boost, 1.8)   # Late-draft closers are gold
            elif current_round >= 6:
                boost = max(boost, 1.4)
            elif current_round >= 5:
                boost = max(boost, 1.2)
            # rounds 1-4: no boost — don't sacrifice elite bat for a closer

        return boost

    @staticmethod
    def _build_rationale(player: dict, need_boost: float, adp_diff: float,
                          needs: dict, round_num: int) -> str:
        parts = []
        proj = player.get("proj", {})
        positions = player.get("positions", [])
        ptype = player.get("type", "batter")

        # Value indicator
        if adp_diff > 15:
            parts.append(f"Strong value — available ~{adp_diff:.0f} picks after expected ADP")
        elif adp_diff > 5:
            parts.append(f"Slight value — ADP suggests round {int(player.get('adp', 0) / 12) + 1}")
        elif adp_diff < -8:
            parts.append(f"REACH — drafting {abs(adp_diff):.0f} picks early; confirm talent warrants it")

        # Key category contributions
        if ptype == "batter":
            highlights = []
            if proj.get("hr", 0) >= 30:
                highlights.append(f"{proj['hr']:.0f} HR")
            if proj.get("nsb", 0) >= 20:
                highlights.append(f"{proj['nsb']:.0f} NSB (scarce!)")
            if proj.get("ops", 0) >= 0.900:
                highlights.append(f".{int(proj['ops']*1000)} OPS")
            if proj.get("avg", 0) >= 0.290:
                highlights.append(f".{int(proj['avg']*1000)} AVG")
            if proj.get("k_bat", 0) / max(proj.get("pa", 1), 1) > 0.25:
                highlights.append(f"K% {proj.get('k_bat', 0)/max(proj.get('pa', 1), 1):.0%} hurts")
            if highlights:
                parts.append("Projects: " + ", ".join(highlights))
        else:
            highlights = []
            if proj.get("nsv", 0) >= 25:
                highlights.append(f"{proj['nsv']:.0f} NSV (closer!)")
            if proj.get("k9", 0) >= 10.0:
                highlights.append(f"{proj['k9']:.1f} K/9")
            if proj.get("era", 5.0) < 3.20:
                highlights.append(f"{proj['era']:.2f} ERA")
            if proj.get("qs", 0) >= 20:
                highlights.append(f"{proj['qs']:.0f} QS")
            if proj.get("l", 0) >= 10:
                highlights.append(f"{proj['l']:.0f} projected L (hurts)")
            if highlights:
                parts.append("Projects: " + ", ".join(highlights))

        # Position scarcity note
        if "C" in positions:
            parts.append("Only catcher worth rostering — take early or miss out")
        if need_boost >= 1.6:
            parts.append("HIGH NEED — prioritizing scarce position/category")

        return " · ".join(parts) if parts else "Solid value pick at this range"
