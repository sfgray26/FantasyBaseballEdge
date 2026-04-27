"""
Live Draft Tracker — Treemendous League

Polls Yahoo Fantasy API for new draft picks and fires Discord notifications
via send_draft_pick(). Designed to run in a background thread or async loop
during a live draft.

Snake draft logic is delegated entirely to draft_engine.picks_for_position()
so the two modules share a single source of truth.

Usage:
    client = YahooFantasyClient()
    tracker = DraftTracker(client, my_draft_position=7)

    # In a polling loop:
    while not tracker.draft_complete:
        new_count = tracker.run_poll_once()
        time.sleep(5)

Manual / test mode (pass mock_results to bypass Yahoo):
    tracker.run_poll_once(mock_results=[...])
"""

import logging
import time
from typing import Optional

from backend.fantasy_baseball.draft_engine import (
    NUM_ROUNDS,
    NUM_TEAMS,
    picks_for_position,
)

logger = logging.getLogger(__name__)

# How many picks before our turn triggers the on-the-clock alert
ON_THE_CLOCK_THRESHOLD = 2


class DraftTracker:
    """
    Polls yahoo_client.get_draft_results() and sends Discord notifications
    for each new pick.

    Args:
        yahoo_client: A YahooFantasyClient instance (or any object with
                      get_draft_results() returning list[dict]).
        my_draft_position: 1-indexed draft slot (1-12).
        num_teams: Number of teams in the league (default 12).
        num_rounds: Total rounds (default 23).
    """

    def __init__(
        self,
        yahoo_client,
        my_draft_position: int,
        num_teams: int = NUM_TEAMS,
        num_rounds: int = NUM_ROUNDS,
    ):
        self.yahoo_client = yahoo_client
        self.my_draft_position = my_draft_position
        self.num_teams = num_teams
        self.num_rounds = num_rounds

        # Pre-build the set of overall pick numbers that belong to us.
        # This delegates snake logic entirely to draft_engine.
        self._my_pick_numbers: set[int] = {
            overall for overall, _rnd in picks_for_position(
                my_draft_position, num_teams, num_rounds
            )
        }

        # Ordered list of (overall_pick, round) tuples for sequential look-up
        self._my_pick_order: list[tuple[int, int]] = sorted(
            picks_for_position(my_draft_position, num_teams, num_rounds),
            key=lambda x: x[0],
        )

        self._last_pick_count: int = 0
        self._total_picks: int = num_teams * num_rounds

    # ------------------------------------------------------------------
    # Yahoo polling
    # ------------------------------------------------------------------

    def get_current_results(self) -> list:
        """
        Call yahoo_client.get_draft_results() with error isolation.
        Returns an empty list on any failure so the polling loop never crashes.
        """
        try:
            return self.yahoo_client.get_draft_results()
        except Exception as exc:
            logger.warning("Yahoo get_draft_results() failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Pick logic
    # ------------------------------------------------------------------

    def is_my_pick(self, overall_pick: int) -> bool:
        """
        Return True if overall_pick N belongs to our draft slot.

        Uses the pre-built set derived from draft_engine.picks_for_position(),
        which encodes the Treemendous snake rule (rounds 1-2 linear,
        round 3+ snake).
        """
        return overall_pick in self._my_pick_numbers

    def picks_until_my_turn(self, overall_pick: int) -> int:
        """
        Return how many picks stand between overall_pick N and our next turn.

        Returns 0 when overall_pick is exactly our pick, or when the draft
        is past all of our picks.
        """
        for my_pick, _rnd in self._my_pick_order:
            if my_pick >= overall_pick:
                return my_pick - overall_pick
        return 0

    @property
    def draft_complete(self) -> bool:
        return self._last_pick_count >= self._total_picks

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def format_pick_message(self, pick: dict, is_ours: bool) -> str:
        """
        Build a human-readable pick summary string.

        Format: 'Pick #N (Rd X): <team_key> selected <player_name> (<positions>)'

        pick dict keys (from get_draft_results()):
            pick        — overall pick number
            round       — round number
            team_key    — Yahoo team key of the picking team
            player_key  — Yahoo player key
            player_name — player name (may be absent if not enriched)
            positions   — list of position strings (may be absent)
        """
        overall = pick.get("pick", "?")
        rnd = pick.get("round", "?")
        team_key = pick.get("team_key", "unknown")
        player_name = pick.get("player_name") or pick.get("player_key", "Unknown Player")
        positions = pick.get("positions")
        pos_str = "/".join(positions) if positions else ""
        pos_part = f" ({pos_str})" if pos_str else ""

        prefix = "YOUR PICK - " if is_ours else ""
        return f"{prefix}Pick #{overall} (Rd {rnd}): {team_key} selected {player_name}{pos_part}"

    def process_new_picks(self, new_picks: list, total_picks_so_far: int) -> None:
        """
        For each new pick in new_picks, send a Discord notification and fire
        on-the-clock alerts when we are approaching our turn.

        Args:
            new_picks: Picks that arrived since the last poll (already sliced).
            total_picks_so_far: The overall pick number of the LAST pick already
                                processed before this batch.
        """
        # Import here to avoid circular imports and keep Yahoo/Discord decoupled
        from backend.services.discord_notifier import (
            send_draft_pick,
            send_on_the_clock_alert,
        )

        for i, pick in enumerate(new_picks):
            overall = pick.get("pick") or (total_picks_so_far + i + 1)
            is_ours = self.is_my_pick(overall)

            positions = pick.get("positions", [])
            player_name = pick.get("player_name") or pick.get("player_key", "Unknown")
            team_key = pick.get("team_key", "")
            rnd = pick.get("round", 0)

            try:
                send_draft_pick(
                    pick_number=overall,
                    round_number=rnd,
                    player_name=player_name,
                    positions=positions,
                    team_key=team_key,
                    is_our_pick=is_ours,
                )
            except Exception as exc:
                logger.warning("send_draft_pick failed for pick #%s: %s", overall, exc)

            # On-the-clock alert: warn when we are exactly ON_THE_CLOCK_THRESHOLD
            # picks away from our next turn, or when it is our pick
            next_pick_n = overall + 1  # the pick number that comes next
            gap = self.picks_until_my_turn(next_pick_n)
            if 0 < gap <= ON_THE_CLOCK_THRESHOLD:
                try:
                    send_on_the_clock_alert(picks_away=gap, top_recommendations=[])
                except Exception as exc:
                    logger.warning("send_on_the_clock_alert failed: %s", exc)

    def run_poll_once(self, mock_results: Optional[list] = None) -> int:
        """
        Execute one poll cycle.

        Fetches current draft results (or uses mock_results in test mode),
        detects new picks by comparing list length against the previous count,
        processes each new pick, and updates internal state.

        Args:
            mock_results: If provided, skips the Yahoo API call entirely.
                          Useful for unit tests and the manual fallback mode.

        Returns:
            Count of new picks found in this cycle.
        """
        results = mock_results if mock_results is not None else self.get_current_results()

        current_count = len(results)
        new_count = current_count - self._last_pick_count

        if new_count <= 0:
            return 0

        new_picks = results[self._last_pick_count:current_count]
        self.process_new_picks(new_picks, total_picks_so_far=self._last_pick_count)
        self._last_pick_count = current_count
        return new_count
