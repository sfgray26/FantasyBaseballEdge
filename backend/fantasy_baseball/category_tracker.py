"""
Category tracking from Yahoo matchup scoreboard.

Fetches current H2H matchup and calculates category needs/urgency.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.fantasy_baseball.smart_lineup_selector import CategoryNeed
from backend.utils.fantasy_stat_contract import BATTING_CATEGORIES, CATEGORY_NEED_STAT_MAP

logger = logging.getLogger(__name__)


YAHOO_STAT_MAP = dict(CATEGORY_NEED_STAT_MAP)


@dataclass
class MatchupStatus:
    """Current status of H2H matchup."""
    week: int
    is_my_matchup: bool
    my_team_key: str
    opponent_team_key: str
    my_stats: Dict[str, float]
    opp_stats: Dict[str, float]
    category_needs: List[CategoryNeed]
    
    @property
    def summary(self) -> str:
        """Human-readable matchup summary."""
        winning = sum(1 for c in self.category_needs if c.needed > 0)
        losing = sum(1 for c in self.category_needs if c.needed < 0)
        tied = sum(1 for c in self.category_needs if c.needed == 0)
        return f"Winning {winning}, Losing {losing}, Tied {tied}"


class CategoryTracker:
    """Track category needs from Yahoo matchup scoreboard."""
    
    def __init__(self, client: Optional[YahooFantasyClient] = None):
        self.client = client or YahooFantasyClient()
    
    def get_category_needs(self, week: Optional[int] = None) -> List[CategoryNeed]:
        """
        Fetch current matchup and calculate category needs.
        
        Returns list of CategoryNeed with urgency calculations.
        """
        matchup = self._get_my_matchup(week)
        if not matchup:
            logger.warning("Could not fetch matchup data")
            return []
        
        # Extract both teams separately from matchup["teams"]
        teams = matchup.get("teams", {})
        team_list = []
        count = int(teams.get("count", 0)) if isinstance(teams, dict) else 0
        for i in range(count):
            team_data = teams.get(str(i), {}).get("team", [])
            team_list.append(self._parse_team_stats(team_data))
        
        if len(team_list) != 2:
            logger.warning(f"Expected 2 teams in matchup, got {len(team_list)}")
            return []
        
        # Identify my team vs opponent
        my_team_key = self.client.get_my_team_key()
        if team_list[0].get("team_key") == my_team_key:
            my_stats = team_list[0].get("stats", {})
            opp_stats = team_list[1].get("stats", {})
        else:
            my_stats = team_list[1].get("stats", {})
            opp_stats = team_list[0].get("stats", {})
        
        return self._calculate_needs(my_stats, opp_stats)
    
    def _get_my_matchup(self, week: Optional[int] = None) -> Optional[Dict]:
        """Find my team's current matchup."""
        try:
            matchups = self.client.get_scoreboard(week)
            my_team_key = self.client.get_my_team_key()
            
            for matchup in matchups:
                teams = matchup.get("teams", {})
                count = int(teams.get("count", 0)) if isinstance(teams, dict) else 0
                for i in range(count):
                    team_data = teams.get(str(i), {}).get("team", [])
                    if isinstance(team_data, list):
                        for item in team_data:
                            if isinstance(item, dict) and item.get("team_key") == my_team_key:
                                return matchup
            
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch matchup: {e}")
            return None
    
    def _parse_team_stats(self, team_data: List) -> Dict:
        """Extract team key and stats from team data."""
        result = {"team_key": None, "stats": {}}
        
        if not isinstance(team_data, list):
            return result
        
        for item in team_data:
            if isinstance(item, dict):
                if "team_key" in item:
                    result["team_key"] = item["team_key"]
                if "team_stats" in item:
                    stats = item["team_stats"]
                    result["stats"] = self._parse_stats_block(stats)
        
        return result
    
    def _parse_stats_block(self, stats_block: Dict) -> Dict[str, float]:
        """Parse Yahoo stats block into category dict."""
        result = {}
        
        stats_list = stats_block.get("stats", [])
        for stat_entry in stats_list:
            if isinstance(stat_entry, dict):
                stat = stat_entry.get("stat", {})
                stat_id = str(stat.get("stat_id", ""))
                value = stat.get("value", "0")
                
                category = YAHOO_STAT_MAP.get(stat_id)
                if category and value:
                    try:
                        result[category] = float(value)
                    except (ValueError, TypeError):
                        pass
        
        return result
    
    def _calculate_needs(
        self, 
        my_stats: Dict[str, float], 
        opp_stats: Dict[str, float]
    ) -> List[CategoryNeed]:
        """Calculate category needs from current stats."""
        needs = []
        
        for category in BATTING_CATEGORIES:
            my_val = my_stats.get(category, 0.0)
            opp_val = opp_stats.get(category, 0.0)
            
            # For average-based cats (AVG, OPS), higher is better
            # For counting cats (R, HR, RBI, SB), higher is better
            # So if my_val > opp_val, I'm winning (needed is positive surplus)
            needed = my_val - opp_val
            
            needs.append(CategoryNeed(
                category=category,
                current=my_val,
                projected_opponent=opp_val,
                needed=needed
            ))
        
        return needs
    
    def get_matchup_status(self, week: Optional[int] = None) -> Optional[MatchupStatus]:
        """Get full matchup status with all metadata."""
        matchup = self._get_my_matchup(week)
        if not matchup:
            return None
        
        teams = matchup.get("teams", {})
        team_list = []
        count = int(teams.get("count", 0)) if isinstance(teams, dict) else 0
        for i in range(count):
            team_data = teams.get(str(i), {}).get("team", [])
            team_list.append(self._parse_team_stats(team_data))
        
        if len(team_list) != 2:
            return None
        
        my_team_key = self.client.get_my_team_key()
        if team_list[0].get("team_key") == my_team_key:
            my_stats = team_list[0].get("stats", {})
            opp_stats = team_list[1].get("stats", {})
            opp_key = team_list[1].get("team_key")
        else:
            my_stats = team_list[1].get("stats", {})
            opp_stats = team_list[0].get("stats", {})
            opp_key = team_list[0].get("team_key")
        
        needs = self._calculate_needs(my_stats, opp_stats)
        
        return MatchupStatus(
            week=matchup.get("week", 0),
            is_my_matchup=True,
            my_team_key=my_team_key,
            opponent_team_key=opp_key,
            my_stats=my_stats,
            opp_stats=opp_stats,
            category_needs=needs
        )


def get_category_tracker() -> CategoryTracker:
    """Factory function for CategoryTracker."""
    return CategoryTracker()
