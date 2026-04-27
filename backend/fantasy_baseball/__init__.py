"""Fantasy Baseball module — Yahoo Fantasy integration, keeper evaluation, draft assistant."""

# Original exports (unified client — YahooFantasyClient + ResilientYahooClient in one module)
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient, YahooAuthError, YahooAPIError

# Resilience components (new)
from backend.fantasy_baseball.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.fantasy_baseball.cache_manager import StaleCacheManager, CacheResult, NoDataAvailableError
from backend.fantasy_baseball.position_normalizer import PositionNormalizer, LineupValidationError
from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient, WaiverResponse, LineupResult

# Game-aware lineup validator (new)
from backend.fantasy_baseball.lineup_validator import (
    LineupValidator,
    ScheduleFetcher,
    OptimizedSlot,
    GameStatus,
    PlayerGameInfo,
    LineupValidation,
    LineupSubmission,
    format_lineup_report,
)

# Platoon splits fetcher (new)
from backend.fantasy_baseball.platoon_fetcher import (
    PlatoonSplitFetcher,
    PlatoonSplits,
    get_platoon_fetcher,
)

# Category tracker (new)
from backend.fantasy_baseball.category_tracker import (
    CategoryTracker,
    MatchupStatus,
    get_category_tracker,
)

# Smart lineup selector (advanced optimizer)
from backend.fantasy_baseball.smart_lineup_selector import (
    SmartLineupSelector,
    SmartBatterRanking,
    OpposingPitcher,
    CategoryNeed,
    Handedness,
    get_smart_selector,
)

# Elite context (new)
from backend.fantasy_baseball.elite_context import (
    EliteManagerContextBuilder,
    PlayerDecisionContext,
    LineupDecisionReport,
    RiskProfile,
    MatchupStrategy,
    WeatherContext,
    RecentForm,
    LineupSpot,
)

# Pitcher deep dive (new)
from backend.fantasy_baseball.pitcher_deep_dive import (
    PitcherDeepDiveFetcher,
    get_pitcher_fetcher,
)

# Weather (new)
from backend.fantasy_baseball.weather_fetcher import (
    WeatherFetcher,
    GameWeather,
    get_weather_fetcher,
)

# Park weather analyzer (new)
from backend.fantasy_baseball.park_weather import (
    ParkWeatherAnalyzer,
    ParkWeatherProfile,
    StadiumOrientation,
    WindImpact,
    get_park_analyzer,
)

# Daily briefing (new)
from backend.fantasy_baseball.daily_briefing import (
    DailyBriefingGenerator,
    DailyBriefing,
    PlayerBriefing,
    CategoryBriefing,
    DecisionAction,
    get_briefing_generator,
)

# Decision tracking (new)
from backend.fantasy_baseball.decision_tracker import (
    DecisionTracker,
    PlayerDecision,
    DailyAccuracy,
    TrendReport,
    DecisionType,
    DecisionOutcome,
    get_decision_tracker,
)

# Alias for backward compatibility
YahooClient = YahooFantasyClient

__all__ = [
    # Original (with aliases for backward compatibility)
    "YahooFantasyClient",
    "YahooClient",  # Alias
    # Resilience components
    "CircuitBreaker",
    "CircuitOpenError",
    "StaleCacheManager",
    "CacheResult",
    "NoDataAvailableError",
    "PositionNormalizer",
    "LineupValidationError",
    "ResilientYahooClient",
    "WaiverResponse",
    "LineupResult",
    # Lineup validator (game-aware validation)
    "LineupValidator",
    "ScheduleFetcher",
    "OptimizedSlot",
    "GameStatus",
    "PlayerGameInfo",
    "LineupValidation",
    "LineupSubmission",
    "format_lineup_report",
    # Platoon fetcher
    "PlatoonSplitFetcher",
    "PlatoonSplits",
    "get_platoon_fetcher",
    # Category tracker
    "CategoryTracker",
    "MatchupStatus",
    "get_category_tracker",
    # Smart lineup selector
    "SmartLineupSelector",
    "SmartBatterRanking",
    "OpposingPitcher",
    "CategoryNeed",
    "Handedness",
    "get_smart_selector",
    # Pitcher deep dive
    "PitcherDeepDiveFetcher",
    "get_pitcher_fetcher",
    # Weather
    "WeatherFetcher",
    "GameWeather",
    "get_weather_fetcher",
    # Park weather analyzer
    "ParkWeatherAnalyzer",
    "ParkWeatherProfile",
    "StadiumOrientation",
    "WindImpact",
    "get_park_analyzer",
    # Daily briefing
    "DailyBriefingGenerator",
    "DailyBriefing",
    "PlayerBriefing",
    "CategoryBriefing",
    "DecisionAction",
    "get_briefing_generator",
    # Decision tracking
    "DecisionTracker",
    "PlayerDecision",
    "DailyAccuracy",
    "TrendReport",
    "DecisionType",
    "DecisionOutcome",
    "get_decision_tracker",
    # Elite context
    "EliteManagerContextBuilder",
    "PlayerDecisionContext",
    "LineupDecisionReport",
    "RiskProfile",
    "MatchupStrategy",
    "WeatherContext",
    "RecentForm",
    "LineupSpot",
]
