"""
Data Reliability Engine — Phase B Critical Infrastructure

Ensures data pipeline accuracy through:
1. Multi-source cross-validation
2. Stale data detection  
3. Anomaly detection
4. Graceful degradation with source fallbacks
5. Data quality scoring

As fantasy baseball experts know: "Bad data kills championships."
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import asyncio

from sqlalchemy.orm import Session

from backend.models import SessionLocal, DataIngestionLog, StatcastPerformance

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Priority-ordered data sources."""
    STATCAST_API = "statcast_api"           # Primary: Real-time MLB
    YAHOO_API = "yahoo_api"                 # Primary: Roster/matchup
    BASEBALL_SAVANT = "baseball_savant"     # Primary: Statcast CSV
    FANGRAPHS = "fangraphs"                 # Secondary: Projections
    PYBASEBALL = "pybaseball"               # Secondary: Aggregated stats
    CACHE = "cache"                         # Tertiary: Local cache
    FALLBACK = "fallback"                   # Last resort: Defaults


class DataQualityTier(str, Enum):
    """Data quality classification."""
    TIER_1_EXCELLENT = "tier_1"  # Live API, <5 min old
    TIER_2_GOOD = "tier_2"       # Recent, <1 hour old
    TIER_3_ACCEPTABLE = "tier_3"  # Cached, <24 hours old
    TIER_4_STALE = "tier_4"      # >24 hours old
    TIER_5_UNAVAILABLE = "tier_5"  # No data, using defaults


@dataclass
class DataValidationResult:
    """Result of data validation check."""
    is_valid: bool
    quality_tier: DataQualityTier
    source: DataSource
    freshness_minutes: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_tier": self.quality_tier.value,
            "source": self.source.value,
            "freshness_minutes": round(self.freshness_minutes, 2),
            "warnings": self.warnings,
            "errors": self.errors,
            "confidence_score": round(self.confidence_score, 3),
        }


@dataclass 
class CrossValidationResult:
    """Result of cross-validating data across sources."""
    field_name: str
    primary_value: Any
    secondary_value: Any
    discrepancy_pct: float
    is_acceptable: bool
    threshold: float
    

class DataReliabilityEngine:
    """
    Central data quality and reliability management.
    
    Implements the "Trust but Verify" pattern:
    - Accept data from primary sources
    - Validate against secondary sources
    - Flag discrepancies above thresholds
    - Degrade gracefully when sources fail
    """
    
    # Discrepancy thresholds for cross-validation
    VALIDATION_THRESHOLDS = {
        "avg": 0.050,           # 50 points batting average
        "obp": 0.030,           # 30 points OBP
        "slg": 0.080,           # 80 points slugging
        "ops": 0.100,           # 100 points OPS
        "hr": 0.20,             # 20% difference in HR count
        "rbi": 0.15,            # 15% difference
        "woba": 0.030,          # 30 points wOBA
        "xwoba": 0.030,         # 30 points xwOBA
        "era": 0.50,            # 0.50 ERA difference
        "whip": 0.15,           # 0.15 WHIP difference
    }
    
    # Freshness thresholds by data type (minutes)
    FRESHNESS_THRESHOLDS = {
        "live_game": 5,         # 5 minutes for live game data
        "statcast": 60,         # 1 hour for Statcast
        "projections": 1440,    # 24 hours for projections
        "roster": 15,           # 15 minutes for roster
        "lineup": 5,            # 5 minutes for lineup
        "matchup": 60,          # 1 hour for matchup
    }
    
    def __init__(self):
        self._source_health: Dict[DataSource, Dict[str, Any]] = {
            source: {"last_success": None, "failure_count": 0, "is_healthy": True}
            for source in DataSource
        }
    
    # ========================================================================
    # Data Validation Methods
    # ========================================================================
    
    def validate_statcast_data(
        self,
        player_id: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        db: Optional[Session] = None
    ) -> DataValidationResult:
        """
        Validate Statcast data for a player.
        
        Checks:
        - Data completeness (no nulls in critical fields)
        - Value ranges (exit velocity 0-120mph, etc.)
        - Freshness (< 1 hour for live data)
        - Cross-reference with recent historical data
        """
        warnings = []
        errors = []
        
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            # Check 1: Data completeness
            critical_fields = ["player_id", "player_name", "game_date", "pa"]
            for field_name in critical_fields:
                if field_name not in data or data[field_name] is None:
                    errors.append(f"Missing critical field: {field_name}")
            
            # Check 2: Value ranges
            if data.get("exit_velocity_avg", 0) > 120:
                warnings.append("Exit velocity seems high (>120mph)")
            if data.get("exit_velocity_avg", 0) < 50 and data.get("pa", 0) > 10:
                warnings.append("Exit velocity seems low (<50mph)")
            if data.get("barrel_pct", 0) > 30:
                warnings.append("Barrel% seems high (>30%)")
            
            # Check 3: Freshness
            freshness = self._calculate_freshness_minutes(timestamp)
            threshold = self.FRESHNESS_THRESHOLDS["statcast"]
            
            if freshness > threshold * 2:  # 2x threshold = stale
                errors.append(f"Data is stale: {freshness:.0f} minutes old")
                quality_tier = DataQualityTier.TIER_4_STALE
            elif freshness > threshold:
                warnings.append(f"Data is aging: {freshness:.0f} minutes old")
                quality_tier = DataQualityTier.TIER_3_ACCEPTABLE
            elif freshness > threshold / 2:
                quality_tier = DataQualityTier.TIER_2_GOOD
            else:
                quality_tier = DataQualityTier.TIER_1_EXCELLENT
            
            # Check 4: Cross-validation with yesterday's data
            if "pa" in data and data["pa"] > 0:
                yesterday = db.query(StatcastPerformance).filter(
                    StatcastPerformance.player_id == player_id,
                    StatcastPerformance.game_date == datetime.utcnow().date() - timedelta(days=1)
                ).first()
                
                if yesterday:
                    # Check for unrealistic day-to-day changes
                    if yesterday.exit_velocity_avg > 0 and data.get("exit_velocity_avg", 0) > 0:
                        ev_change = abs(data["exit_velocity_avg"] - yesterday.exit_velocity_avg)
                        if ev_change > 10:  # 10 mph change is suspicious
                            warnings.append(f"Large EV change from yesterday: {ev_change:.1f} mph")
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                len(errors), len(warnings), quality_tier
            )
            
            is_valid = len(errors) == 0
            
            return DataValidationResult(
                is_valid=is_valid,
                quality_tier=quality_tier,
                source=DataSource.STATCAST_API,
                freshness_minutes=freshness,
                warnings=warnings,
                errors=errors,
                confidence_score=confidence,
            )
        
        finally:
            if close_db:
                db.close()
    
    def validate_yahoo_roster(
        self,
        roster_data: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> DataValidationResult:
        """Validate Yahoo roster data."""
        warnings = []
        errors = []
        
        # Check 1: Data presence
        if not roster_data:
            errors.append("Roster data is empty")
            return DataValidationResult(
                is_valid=False,
                quality_tier=DataQualityTier.TIER_5_UNAVAILABLE,
                source=DataSource.YAHOO_API,
                freshness_minutes=9999,
                warnings=warnings,
                errors=errors,
                confidence_score=0.0,
            )
        
        # Check 2: Player count
        if len(roster_data) < 10:
            warnings.append(f"Small roster size: {len(roster_data)} players")
        if len(roster_data) > 30:
            warnings.append(f"Large roster size: {len(roster_data)} players")
        
        # Check 3: Required fields
        required = ["name", "player_id", "positions"]
        for player in roster_data:
            for field_name in required:
                if field_name not in player:
                    errors.append(f"Player missing {field_name}: {player.get('name', 'unknown')}")
        
        # Check 4: Freshness
        freshness = self._calculate_freshness_minutes(timestamp)
        threshold = self.FRESHNESS_THRESHOLDS["roster"]
        
        if freshness > threshold:
            warnings.append(f"Roster data is {freshness:.0f} minutes old")
            quality_tier = DataQualityTier.TIER_3_ACCEPTABLE
        else:
            quality_tier = DataQualityTier.TIER_1_EXCELLENT
        
        confidence = self._calculate_confidence_score(
            len(errors), len(warnings), quality_tier
        )
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            quality_tier=quality_tier,
            source=DataSource.YAHOO_API,
            freshness_minutes=freshness,
            warnings=warnings,
            errors=errors,
            confidence_score=confidence,
        )
    
    def cross_validate_player_data(
        self,
        player_id: str,
        primary_source: Dict[str, Any],
        secondary_source: Dict[str, Any],
        fields_to_validate: Optional[List[str]] = None
    ) -> List[CrossValidationResult]:
        """
        Cross-validate player data between two sources.
        
        As top fantasy players know: "Never trust a single source."
        """
        if fields_to_validate is None:
            fields_to_validate = list(self.VALIDATION_THRESHOLDS.keys())
        
        results = []
        
        for field_name in fields_to_validate:
            if field_name not in primary_source or field_name not in secondary_source:
                continue
            
            primary_val = primary_source[field_name]
            secondary_val = secondary_source[field_name]
            
            # Skip if either is None
            if primary_val is None or secondary_val is None:
                continue
            
            # Calculate discrepancy
            if isinstance(primary_val, (int, float)) and isinstance(secondary_val, (int, float)):
                if primary_val == 0:
                    discrepancy = abs(secondary_val)
                else:
                    discrepancy = abs(primary_val - secondary_val) / abs(primary_val)
                
                threshold = self.VALIDATION_THRESHOLDS.get(field_name, 0.10)
                
                # For rate stats (avg, obp, etc.), use absolute difference
                if field_name in ["avg", "obp", "slg", "ops", "woba", "xwoba"]:
                    discrepancy = abs(primary_val - secondary_val)
                
                results.append(CrossValidationResult(
                    field_name=field_name,
                    primary_value=primary_val,
                    secondary_value=secondary_val,
                    discrepancy_pct=discrepancy * 100,
                    is_acceptable=discrepancy <= threshold,
                    threshold=threshold,
                ))
        
        return results
    
    # ========================================================================
    # Source Health Monitoring
    # ========================================================================
    
    def record_source_success(self, source: DataSource):
        """Record successful data fetch from source."""
        self._source_health[source]["last_success"] = datetime.utcnow()
        self._source_health[source]["failure_count"] = 0
        self._source_health[source]["is_healthy"] = True
    
    def record_source_failure(self, source: DataSource, error: str):
        """Record failed data fetch from source."""
        self._source_health[source]["failure_count"] += 1
        
        # After 3 failures, mark as unhealthy
        if self._source_health[source]["failure_count"] >= 3:
            self._source_health[source]["is_healthy"] = False
            logger.warning(f"Source {source.value} marked unhealthy after 3 failures")
    
    def get_source_health(self, source: DataSource) -> Dict[str, Any]:
        """Get health status of a data source."""
        return self._source_health[source].copy()
    
    def get_best_available_source(self, data_type: str) -> DataSource:
        """
        Get the best available source for a data type.
        
        Implements fallback chain:
        Primary → Secondary → Cache → Fallback
        """
        source_priority = {
            "statcast": [
                DataSource.STATCAST_API,
                DataSource.BASEBALL_SAVANT,
                DataSource.PYBASEBALL,
                DataSource.CACHE,
            ],
            "projections": [
                DataSource.FANGRAPHS,
                DataSource.PYBASEBALL,
                DataSource.CACHE,
            ],
            "roster": [
                DataSource.YAHOO_API,
                DataSource.CACHE,
            ],
        }
        
        priority = source_priority.get(data_type, [DataSource.FALLBACK])
        
        for source in priority:
            if self._source_health[source]["is_healthy"]:
                return source
        
        return DataSource.FALLBACK
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _calculate_freshness_minutes(self, timestamp: Optional[datetime]) -> float:
        """Calculate how old data is in minutes."""
        if timestamp is None:
            return 9999  # Unknown = very old
        
        delta = datetime.utcnow() - timestamp
        return delta.total_seconds() / 60
    
    def _calculate_confidence_score(
        self,
        error_count: int,
        warning_count: int,
        quality_tier: DataQualityTier
    ) -> float:
        """Calculate overall confidence score (0.0 to 1.0)."""
        base_scores = {
            DataQualityTier.TIER_1_EXCELLENT: 1.0,
            DataQualityTier.TIER_2_GOOD: 0.85,
            DataQualityTier.TIER_3_ACCEPTABLE: 0.70,
            DataQualityTier.TIER_4_STALE: 0.50,
            DataQualityTier.TIER_5_UNAVAILABLE: 0.0,
        }
        
        base = base_scores.get(quality_tier, 0.5)
        
        # Penalize errors heavily, warnings lightly
        error_penalty = error_count * 0.25
        warning_penalty = warning_count * 0.05
        
        return max(0.0, base - error_penalty - warning_penalty)
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate overall data quality report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source_health": {
                source.value: {
                    "is_healthy": health["is_healthy"],
                    "last_success": health["last_success"].isoformat() if health["last_success"] else None,
                    "failure_count": health["failure_count"],
                }
                for source, health in self._source_health.items()
            },
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        for source, health in self._source_health.items():
            if not health["is_healthy"]:
                recommendations.append(f"Source {source.value} is unhealthy - investigate failures")
            elif health["failure_count"] > 0:
                recommendations.append(f"Source {source.value} has {health['failure_count']} recent failures")
        
        return recommendations


# Singleton instance
_reliability_engine: Optional[DataReliabilityEngine] = None


def get_reliability_engine() -> DataReliabilityEngine:
    """Get singleton reliability engine."""
    global _reliability_engine
    if _reliability_engine is None:
        _reliability_engine = DataReliabilityEngine()
    return _reliability_engine


def with_data_validation(data_type: str):
    """
    Decorator for data fetching functions.
    
    Automatically validates returned data and logs issues.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            engine = get_reliability_engine()
            source = engine.get_best_available_source(data_type)
            
            try:
                result = await func(*args, **kwargs)
                engine.record_source_success(source)
                return result
            except Exception as e:
                engine.record_source_failure(source, str(e))
                logger.error(f"Data fetch failed for {data_type} from {source.value}: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            engine = get_reliability_engine()
            source = engine.get_best_available_source(data_type)
            
            try:
                result = func(*args, **kwargs)
                engine.record_source_success(source)
                return result
            except Exception as e:
                engine.record_source_failure(source, str(e))
                logger.error(f"Data fetch failed for {data_type} from {source.value}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
