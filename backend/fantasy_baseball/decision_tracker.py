"""
Decision tracking and accuracy reporting for fantasy baseball.

Tracks every recommendation vs what the user actually did,
then reports on how well the system performed.
"""

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DECISIONS_FILE = DATA_DIR / "decisions.jsonl"


class DecisionType(Enum):
    START = "start"
    BENCH = "bench"


class DecisionOutcome(Enum):
    """What actually happened."""
    PENDING = "pending"      # Game hasn't happened yet
    SUCCESS = "success"      # Recommendation was right
    FAILURE = "failure"      # Recommendation was wrong
    PARTIAL = "partial"      # Mixed results
    CANCELLED = "cancelled"  # Game postponed


@dataclass
class PlayerDecision:
    """A single lineup decision."""
    # IDs
    decision_id: str
    date: str  # YYYY-MM-DD
    player_name: str
    player_id: str
    team: str
    
    # The recommendation
    recommended_action: str  # "START" | "BENCH"
    confidence: int  # 0-100
    factors: List[str]  # Why this recommendation
    
    # Context at decision time
    opponent: str
    opposing_pitcher: Optional[str]
    venue: str
    weather_factor: float  # HR factor
    projected_stats: Dict[str, float]
    
    # What user did
    user_action: Optional[str] = None  # "OVERRIDE" if different from recommendation
    override_reason: Optional[str] = None
    
    # Results
    actual_stats: Optional[Dict[str, float]] = None
    outcome: str = "pending"
    accuracy_score: Optional[float] = None  # 0-1 how right we were
    
    # Meta
    created_at: str = None
    resolved_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class DailyAccuracy:
    """Accuracy stats for a single day."""
    date: str
    total_decisions: int
    followed_recommendations: int
    overrides: int
    
    # Accuracy breakdown
    correct_predictions: int
    incorrect_predictions: int
    
    # By confidence tier
    high_conf_accuracy: float  # 80-100%
    med_conf_accuracy: float   # 60-79%
    low_conf_accuracy: float   # 0-59%
    
    # User override success
    override_better_count: int  # User was right to override
    override_worse_count: int   # User should have listened
    
    # By category
    start_success_rate: float
    bench_success_rate: float


@dataclass
class TrendReport:
    """Trends over time."""
    period_days: int
    start_date: str
    end_date: str
    
    overall_accuracy: float
    trend_direction: str  # "improving", "declining", "stable"
    
    best_confidence_threshold: int  # e.g., 75 - recommendations above this are most accurate
    
    # Insights
    overvalued_factors: List[str]  # Factors that don't predict well
    undervalued_factors: List[str]  # Factors we should weight more
    
    # Recommendations
    suggested_adjustments: List[str]


class DecisionTracker:
    """Track all lineup decisions."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_file = DECISIONS_FILE
    
    def record_decision(self, decision: PlayerDecision) -> None:
        """Record a new decision."""
        try:
            with open(self.decisions_file, "a") as f:
                f.write(json.dumps(asdict(decision)) + "\n")
            logger.debug(f"Recorded decision {decision.decision_id} for {decision.player_name}")
        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
    
    def record_override(
        self,
        decision_id: str,
        user_action: str,
        reason: Optional[str] = None
    ) -> None:
        """Record that user overrode a recommendation."""
        decision = self._find_decision(decision_id)
        if not decision:
            logger.warning(f"Decision {decision_id} not found for override")
            return
        
        decision.user_action = user_action
        decision.override_reason = reason
        self._update_decision(decision)
    
    def resolve_decision(
        self,
        decision_id: str,
        actual_stats: Dict[str, float],
        game_happened: bool = True
    ) -> None:
        """
        Record what actually happened.
        
        actual_stats: {"hr": 1, "r": 2, "rbi": 3, "avg": 0.333}
        """
        decision = self._find_decision(decision_id)
        if not decision:
            logger.warning(f"Decision {decision_id} not found for resolution")
            return
        
        decision.actual_stats = actual_stats
        decision.resolved_at = datetime.now().isoformat()
        
        if not game_happened:
            decision.outcome = "cancelled"
            decision.accuracy_score = None
        else:
            decision.accuracy_score = self._calculate_accuracy(decision, actual_stats)
            decision.outcome = "success" if decision.accuracy_score > 0.6 else "failure"
        
        self._update_decision(decision)
    
    def get_daily_accuracy(self, date: str) -> Optional[DailyAccuracy]:
        """Get accuracy report for a specific day."""
        decisions = self._load_decisions_for_date(date)
        if not decisions:
            return None
        
        resolved = [d for d in decisions if d.outcome != "pending"]
        if not resolved:
            return None
        
        # Calculate metrics
        total = len(decisions)
        followed = sum(1 for d in decisions if d.user_action is None)
        overrides = total - followed
        
        correct = sum(1 for d in resolved if d.outcome == "success")
        incorrect = len(resolved) - correct
        
        # By confidence
        high_conf = [d for d in resolved if d.confidence >= 80]
        med_conf = [d for d in resolved if 60 <= d.confidence < 80]
        low_conf = [d for d in resolved if d.confidence < 60]
        
        # By action
        start_recs = [d for d in resolved if d.recommended_action == "START"]
        bench_recs = [d for d in resolved if d.recommended_action == "BENCH"]
        
        return DailyAccuracy(
            date=date,
            total_decisions=total,
            followed_recommendations=followed,
            overrides=overrides,
            correct_predictions=correct,
            incorrect_predictions=incorrect,
            high_conf_accuracy=self._accuracy_of(high_conf),
            med_conf_accuracy=self._accuracy_of(med_conf),
            low_conf_accuracy=self._accuracy_of(low_conf),
            override_better_count=0,  # TODO: Compare user vs system
            override_worse_count=0,
            start_success_rate=self._accuracy_of(start_recs),
            bench_success_rate=self._accuracy_of(bench_recs),
        )
    
    def get_trend_report(self, days: int = 14) -> TrendReport:
        """Get trend analysis over N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load all decisions in range
        all_decisions = []
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            all_decisions.extend(self._load_decisions_for_date(date))
        
        resolved = [d for d in all_decisions if d.outcome != "pending"]
        if not resolved:
            return TrendReport(
                period_days=days,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                overall_accuracy=0.0,
                trend_direction="insufficient_data",
                best_confidence_threshold=75,
                overvalued_factors=[],
                undervalued_factors=[],
                suggested_adjustments=["Need more data to generate insights"],
            )
        
        # Overall accuracy
        correct = sum(1 for d in resolved if d.outcome == "success")
        overall_acc = correct / len(resolved)
        
        # Find best confidence threshold
        best_threshold = 75
        best_acc = 0.0
        for threshold in range(50, 95, 5):
            above = [d for d in resolved if d.confidence >= threshold]
            if above:
                acc = self._accuracy_of(above)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
        
        # Analyze factors
        factor_performance = self._analyze_factors(resolved)
        overvalued = [f for f, score in factor_performance.items() if score < 0.5]
        undervalued = [f for f, score in factor_performance.items() if score > 0.7]
        
        # Generate suggestions
        suggestions = []
        if overall_acc < 0.55:
            suggestions.append("System accuracy below 55% - review scoring weights")
        if best_threshold > 80:
            suggestions.append(f"Only trust recommendations with {best_threshold}% confidence")
        if overvalued:
            suggestions.append(f"Consider reducing weight on: {', '.join(overvalued[:3])}")
        
        # Trend direction
        if days >= 7:
            mid_point = len(resolved) // 2
            first_half = resolved[:mid_point]
            second_half = resolved[mid_point:]
            first_acc = self._accuracy_of(first_half)
            second_acc = self._accuracy_of(second_half)
            
            if second_acc > first_acc + 0.1:
                trend = "improving"
            elif second_acc < first_acc - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return TrendReport(
            period_days=days,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            overall_accuracy=round(overall_acc, 2),
            trend_direction=trend,
            best_confidence_threshold=best_threshold,
            overvalued_factors=overvalued[:5],
            undervalued_factors=undervalued[:5],
            suggested_adjustments=suggestions,
        )
    
    def _find_decision(self, decision_id: str) -> Optional[PlayerDecision]:
        """Find a decision by ID."""
        if not self.decisions_file.exists():
            return None
        
        try:
            with open(self.decisions_file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("decision_id") == decision_id:
                        return PlayerDecision(**data)
        except Exception as e:
            logger.error(f"Error reading decisions: {e}")
        
        return None
    
    def _update_decision(self, updated: PlayerDecision) -> None:
        """Update a decision in place."""
        if not self.decisions_file.exists():
            return
        
        lines = []
        try:
            with open(self.decisions_file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("decision_id") == updated.decision_id:
                        lines.append(json.dumps(asdict(updated)))
                    else:
                        lines.append(line.strip())
            
            with open(self.decisions_file, "w") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            logger.error(f"Error updating decision: {e}")
    
    def _load_decisions_for_date(self, date: str) -> List[PlayerDecision]:
        """Load all decisions for a date."""
        decisions = []
        if not self.decisions_file.exists():
            return decisions
        
        try:
            with open(self.decisions_file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("date") == date:
                        decisions.append(PlayerDecision(**data))
        except Exception as e:
            logger.error(f"Error loading decisions: {e}")
        
        return decisions
    
    def _calculate_accuracy(
        self, 
        decision: PlayerDecision, 
        actual: Dict[str, float]
    ) -> float:
        """
        Calculate how accurate the recommendation was.
        
        If recommended START: higher stats = higher accuracy
        If recommended BENCH: lower stats = higher accuracy
        """
        proj = decision.projected_stats
        
        # Calculate fantasy points (simplified)
        proj_fp = proj.get("hr", 0) * 4 + proj.get("r", 0) + proj.get("rbi", 0) + proj.get("sb", 0) * 2
        actual_fp = actual.get("hr", 0) * 4 + actual.get("r", 0) + actual.get("rbi", 0) + actual.get("sb", 0) * 2
        
        if decision.recommended_action == "START":
            # Recommended start: want high actual FP
            # Score based on how close actual was to or above projection
            if proj_fp > 0:
                ratio = min(actual_fp / proj_fp, 2.0)  # Cap at 2x
                return min(ratio, 1.0)  # Cap at 1.0
            else:
                return 0.5 if actual_fp > 0 else 0.0
        else:
            # Recommended bench: want low actual FP
            # If actual FP is low, we were right
            if proj_fp > 0:
                # Lower actual is better for bench recommendation
                return max(0, 1.0 - (actual_fp / proj_fp))
            else:
                return 1.0 if actual_fp == 0 else 0.0
    
    def _accuracy_of(self, decisions: List[PlayerDecision]) -> float:
        """Calculate accuracy for a list of decisions."""
        if not decisions:
            return 0.0
        
        resolved = [d for d in decisions if d.outcome not in ["pending", "cancelled"]]
        if not resolved:
            return 0.0
        
        correct = sum(1 for d in resolved if d.outcome == "success")
        return round(correct / len(resolved), 2)
    
    def _analyze_factors(self, decisions: List[PlayerDecision]) -> Dict[str, float]:
        """Analyze which factors predict success."""
        factor_stats: Dict[str, List[float]] = {}
        
        for d in decisions:
            if d.outcome == "pending":
                continue
            
            # Get accuracy (use binary for simplicity)
            acc = 1.0 if d.outcome == "success" else 0.0
            
            # Record for each factor
            for factor in d.factors:
                if factor not in factor_stats:
                    factor_stats[factor] = []
                factor_stats[factor].append(acc)
        
        # Calculate average accuracy per factor
        return {
            factor: sum(scores) / len(scores)
            for factor, scores in factor_stats.items()
            if len(scores) >= 3  # Need minimum sample
        }


def get_decision_tracker() -> DecisionTracker:
    """Factory function."""
    return DecisionTracker()
