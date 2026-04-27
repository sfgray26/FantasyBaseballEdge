"""
Circuit breaker pattern for external API resilience.

Prevents cascading failures by opening the circuit after threshold failures,
then periodically attempting reset (half-open) before fully closing again.
"""

import time
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for external API calls.
    
    State transitions:
        CLOSED -> OPEN: After FAILURE_THRESHOLD failures
        OPEN -> HALF_OPEN: After RECOVERY_TIMEOUT seconds
        HALF_OPEN -> CLOSED: On success
        HALF_OPEN -> OPEN: On failure
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 300,  # 5 minutes
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._success_count = 0
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def failure_count(self) -> int:
        return self._failure_count
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute func with circuit breaker protection.
        
        Raises CircuitOpenError if circuit is open.
        """
        self._update_state()
        
        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN. Last failure: {self._last_failure_time}"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            # Count ALL exception types toward the failure threshold, not just
            # self.expected_exception.  This ensures the circuit opens even when
            # the wrapped function raises an unexpected error type.
            # Re-raise the original exception unconditionally so callers see the
            # real failure; subsequent calls against an OPEN circuit will receive
            # CircuitOpenError via the guard at the top of this method.
            self._on_failure()
            raise

    async def call_async(self, async_func: Callable, *args, **kwargs) -> Any:
        """Async version of call()."""
        self._update_state()

        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN. Last failure: {self._last_failure_time}"
            )

        try:
            result = await async_func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise
    
    def _update_state(self):
        """Check if we should transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit '{self.name}': OPEN -> HALF_OPEN (testing recovery)")
                self._state = CircuitState.HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if not self._last_failure_time:
            return False
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            # Require 2 consecutive successes to close
            if self._success_count >= 2:
                logger.info(f"Circuit '{self.name}': HALF_OPEN -> CLOSED (recovered)")
                self._reset()
        else:
            # In CLOSED state, just track success
            pass
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery test
            logger.warning(f"Circuit '{self.name}': HALF_OPEN -> OPEN (recovery failed)")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            # Hit threshold, open circuit
            logger.error(
                f"Circuit '{self.name}': CLOSED -> OPEN after {self._failure_count} failures"
            )
            self._state = CircuitState.OPEN
    
    def _reset(self):
        """Reset to initial CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
    
    def force_open(self):
        """Manually open the circuit (for testing or emergency)."""
        logger.warning(f"Circuit '{self.name}': Forced OPEN")
        self._state = CircuitState.OPEN
        self._last_failure_time = datetime.now()
    
    def force_close(self):
        """Manually close the circuit (after fixing issue)."""
        logger.info(f"Circuit '{self.name}': Forced CLOSED")
        self._reset()
    
    def get_stats(self) -> dict:
        """Return current circuit statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "recovery_timeout": self.recovery_timeout,
            "failure_threshold": self.failure_threshold,
        }
