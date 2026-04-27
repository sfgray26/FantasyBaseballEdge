"""
Stale cache manager for graceful API degradation.

When external APIs fail, serves cached data with metadata about freshness.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    data: Any
    timestamp: datetime
    key: str
    
    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            data=d["data"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            key=d["key"],
        )


@dataclass  
class CacheResult:
    """Result wrapper indicating data source and freshness."""
    data: Any
    fresh: bool
    source: str  # "api" or "cache"
    age_hours: Optional[float] = None
    errors: list = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class StaleCacheManager:
    """
    Manages fallback to cached data when APIs fail.
    
    Serve stale data up to max_age if fresh fetch fails.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/fantasy",
        max_age: timedelta = timedelta(hours=24),
        enabled: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        self.enabled = enabled
        
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        """Get filesystem path for cache key."""
        # Sanitize key for filesystem
        safe_key = "".join(c for c in key if c.isalnum() or c in "_-").rstrip()
        return self.cache_dir / f"{safe_key}.json"
    
    def write(self, key: str, data: Any, metadata: Optional[Dict] = None):
        """Write data to cache."""
        if not self.enabled:
            return
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": metadata or {},
        }
        
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump(entry, f, default=str)
            logger.debug(f"Cache write: {key} -> {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache for {key}: {e}")
    
    def read(self, key: str) -> Optional[CacheEntry]:
        """Read data from cache if it exists."""
        if not self.enabled:
            return None
            
        cache_path = self.get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path) as f:
                entry = json.load(f)
            
            timestamp = datetime.fromisoformat(entry["timestamp"])
            return CacheEntry(
                data=entry["data"],
                timestamp=timestamp,
                key=key,
            )
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None
    
    def is_fresh(self, entry: CacheEntry) -> bool:
        """Check if cache entry is within acceptable age."""
        age = datetime.now() - entry.timestamp
        return age <= self.max_age
    
    def get_age_hours(self, entry: CacheEntry) -> float:
        """Get age of cache entry in hours."""
        age = datetime.now() - entry.timestamp
        return age.total_seconds() / 3600
    
    async def get_with_fallback(
        self,
        key: str,
        fetch_func: Callable,
        *fetch_args,
        **fetch_kwargs
    ) -> CacheResult:
        """
        Try to fetch fresh data, fallback to cache on failure.
        
        Args:
            key: Cache key for this data
            fetch_func: Async function to fetch fresh data
            *fetch_args, **fetch_kwargs: Arguments for fetch_func
            
        Returns:
            CacheResult with data and metadata about source/freshness
        """
        # Attempt fresh fetch
        try:
            data = await fetch_func(*fetch_args, **fetch_kwargs)
            self.write(key, data)
            return CacheResult(
                data=data,
                fresh=True,
                source="api",
            )
        except Exception as e:
            logger.warning(f"Fetch failed for {key}: {e}. Checking cache...")
            
            # Try cache fallback
            cached = self.read(key)
            if cached and self.is_fresh(cached):
                return CacheResult(
                    data=cached.data,
                    fresh=False,
                    source="cache",
                    age_hours=self.get_age_hours(cached),
                    errors=[f"API failed: {str(e)}. Serving cached data."],
                )
            
            # No acceptable cache available
            raise NoDataAvailableError(
                f"API failed and no acceptable cache for {key}. "
                f"Last cached: {cached.timestamp if cached else 'never'}"
            ) from e
    
    def get_with_fallback_sync(
        self,
        key: str,
        fetch_func: Callable,
        *fetch_args,
        **fetch_kwargs
    ) -> CacheResult:
        """Synchronous version of get_with_fallback."""
        try:
            data = fetch_func(*fetch_args, **fetch_kwargs)
            self.write(key, data)
            return CacheResult(
                data=data,
                fresh=True,
                source="api",
            )
        except Exception as e:
            logger.warning(f"Fetch failed for {key}: {e}. Checking cache...")
            
            cached = self.read(key)
            if cached and self.is_fresh(cached):
                return CacheResult(
                    data=cached.data,
                    fresh=False,
                    source="cache",
                    age_hours=self.get_age_hours(cached),
                    errors=[f"API failed: {str(e)}. Serving cached data."],
                )
            
            raise NoDataAvailableError(
                f"API failed and no acceptable cache for {key}. "
                f"Last cached: {cached.timestamp if cached else 'never'}"
            ) from e
    
    def invalidate(self, key: str):
        """Remove a specific key from cache."""
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cache invalidated: {key}")
    
    def clear_all(self):
        """Clear all cached data."""
        if not self.cache_dir.exists():
            return
            
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info(f"Cache cleared: {self.cache_dir}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"entries": 0, "enabled": self.enabled}
        
        entries = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in entries)
        
        return {
            "entries": len(entries),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "max_age_hours": self.max_age.total_seconds() / 3600,
            "enabled": self.enabled,
        }


class NoDataAvailableError(Exception):
    """Raised when both API and cache are unavailable."""
    pass
