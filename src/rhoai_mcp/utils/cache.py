"""Optional caching utilities for reducing repeated fetches.

This module provides a TTL-based caching decorator that can be applied
to client methods to cache responses and reduce Kubernetes API calls.
Caching is disabled by default and must be explicitly enabled via config.
"""

import time
from collections.abc import Callable
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

from rhoai_mcp.config import get_config

F = TypeVar("F", bound=Callable[..., Any])

# Thread-safe cache storage
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = Lock()


def _make_cache_key(prefix: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Generate a cache key from function arguments.

    Args:
        prefix: Key prefix (typically function name).
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Cache key string.
    """
    # Skip 'self' argument if present (first positional arg for methods)
    key_parts = [prefix]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)


def cached(key_prefix: str | None = None) -> Callable[[F], F]:
    """TTL-based caching decorator for client methods.

    The cache is only active when config.enable_response_caching is True.
    Cache entries expire after config.cache_ttl_seconds.

    Args:
        key_prefix: Optional key prefix. Defaults to function name.

    Returns:
        Decorated function with caching.

    Example:
        @cached("list_workbenches")
        def list_workbenches(self, namespace: str) -> list[Workbench]:
            ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_config()

            # If caching is disabled, just call the function
            if not config.enable_response_caching:
                return fn(*args, **kwargs)

            # Generate cache key
            prefix = key_prefix or fn.__name__
            cache_key = _make_cache_key(prefix, args, kwargs)

            # Check cache
            with _cache_lock:
                if cache_key in _cache:
                    cached_time, cached_value = _cache[cache_key]
                    if time.time() - cached_time < config.cache_ttl_seconds:
                        return cached_value
                    # Expired, remove it
                    del _cache[cache_key]

            # Call the function
            result = fn(*args, **kwargs)

            # Store in cache
            with _cache_lock:
                _cache[cache_key] = (time.time(), result)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def clear_cache() -> int:
    """Clear all cached entries.

    Returns:
        Number of entries cleared.
    """
    with _cache_lock:
        count = len(_cache)
        _cache.clear()
        return count


def clear_expired() -> int:
    """Clear only expired cache entries.

    Returns:
        Number of entries cleared.
    """
    config = get_config()
    now = time.time()
    cleared = 0

    with _cache_lock:
        expired_keys = [
            key
            for key, (cached_time, _) in _cache.items()
            if now - cached_time >= config.cache_ttl_seconds
        ]
        for key in expired_keys:
            del _cache[key]
            cleared += 1

    return cleared


def cache_stats() -> dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dict with cache statistics.
    """
    config = get_config()
    now = time.time()

    with _cache_lock:
        total = len(_cache)
        expired = sum(
            1 for cached_time, _ in _cache.values() if now - cached_time >= config.cache_ttl_seconds
        )

    return {
        "total_entries": total,
        "expired_entries": expired,
        "active_entries": total - expired,
        "caching_enabled": config.enable_response_caching,
        "ttl_seconds": config.cache_ttl_seconds,
    }


def invalidate(pattern: str) -> int:
    """Invalidate cache entries matching a pattern.

    Args:
        pattern: Pattern to match (simple substring match).

    Returns:
        Number of entries invalidated.
    """
    with _cache_lock:
        keys_to_remove = [key for key in _cache if pattern in key]
        for key in keys_to_remove:
            del _cache[key]
        return len(keys_to_remove)
