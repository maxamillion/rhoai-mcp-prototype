"""Tests for caching utilities."""

import time
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.utils.cache import (
    cache_stats,
    cached,
    clear_cache,
    clear_expired,
    invalidate,
    _cache,
)


@pytest.fixture(autouse=True)
def clear_cache_before_each() -> None:
    """Clear cache before each test."""
    clear_cache()


class TestCachedDecorator:
    """Tests for the cached decorator."""

    def test_caching_disabled_by_default(self) -> None:
        """Test that caching is disabled when config says so."""
        call_count = 0

        @cached("test")
        def test_func(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"

        # With caching disabled, function should be called each time
        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=False,
                cache_ttl_seconds=30,
            )

            result1 = test_func("a")
            result2 = test_func("a")

            assert result1 == "result-a"
            assert result2 == "result-a"
            assert call_count == 2

    def test_caching_enabled(self) -> None:
        """Test that caching works when enabled."""
        call_count = 0

        @cached("test")
        def test_func(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=30,
            )

            result1 = test_func("a")
            result2 = test_func("a")

            assert result1 == "result-a"
            assert result2 == "result-a"
            assert call_count == 1  # Only called once due to caching

    def test_different_args_different_cache_keys(self) -> None:
        """Test that different arguments create different cache entries."""
        call_count = 0

        @cached("test")
        def test_func(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=30,
            )

            result1 = test_func("a")
            result2 = test_func("b")
            result3 = test_func("a")  # Should be cached

            assert result1 == "result-a"
            assert result2 == "result-b"
            assert result3 == "result-a"
            assert call_count == 2  # Only a and b, second a is cached

    def test_cache_expiration(self) -> None:
        """Test that cache entries expire after TTL."""
        call_count = 0

        @cached("test")
        def test_func(arg: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=1,  # 1 second TTL
            )

            result1 = test_func("a")
            time.sleep(1.1)  # Wait for expiration
            result2 = test_func("a")

            assert result1 == "result-a"
            assert result2 == "result-a"
            assert call_count == 2  # Called twice due to expiration

    def test_method_caching_per_instance(self) -> None:
        """Test that instance methods cache per-instance."""
        call_count = 0

        class TestClass:
            @cached("method")
            def test_method(self, arg: str) -> str:
                nonlocal call_count
                call_count += 1
                return f"result-{arg}"

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=30,
            )

            obj1 = TestClass()
            obj2 = TestClass()

            result1 = obj1.test_method("a")
            result2 = obj1.test_method("a")  # Same instance, same args - cached
            result3 = obj2.test_method("a")  # Different instance - not cached

            assert result1 == "result-a"
            assert result2 == "result-a"
            assert result3 == "result-a"
            assert call_count == 2  # obj1 cached, obj2 not


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_clear_cache(self) -> None:
        """Test clearing all cache entries."""
        # Add some entries
        _cache["key1"] = (time.time(), "value1")
        _cache["key2"] = (time.time(), "value2")

        count = clear_cache()

        assert count == 2
        assert len(_cache) == 0

    def test_clear_expired(self) -> None:
        """Test clearing only expired entries."""
        now = time.time()
        # Add expired and non-expired entries
        _cache["old"] = (now - 100, "old_value")
        _cache["new"] = (now, "new_value")

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=30,
            )

            count = clear_expired()

            assert count == 1
            assert "old" not in _cache
            assert "new" in _cache

    def test_invalidate_pattern(self) -> None:
        """Test invalidating entries by pattern."""
        _cache["workbenches:ns1"] = (time.time(), [])
        _cache["workbenches:ns2"] = (time.time(), [])
        _cache["projects:all"] = (time.time(), [])

        count = invalidate("workbenches")

        assert count == 2
        assert "workbenches:ns1" not in _cache
        assert "workbenches:ns2" not in _cache
        assert "projects:all" in _cache

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        now = time.time()
        _cache["fresh"] = (now, "value")
        _cache["stale"] = (now - 100, "old_value")

        with patch("rhoai_mcp.utils.cache.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                enable_response_caching=True,
                cache_ttl_seconds=30,
            )

            stats = cache_stats()

            assert stats["total_entries"] == 2
            assert stats["expired_entries"] == 1
            assert stats["active_entries"] == 1
            assert stats["caching_enabled"] is True
            assert stats["ttl_seconds"] == 30
