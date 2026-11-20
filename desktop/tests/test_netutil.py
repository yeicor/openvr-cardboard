"""Unit coverage for the shared UDP packet rate limiter."""

from openvrcardboard.netutil import PacketRateLimiter


def test_burst_is_allowed_then_excess_is_dropped():
    limiter = PacketRateLimiter(rate_hz=100.0, burst=5.0)
    accepted = [limiter.allow("phone", now=10.0) for _ in range(7)]
    assert accepted == [True] * 5 + [False] * 2
    assert limiter.dropped == 2


def test_tokens_refill_at_the_configured_rate():
    limiter = PacketRateLimiter(rate_hz=100.0, burst=2.0)
    assert limiter.allow("phone", now=1.0)
    assert limiter.allow("phone", now=1.0)
    assert not limiter.allow("phone", now=1.0)
    assert limiter.allow("phone", now=1.01)
    assert not limiter.allow("phone", now=1.01)


def test_sources_are_limited_independently():
    limiter = PacketRateLimiter(rate_hz=10.0, burst=1.0)
    assert limiter.allow("a", now=0.0)
    assert not limiter.allow("a", now=0.0)
    assert limiter.allow("b", now=0.0)


def test_bucket_table_is_capped():
    limiter = PacketRateLimiter(rate_hz=10.0, burst=1.0, max_sources=2)
    assert limiter.allow("a", now=0.0)
    assert limiter.allow("b", now=0.0)
    assert limiter.allow("c", now=0.0)
    assert len(limiter._buckets) <= 2
    # The newest source must still be usable after eviction.
    assert limiter.allow("c", now=0.2)


def test_quiet_sources_are_forgotten():
    limiter = PacketRateLimiter(rate_hz=10.0, burst=1.0, max_sources=4)
    assert limiter.allow("old", now=0.0)
    assert limiter.allow("new", now=10.0)
    assert "old" not in limiter._buckets
