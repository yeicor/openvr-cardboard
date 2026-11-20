"""Small networking helpers shared by the UDP receivers.

The drivers and the phone links are unauthenticated UDP servers on the local
network. A flood of datagrams from one host must not starve the receive loop or
make the application spend all of its time parsing packets, so each receiver
applies a per-source token bucket before doing any work.
"""

from __future__ import annotations

import time
from typing import Hashable, Optional


class PacketRateLimiter:
    """Per-source token bucket for UDP datagrams.

    ``rate_hz`` is the sustained number of packets accepted from one address
    and ``burst`` the short allowance (default: 100 ms worth). Sources that go
    quiet are forgotten after a few seconds, and the table is capped so a scan
    over spoofed source ports cannot grow it without bound.
    """

    def __init__(
        self,
        rate_hz: float = 2000.0,
        burst: Optional[float] = None,
        max_sources: int = 64,
    ):
        self.rate_hz = max(1.0, float(rate_hz))
        self.burst = max(1.0, float(burst if burst is not None else self.rate_hz * 0.1))
        self.max_sources = max(1, int(max_sources))
        self.dropped = 0
        self._buckets: dict[Hashable, tuple[float, float]] = {}
        self._last_prune = 0.0

    def allow(self, source: Hashable, now: Optional[float] = None) -> bool:
        """Return True when a datagram from ``source`` may be processed."""
        stamp = time.monotonic() if now is None else float(now)
        if stamp - self._last_prune > 5.0:
            self._prune(stamp)
            self._last_prune = stamp
        bucket = self._buckets.get(source)
        if bucket is None:
            if len(self._buckets) >= self.max_sources:
                self._prune(stamp)
                if len(self._buckets) >= self.max_sources:
                    self._buckets.pop(next(iter(self._buckets)))
            tokens = self.burst
        else:
            tokens, last = bucket
            tokens = min(self.burst, tokens + max(0.0, stamp - last) * self.rate_hz)
        if tokens < 1.0:
            self._buckets[source] = (tokens, stamp)
            self.dropped += 1
            return False
        self._buckets[source] = (tokens - 1.0, stamp)
        return True

    def _prune(self, now: float) -> None:
        stale = [
            source
            for source, (_, last) in self._buckets.items()
            if now - last > 5.0
        ]
        for source in stale:
            del self._buckets[source]
