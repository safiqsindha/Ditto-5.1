"""
Shared rate-limiting primitives for v5.1 runners.

TokenBucket: per-provider token bucket with adaptive backpressure.
  - Thread-safe: multiple OR model threads share one bucket per provider.
  - Adaptive: record_429() automatically reduces the rate on sustained 429 bursts.
  - acquire() blocks until a token is available; never spins.
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """
    Token bucket rate limiter with adaptive backpressure on 429 responses.

    Parameters
    ----------
    rate  : Steady-state tokens/sec (= max sustained RPS).
    burst : Bucket capacity; allows short bursts above the steady rate.
    """

    def __init__(self, rate: float, burst: float) -> None:
        self._rate   = rate
        self._burst  = burst
        self._tokens = burst          # start full — first call is immediate
        self._last   = time.monotonic()
        self._lock   = threading.Lock()
        self._n_ok   = 0
        self._n_429  = 0

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._burst,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / max(self._rate, 1e-6)
            self._tokens = 0.0
        time.sleep(wait)

    def record_success(self) -> None:
        with self._lock:
            self._n_ok += 1

    def record_429(self) -> None:
        """
        Adaptive backpressure: reduce rate when 429 fraction is high.
        Only fires after ≥10 total requests to avoid jitter on sparse data.
        """
        with self._lock:
            self._n_429 += 1
            total = self._n_ok + self._n_429
            if total < 10:
                return
            rate_429 = self._n_429 / total
            if rate_429 > 0.15:
                self._rate = max(0.25, self._rate * 0.50)
            elif rate_429 > 0.05:
                self._rate = max(0.25, self._rate * 0.70)
