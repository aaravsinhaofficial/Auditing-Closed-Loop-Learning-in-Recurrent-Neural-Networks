from __future__ import annotations

import time
from datetime import datetime, timezone


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


class Heartbeat:
    def __init__(self, label: str, total: int, interval_seconds: float = 30.0) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.started = time.monotonic()
        self.last = self.started
        log(f"{self.label}: started; total={self.total}")

    def maybe(self, current: int, detail: str = "") -> None:
        now = time.monotonic()
        if now - self.last < self.interval_seconds and current < self.total:
            return
        self.last = now
        elapsed = now - self.started
        pct = 100.0 * min(current, self.total) / self.total
        suffix = f"; {detail}" if detail else ""
        log(f"{self.label}: {current}/{self.total} ({pct:.1f}%) elapsed={elapsed:.1f}s{suffix}")

    def done(self, detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        suffix = f"; {detail}" if detail else ""
        log(f"{self.label}: done in {elapsed:.1f}s{suffix}")
