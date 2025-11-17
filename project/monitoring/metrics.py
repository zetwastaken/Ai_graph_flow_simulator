"""Prometheus metrics helpers."""

from __future__ import annotations

from typing import Dict

try:
    from prometheus_client import Counter, Histogram, Summary, start_http_server
except Exception:  # pragma: no cover - optional dependency
    Counter = Histogram = Summary = None  # type: ignore
    start_http_server = None  # type: ignore


class MetricsRegistry:
    def __init__(self, port: int = 8008):
        self.enabled = Counter is not None
        if self.enabled and start_http_server:
            start_http_server(port)
            self.stream_samples = Counter(
                "stream_samples_total", "Number of streamed payloads"
            )
            self.sample_latency = Summary(
                "stream_generation_seconds", "Latency of stream generation"
            )
        else:
            self.stream_samples = None
            self.sample_latency = None

    def observe_payload(self, payload: Dict):
        if not self.enabled:
            return
        if self.stream_samples:
            self.stream_samples.inc()
