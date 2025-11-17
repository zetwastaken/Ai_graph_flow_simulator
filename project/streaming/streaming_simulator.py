"""Real-time streaming utilities for the flow simulator."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from ..database.timeseries import TimeseriesClient
from ..monitoring.metrics import MetricsRegistry
from ..simulator import FlowSimulator


class StreamingSimulator:
    """Wraps the batch simulator with a time-step streaming API."""

    def __init__(self, simulator: FlowSimulator):
        self.simulator = simulator
        self.config = simulator.config
        self.buffer: Deque[Dict] = deque(maxlen=self.config.stream_buffer_size)
        self.listeners: List[Callable[[Dict], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.timeseries = TimeseriesClient(self.config.database_url, self.config.output_dir)
        self.metrics = MetricsRegistry()

    def register_listener(self, listener: Callable[[Dict], None]):
        self.listeners.append(listener)

    def _notify(self, payload: Dict):
        for listener in list(self.listeners):
            try:
                listener(payload)
            except Exception:
                continue

    def _stream_loop(self, real_time: bool):
        generator = self.simulator.data_generator.stream_time_series(
            self.simulator.topology, real_time=real_time
        )
        for payload in generator:
            if self._stop.is_set():
                break
            self.buffer.append(payload)
            self.timeseries.write_points(payload)
            self.metrics.observe_payload(payload)
            self._notify(payload)

    def start_background(self, real_time: bool = True):
        if self._thread and self._thread.is_alive():
            return
        if self.simulator.time_series is None:
            self.simulator.setup()
            self.simulator.run()
        self._stop.clear()
        self._thread = threading.Thread(target=self._stream_loop, args=(real_time,), daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=2.0)

    def latest_payload(self) -> Optional[Dict]:
        if not self.buffer:
            return None
        return self.buffer[-1]

    def snapshot(self, limit: int = 100) -> List[Dict]:
        limit = min(limit, len(self.buffer))
        return list(self.buffer)[-limit:]
