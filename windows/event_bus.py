"""Thread-safe in-process pub/sub event bus.

Subsystems publish state changes here; the web server subscribes once
and broadcasts every event to connected WebSocket clients. Handlers
run synchronously in the emitter's thread so the bus itself adds no
queueing latency. Long-running subscribers must hand off to their own
thread (the WebSocket hub does this with an asyncio queue).
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger("merlin.bus")


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._wildcards: list[Callable] = []
        self._lock = threading.Lock()

    def on(self, event: str, handler: Callable) -> None:
        with self._lock:
            if event == "*":
                self._wildcards.append(handler)
            else:
                self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        with self._lock:
            try:
                if event == "*":
                    self._wildcards.remove(handler)
                else:
                    self._handlers[event].remove(handler)
            except ValueError:
                pass

    def emit(self, event: str, **kwargs: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event, []))
            wildcards = list(self._wildcards)
        for handler in handlers:
            try:
                handler(**kwargs)
            except Exception:
                log.exception(f"Handler error for event '{event}'")
        for handler in wildcards:
            try:
                handler(event, **kwargs)
            except Exception:
                log.exception(f"Wildcard handler error for event '{event}'")
