"""Merlin v2 — Simple in-process pub/sub event bus."""

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger("merlin.bus")


class EventBus:
    """Thread-safe event bus. Handlers run synchronously in the emitter's thread."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event: str, handler: Callable) -> None:
        """Subscribe to an event."""
        with self._lock:
            self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unsubscribe from an event."""
        with self._lock:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event. Handlers are called with keyword arguments."""
        with self._lock:
            handlers = list(self._handlers.get(event, []))
        for handler in handlers:
            try:
                handler(**kwargs)
            except Exception:
                log.exception(f"Handler error for event '{event}'")
