"""Merlin v2 — Orchestrator: starts modules, supervises, serves HTTP."""

import http.server
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime

from event_bus import EventBus
from audio_pipeline import AudioPipeline
from voice import Voice
from brain import Brain
from vision import Vision
import config

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("merlin.main")

# ── Module Registry ──────────────────────────────────────────────


class ModuleInfo:
    def __init__(self, name, instance):
        self.name = name
        self.instance = instance
        self.restarts = 0
        self.last_restart = None
        self.restart_times = []  # timestamps of recent restarts
        self.failed = False


class Orchestrator:
    def __init__(self):
        self.bus = EventBus()
        self.start_time = time.time()
        self.modules: list[ModuleInfo] = []
        self._running = True
        self._muted = False
        self.bus.on("mute_toggled", self._on_mute)

    def _on_mute(self, muted=False, **kw):
        self._muted = muted

    def register(self, name: str, module_class):
        info = ModuleInfo(name, module_class())
        self.modules.append(info)
        return info

    def start_all(self):
        for mod in self.modules:
            self._start_module(mod)

    def _start_module(self, mod: ModuleInfo):
        try:
            mod.instance.start(self.bus)
            log.info(f"Started: {mod.name}")
        except Exception:
            log.exception(f"Failed to start: {mod.name}")

    def _restart_module(self, mod: ModuleInfo):
        now = time.time()
        # Check if too many restarts
        mod.restart_times = [t for t in mod.restart_times if now - t < 60]
        if len(mod.restart_times) >= 3:
            mod.failed = True
            log.critical(f"{mod.name} failed — 3 restarts in 60s, giving up")
            self.bus.emit("module_failed", name=mod.name)
            return

        try:
            mod.instance.stop()
        except Exception:
            pass

        # Create fresh instance
        mod.instance = type(mod.instance)()
        self._start_module(mod)
        mod.restarts += 1
        mod.last_restart = now
        mod.restart_times.append(now)
        log.warning(f"Restarted: {mod.name} (#{mod.restarts})")
        self.bus.emit("module_restarted", name=mod.name)

    def supervision_loop(self):
        while self._running:
            time.sleep(5)
            for mod in self.modules:
                if mod.failed:
                    continue
                if not mod.instance.is_alive():
                    log.warning(f"{mod.name} is dead, restarting...")
                    self._restart_module(mod)

    def stop_all(self):
        self._running = False
        for mod in self.modules:
            try:
                mod.instance.stop()
                log.info(f"Stopped: {mod.name}")
            except Exception:
                log.exception(f"Error stopping: {mod.name}")

    def health(self) -> dict:
        uptime_s = time.time() - self.start_time
        hours = int(uptime_s // 3600)
        mins = int((uptime_s % 3600) // 60)
        return {
            "status": "ok" if all(not m.failed for m in self.modules) else "degraded",
            "uptime": f"{hours}h {mins}m",
            "modules": {
                m.name: {
                    "alive": m.instance.is_alive() and not m.failed,
                    "restarts": m.restarts,
                    "last_restart": (
                        f"{int(time.time() - m.last_restart)}s ago"
                        if m.last_restart else None
                    ),
                    "failed": m.failed,
                }
                for m in self.modules
            },
            "muted": self._muted,
        }


# ── HTTP Server (tracker bridge + health) ────────────────────────


class MerlinHTTPHandler(http.server.BaseHTTPRequestHandler):
    orchestrator = None  # set before serving

    def do_POST(self):
        if self.path == "/event":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                event_type = data.get("type", "")
                if event_type in ("face_arrived", "face_lost", "pir_motion"):
                    self.orchestrator.bus.emit(event_type)
                    self.send_response(200)
                else:
                    log.debug(f"Unknown event type: {event_type}")
                    self.send_response(200)
            except Exception:
                self.send_response(400)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            health = self.orchestrator.health()
            body = json.dumps(health, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging


# ── Main ─────────────────────────────────────────────────────────

def main():
    orch = Orchestrator()

    # Register modules
    orch.register("audio_pipeline", AudioPipeline)
    orch.register("voice", Voice)
    orch.register("brain", Brain)
    orch.register("vision", Vision)

    # Graceful shutdown
    def shutdown(sig, frame):
        log.info("Shutting down...")
        orch.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start HTTP server (SO_REUSEADDR prevents "Address already in use" on restart)
    MerlinHTTPHandler.orchestrator = orch
    http.server.HTTPServer.allow_reuse_address = True
    http_server = http.server.HTTPServer(("0.0.0.0", config.TRACKER_LISTEN_PORT), MerlinHTTPHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True, name="http")
    http_thread.start()
    log.info(f"HTTP server on :{config.TRACKER_LISTEN_PORT}")

    # Start all modules
    orch.start_all()

    log.info("=" * 50)
    log.info("Merlin v2 — All modules running")
    log.info(f"Health: http://localhost:{config.TRACKER_LISTEN_PORT}/health")
    log.info("=" * 50)

    # Supervision loop (blocking)
    orch.supervision_loop()


if __name__ == "__main__":
    main()
