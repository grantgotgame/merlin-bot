"""
Merlin — Ambient AI Companion (Windows Edition)
================================================

Single-machine setup: EMEET PIXY + MSI laptop + BT speaker.
Everything runs locally. No cloud. No Pi needed.

Start:  python merlin.py            (web UI auto-opens at http://localhost:8800)
        python merlin.py --no-web   (terminal-only fallback)
Stop:   Ctrl+C

Prerequisites:
  - LM Studio running with a model loaded (localhost:1234)
  - EMEET PIXY plugged in via USB
  - BT speaker connected
  - All pip dependencies installed (see requirements.txt)
  - Kokoro model files in this directory (see voice.py)

The boot sequence:
  1. Bring up the EventBus, settings store, and SQLite store immediately.
  2. Build the AudioPipeline on the main thread (WASAPI requires it) so the
     PortAudio stream is alive before anything else.
  3. Start the FastAPI web server on a daemon thread and open Chrome — the
     user sees the UI now, even though STT/voice/brain are still loading.
  4. Load STT, Voice, Brain, and the FaceTracker on a worker thread, emitting
     boot_progress events the UI renders as a system-bubble checklist.
  5. Once everything is ready, run the conversation loop on the main thread.
"""

import argparse
import threading
import signal
import sys
import time
import webbrowser

import config
from audio import AudioPipeline
from event_bus import EventBus
from settings_store import Settings
from store import Store
import sounds


class Merlin:
    def __init__(self, web: bool = True, port: int = 8800):
        self.web_enabled = web
        self.port = port

        self._banner()

        self.bus = EventBus()
        self.settings = Settings()
        self.store = Store(bus=self.bus)

        # Audio MUST be built on the main thread — WASAPI requires COM STA.
        # Everything else can come up afterwards on a worker.
        self._emit_boot("audio", "loading")
        self.audio = AudioPipeline(bus=self.bus)
        self._emit_boot("audio", "ready")

        # Subsystems that take a while to load — built on a worker thread so
        # the web UI is visible while they spin up.
        self.stt = None
        self.voice = None
        self.brain = None
        self.tracker = None

        self._running = False
        self._ready = threading.Event()
        self._server_thread: threading.Thread | None = None

    @staticmethod
    def _banner():
        print()
        print("=" * 50)
        print("  MERLIN — Booting up...")
        print("=" * 50)
        print()

    def _emit_boot(self, stage: str, status: str, detail: str = ""):
        """boot_progress carries human-readable boot state for the UI."""
        self.bus.emit("boot_progress", stage=stage, status=status, detail=detail)
        print(f"[boot] {stage}: {status} {detail}".rstrip())

    # ------------------------------------------------------------------
    # Worker-thread subsystem loader
    # ------------------------------------------------------------------

    def _load_subsystems(self):
        # Imports inside the worker so kokoro_onnx / faster_whisper / cv2 don't
        # block module import for the web server.
        from stt import STT
        from voice import Voice
        from brain import Brain
        from tracker import FaceTracker

        try:
            self._emit_boot("stt", "loading")
            self.stt = STT(bus=self.bus)
            self._emit_boot("stt", "ready",
                            f"{config.WHISPER_MODEL} on {'CPU' if self.stt._on_cpu else 'GPU'}")
        except Exception as e:
            self._emit_boot("stt", "failed", str(e))

        try:
            self._emit_boot("voice", "loading")
            self.voice = Voice(bus=self.bus)
            self._emit_boot("voice", "ready",
                            "ok" if self.voice.tts else "tts disabled (model missing)")
        except Exception as e:
            self._emit_boot("voice", "failed", str(e))

        try:
            self._emit_boot("brain", "loading")
            self.brain = Brain(bus=self.bus)
            self._emit_boot("brain", "ready")
        except Exception as e:
            self._emit_boot("brain", "failed", str(e))

        try:
            self._emit_boot("tracker", "loading")
            self.tracker = FaceTracker(bus=self.bus)
            self.tracker.on_face_arrived = self._on_face_arrived
            self.tracker.on_face_lost = self._on_face_lost
            threading.Thread(target=self.tracker.run, daemon=True,
                             name="merlin-tracker").start()
            self._emit_boot("tracker", "ready")
        except Exception as e:
            self._emit_boot("tracker", "failed", str(e))

        # Tell the server thread the heavy subsystems are now in `comps`.
        self.bus.emit("subsystems_ready")
        self._ready.set()

    # ------------------------------------------------------------------
    # Face callbacks
    # ------------------------------------------------------------------

    def _on_face_arrived(self):
        if not self.brain or not self.voice:
            return
        greeting = self.brain.on_face_arrived()
        if greeting:
            self.audio.suppress(timeout=30.0)
            try:
                sounds.greeting()
                self.bus.emit("merlin_speaks", text=greeting, source="greeting")
                print(f"  Merlin: {greeting}")
                self.voice.speak(greeting)
            finally:
                self.audio.unsuppress()

    def _on_face_lost(self):
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True

        # Bring the UI up FIRST so the user sees boot progress live.
        if self.web_enabled:
            self._start_web()
        else:
            self._announce_terminal()

        # Audio capture starts immediately; STT/voice/brain load in parallel.
        self.audio.start()

        loader = threading.Thread(target=self._load_subsystems, daemon=True,
                                  name="merlin-loader")
        loader.start()

        # Wait for subsystems to finish loading before entering the main loop.
        # The UI is already up and responsive while we wait.
        while self._running and not self._ready.wait(timeout=0.5):
            pass

        if not self._running:
            return

        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _tick(self):
        audio_data = self.audio.get_utterance(timeout=0.2)
        if audio_data is None:
            return

        if not self.stt or not self.brain:
            return  # subsystems not yet loaded

        text = self.stt.transcribe(audio_data)
        if not text:
            return

        print(f"  You: {text}")

        self.audio.suppress(timeout=60.0)
        try:
            sounds.listening()
            response = self.brain.process(text)
            if response is None:
                return

            sounds.thinking()
            print(f"  Merlin: {response}")
            self.bus.emit("merlin_speaks", text=response, source="reply")

            if self.voice:
                self.voice.speak(response)
            time.sleep(0.3)
        finally:
            self.audio.unsuppress()

    def _start_web(self):
        try:
            from server import Components, run_in_thread
        except ImportError as e:
            print(f"[merlin] Web UI dependencies missing: {e}")
            print("[merlin] Install with: pip install fastapi 'uvicorn[standard]'")
            print("[merlin] Falling back to terminal-only mode.")
            self.web_enabled = False
            self._announce_terminal()
            return

        # Components carries lazy refs to subsystems — server.py reads through
        # `comps.<name>` each request, so it sees them as they come online.
        self._comps = Components(
            bus=self.bus, settings=self.settings, store=self.store,
            audio=self.audio,
            stt=None, voice=None, brain=None, tracker=None,
        )

        # When the loader finishes, swap the lazy refs in.
        def _on_ready(**_):
            self._comps.stt = self.stt
            self._comps.voice = self.voice
            self._comps.brain = self.brain
            self._comps.tracker = self.tracker

        self.bus.on("subsystems_ready", _on_ready)

        self._server_thread = run_in_thread(self._comps, host="127.0.0.1", port=self.port)

        # Give uvicorn a moment to bind, then open the browser.
        time.sleep(0.6)
        url = f"http://localhost:{self.port}"
        print()
        print("=" * 50)
        print(f"  Merlin is awake at {url}")
        print(f"  Say \"{config.WAKE_WORDS[0]}\" to start talking.")
        print("  (Subsystems are still loading — watch the UI for progress.)")
        print("  Press Ctrl+C to quit.")
        print("=" * 50)
        print()
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _announce_terminal(self):
        print()
        print("=" * 50)
        print("  Merlin is listening.")
        print(f"  Say \"{config.WAKE_WORDS[0]}\" to start talking.")
        print(f"  Say \"stop listening\" to mute.")
        print("  Press Ctrl+C to quit.")
        print("=" * 50)
        print()

    def stop(self):
        print("\n  Shutting down...")
        self._running = False
        for closer in (
            lambda: self.audio.stop(),
            lambda: self.tracker.stop() if self.tracker else None,
            lambda: self.store.close(),
        ):
            try:
                closer()
            except Exception:
                pass
        print("  Merlin offline. Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="Merlin (Windows)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable the web UI; run in pure terminal mode.")
    parser.add_argument("--port", type=int, default=8800,
                        help="Web server port (default 8800).")
    args = parser.parse_args()

    merlin = Merlin(web=not args.no_web, port=args.port)

    def handle_exit(sig, frame):
        merlin.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    merlin.start()


if __name__ == "__main__":
    main()
