"""FastAPI server: WebSocket hub, REST API, static UI, MJPEG camera feed.

Started on a daemon uvicorn thread by merlin.py. The audio/STT/brain/voice
loop runs unmodified on the main thread; this server reads state from the
shared `Components` bag and pushes bus events to connected browsers.
"""

import asyncio
import io
import json
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# ----------------------------------------------------------------------
# Components bag — passed in from merlin.py
# ----------------------------------------------------------------------

@dataclass
class Components:
    bus: Any
    settings: Any
    store: Any
    audio: Any = None
    stt: Any = None
    voice: Any = None
    brain: Any = None
    tracker: Any = None
    started_at: float = field(default_factory=time.time)
    health: dict = field(default_factory=dict)
    latency: dict = field(default_factory=lambda: {"stt_ms": None, "llm_ms": None, "tts_ms": None})
    # Per-subsystem boot state, updated by boot_progress events. Lets
    # late-joining WebSocket clients render the loading banner correctly
    # even if they connect after subsystems_ready.
    boot: dict = field(default_factory=lambda: {
        "audio": "pending", "stt": "pending", "voice": "pending",
        "brain": "pending", "tracker": "pending",
    })
    boot_detail: dict = field(default_factory=dict)
    subsystems_ready: bool = False
    # Last N events broadcast over the bus, for replay to late joiners.
    recent_events: list = field(default_factory=list)


# ----------------------------------------------------------------------
# WebSocket hub
# ----------------------------------------------------------------------

_REPLAY_TYPES = {
    "boot_progress", "subsystems_ready", "system_message",
    "mute_toggled", "settings_updated",
}
_REPLAY_BUFFER_MAX = 50


class WSHub:
    """Owns the asyncio event loop. The bus emits from many threads;
    we hand events off via a thread-safe call_soon_threadsafe and broadcast
    to every connected client through their own asyncio queue.

    Also maintains a small replay buffer of important events so that
    clients connecting AFTER boot still see the boot banner, mute state,
    etc. without having to call /api/health themselves."""

    def __init__(self, comps: "Components" = None):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self.comps = comps

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def attach(self, bus):
        bus.on("*", self._on_bus_event)

    def _on_bus_event(self, event: str, **payload):
        if self.loop is None:
            return
        msg = {"type": event, "ts": time.time(), "data": payload}

        # Update authoritative boot state on the Components bag so HTTP
        # clients (and late-joining WS clients) see consistent results.
        if self.comps is not None:
            if event == "boot_progress":
                stage = payload.get("stage")
                status = payload.get("status")
                if stage:
                    self.comps.boot[stage] = status
                    if payload.get("detail"):
                        self.comps.boot_detail[stage] = payload["detail"]
            elif event == "subsystems_ready":
                self.comps.subsystems_ready = True

            if event in _REPLAY_TYPES:
                buf = self.comps.recent_events
                buf.append(msg)
                if len(buf) > _REPLAY_BUFFER_MAX:
                    del buf[: len(buf) - _REPLAY_BUFFER_MAX]

        self.loop.call_soon_threadsafe(self._broadcast, msg)

    def _broadcast(self, msg: dict):
        dead = []
        for q in list(self._clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._clients.discard(q)

    async def add_client(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        with self._lock:
            self._clients.add(q)
        return q

    async def remove_client(self, q: asyncio.Queue):
        with self._lock:
            self._clients.discard(q)


# ----------------------------------------------------------------------
# Pydantic input models
# ----------------------------------------------------------------------

class SayBody(BaseModel):
    text: str


class SpeakBody(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None


class PreviewBody(BaseModel):
    voice: str
    text: str = "Hello, I am Merlin."


# ----------------------------------------------------------------------
# Voice catalog — a curated default; if the bin file lists more, return all.
# ----------------------------------------------------------------------

KOKORO_VOICE_CATALOG = [
    # American male
    {"id": "am_adam", "label": "Adam", "gender": "m", "accent": "us"},
    {"id": "am_echo", "label": "Echo", "gender": "m", "accent": "us"},
    {"id": "am_eric", "label": "Eric", "gender": "m", "accent": "us"},
    {"id": "am_fenrir", "label": "Fenrir", "gender": "m", "accent": "us"},
    {"id": "am_liam", "label": "Liam", "gender": "m", "accent": "us"},
    {"id": "am_michael", "label": "Michael", "gender": "m", "accent": "us"},
    {"id": "am_onyx", "label": "Onyx", "gender": "m", "accent": "us"},
    {"id": "am_puck", "label": "Puck", "gender": "m", "accent": "us"},
    {"id": "am_santa", "label": "Santa", "gender": "m", "accent": "us"},
    # American female
    {"id": "af_alloy", "label": "Alloy", "gender": "f", "accent": "us"},
    {"id": "af_aoede", "label": "Aoede", "gender": "f", "accent": "us"},
    {"id": "af_bella", "label": "Bella", "gender": "f", "accent": "us"},
    {"id": "af_heart", "label": "Heart", "gender": "f", "accent": "us"},
    {"id": "af_jessica", "label": "Jessica", "gender": "f", "accent": "us"},
    {"id": "af_kore", "label": "Kore", "gender": "f", "accent": "us"},
    {"id": "af_nicole", "label": "Nicole", "gender": "f", "accent": "us"},
    {"id": "af_nova", "label": "Nova", "gender": "f", "accent": "us"},
    {"id": "af_river", "label": "River", "gender": "f", "accent": "us"},
    {"id": "af_sarah", "label": "Sarah", "gender": "f", "accent": "us"},
    {"id": "af_sky", "label": "Sky", "gender": "f", "accent": "us"},
    # British male
    {"id": "bm_daniel", "label": "Daniel", "gender": "m", "accent": "uk"},
    {"id": "bm_fable", "label": "Fable", "gender": "m", "accent": "uk"},
    {"id": "bm_george", "label": "George", "gender": "m", "accent": "uk"},
    {"id": "bm_lewis", "label": "Lewis", "gender": "m", "accent": "uk"},
    # British female
    {"id": "bf_alice", "label": "Alice", "gender": "f", "accent": "uk"},
    {"id": "bf_emma", "label": "Emma", "gender": "f", "accent": "uk"},
    {"id": "bf_isabella", "label": "Isabella", "gender": "f", "accent": "uk"},
    {"id": "bf_lily", "label": "Lily", "gender": "f", "accent": "uk"},
]


def _samples_to_wav_bytes(samples, sample_rate: int) -> bytes:
    """Encode float32 samples in [-1, 1] to a 16-bit PCM WAV in-memory."""
    arr = np.asarray(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------

def create_app(comps: Components, hub: WSHub) -> FastAPI:
    app = FastAPI(title="Merlin", docs_url="/api/docs", redoc_url=None)

    @app.on_event("startup")
    async def _on_startup():
        hub.bind_loop(asyncio.get_running_loop())

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = await hub.add_client()
        try:
            # Hello carries current health + a replay of recent important
            # events (boot_progress, mute, etc.) so the UI is consistent
            # even if the client connects after subsystems boot.
            await ws.send_json({
                "type": "hello",
                "ts": time.time(),
                "data": _health_payload(comps),
            })
            for ev in list(comps.recent_events):
                await ws.send_json(ev)
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.remove_client(q)

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        return _health_payload(comps)

    @app.get("/api/devices")
    def devices():
        try:
            raw = sd.query_devices()
            apis = sd.query_hostapis()
            inputs, outputs = [], []
            for i, d in enumerate(raw):
                api = apis[d["hostapi"]]["name"]
                entry = {
                    "index": i,
                    "name": d["name"],
                    "api": api,
                    "default_samplerate": int(d.get("default_samplerate", 0)),
                }
                if d["max_input_channels"] > 0:
                    inputs.append(entry)
                if d["max_output_channels"] > 0:
                    outputs.append(entry)
            current_mic = getattr(comps.audio, "mic_device", None)
            return {
                "inputs": inputs,
                "outputs": outputs,
                "current_mic": current_mic,
                "current_speaker": comps.settings.get("SPEAKER_DEVICE"),
            }
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/voices")
    def voices():
        return {
            "current": comps.settings.get("KOKORO_VOICE"),
            "speed": comps.settings.get("KOKORO_SPEED"),
            "voices": KOKORO_VOICE_CATALOG,
        }

    @app.post("/api/voices/preview")
    def voice_preview(body: PreviewBody):
        if not comps.voice or not comps.voice.tts:
            raise HTTPException(503, "TTS not available")
        try:
            samples, sr = comps.voice.tts.create(body.text, voice=body.voice, speed=1.0)
            wav = _samples_to_wav_bytes(samples, sr)
            return Response(content=wav, media_type="audio/wav")
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/api/settings")
    def get_settings():
        return comps.settings.all()

    @app.patch("/api/settings")
    async def patch_settings(req: Request):
        body = await req.json()
        result = comps.settings.patch(body)
        # Apply hot changes immediately
        for key, info in result.items():
            if info.get("status") != "applied":
                continue
            _apply_hot(comps, key, body[key])
        comps.bus.emit("settings_updated", changes=result)
        return result

    @app.post("/api/say")
    def say(body: SayBody):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        # Echo the user's input back over the bus so the UI gets a real
        # 'you' bubble even if the brain is still loading. The optimistic
        # bubble in the UI is removed when this event arrives, so the
        # transcript stays consistent across reloads.
        comps.bus.emit("user_message", text=text, source="typed")
        if not comps.brain:
            comps.bus.emit(
                "system_message",
                text="Brain still loading — try again in a moment.",
                level="warn",
            )
            return JSONResponse({"queued": False, "reason": "brain_loading"}, status_code=202)

        def _run():
            response = (
                comps.brain.process_typed(text)
                if hasattr(comps.brain, "process_typed")
                else comps.brain.process(text)
            )
            if response and comps.voice:
                try:
                    if comps.audio:
                        comps.audio.suppress(timeout=60.0)
                    try:
                        comps.voice.speak(response)
                    finally:
                        if comps.audio:
                            comps.audio.unsuppress()
                except Exception as e:
                    print(f"[server] speak after /say failed: {e}")
                    comps.bus.emit("system_message", text=f"speak failed: {e}", level="error")
        threading.Thread(target=_run, daemon=True).start()
        return {"queued": True}

    @app.post("/api/speak")
    def speak(body: SpeakBody):
        if not comps.voice:
            raise HTTPException(503, "voice not ready")
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        def _run():
            try:
                if comps.audio:
                    comps.audio.suppress(timeout=60.0)
                try:
                    comps.voice.speak(text)
                finally:
                    if comps.audio:
                        comps.audio.unsuppress()
            except Exception as e:
                print(f"[server] /speak failed: {e}")
        threading.Thread(target=_run, daemon=True).start()
        return {"queued": True}

    @app.post("/api/shutdown")
    def shutdown():
        """Shut Merlin down cleanly from the web UI. Used by the
        'Quit Merlin' command-palette entry so the user doesn't have
        to find a hidden console window in Task Manager."""
        import os, sys, threading as _t
        comps.bus.emit("system_message", text="Shutting down…", level="info")

        def _die():
            time.sleep(0.4)
            try:
                comps.store.close()
            except Exception:
                pass
            try:
                if comps.audio: comps.audio.stop()
            except Exception:
                pass
            try:
                if comps.tracker: comps.tracker.stop()
            except Exception:
                pass
            os._exit(0)

        _t.Thread(target=_die, daemon=True).start()
        return {"shutdown": True}

    @app.post("/api/command/{name}")
    def command(name: str):
        if name == "mute":
            if comps.brain:
                comps.brain.muted = True
                comps.bus.emit("mute_toggled", muted=True)
            return {"muted": True}
        if name == "unmute":
            if comps.brain:
                comps.brain.muted = False
                comps.bus.emit("mute_toggled", muted=False)
            return {"muted": False}
        if name == "clear_history":
            if comps.brain:
                comps.brain.history = []
                comps.brain.last_response_time = 0
                comps.bus.emit("history_cleared")
            return {"cleared": True}
        if name == "skip_tts":
            try:
                sd.stop()
                comps.bus.emit("tts_skipped")
            except Exception:
                pass
            return {"skipped": True}
        raise HTTPException(404, f"unknown command: {name}")

    @app.get("/api/history")
    def history(session: int | None = None, limit: int = 200):
        return {"messages": comps.store.history(session_id=session, limit=limit)}

    @app.get("/api/sessions")
    def sessions(limit: int = 50):
        return {"sessions": comps.store.list_sessions(limit=limit)}

    @app.get("/api/search")
    def search(q: str = "", limit: int = 50):
        return {"results": comps.store.search(q, limit=limit)}

    @app.get("/api/logs")
    def logs(since: float | None = None, limit: int = 200):
        return {"events": comps.store.recent_events(since=since, limit=limit)}

    @app.get("/api/debug")
    def debug():
        """Snapshot of subsystem state — handy when something seems wrong."""
        return {
            "subsystems_ready": comps.subsystems_ready,
            "boot": comps.boot,
            "boot_detail": comps.boot_detail,
            "loaded": {
                "audio": comps.audio is not None,
                "stt": comps.stt is not None,
                "voice": comps.voice is not None and getattr(comps.voice, "tts", None) is not None,
                "brain": comps.brain is not None,
                "tracker": comps.tracker is not None,
            },
            "ws_clients": len(hub._clients),
            "session_id": comps.store.session_id,
            "recent_event_buffer": len(comps.recent_events),
        }

    # ------------------------------------------------------------------
    # MJPEG camera feed (best-effort)
    # ------------------------------------------------------------------

    # Single-frame JPEG endpoint — the UI polls this every ~150ms. More
    # robust than multipart MJPEG, which Chrome aborts silently if the
    # first frame is slow to arrive (e.g. tracker still initializing).
    @app.get("/camera.jpg")
    def camera_jpg():
        jpeg = getattr(comps.tracker, "latest_jpeg", None) if comps.tracker else None
        if not jpeg:
            # Return a 1x1 transparent placeholder so the <img> tag stays
            # valid until the tracker produces real frames.
            placeholder = bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605"
                "0808070709090808"
                "0a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c"
                "2837292c30313434"
                "1f27393d38323c2e333432ffc0000b0801000100018220000ffc4001f00000"
                "1050101010101010"
                "10000000000000000010203040506070809ffc4001fffd9"
            )
            return Response(
                content=placeholder,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    # Multipart MJPEG also exposed for power-users / OBS / debugging.
    @app.get("/camera.mjpg")
    def camera_mjpg():
        if not comps.tracker:
            raise HTTPException(503, "tracker not running")
        boundary = "merlin-frame"

        def gen():
            while True:
                jpeg = getattr(comps.tracker, "latest_jpeg", None)
                if jpeg:
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                time.sleep(0.066)
        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------

    if os.path.isdir(WEB_DIR):
        # Serve index.html at "/" and the rest of /web/* as static
        @app.get("/")
        def index():
            return FileResponse(os.path.join(WEB_DIR, "index.html"))

        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


def _health_payload(comps: Components) -> dict:
    uptime = time.time() - comps.started_at
    modules = {}
    if comps.audio is not None:
        modules["audio"] = {
            "alive": getattr(comps.audio, "_running", False),
            "device": getattr(comps.audio, "mic_device", None),
            "api": getattr(comps.audio, "_api_name", None),
            "rate": getattr(comps.audio, "_mic_rate", None),
        }
    if comps.stt is not None:
        modules["stt"] = {
            "alive": True,
            "on_cpu": getattr(comps.stt, "_on_cpu", None),
            "model": comps.settings.get("WHISPER_MODEL"),
        }
    if comps.voice is not None:
        modules["voice"] = {
            "alive": comps.voice.tts is not None,
            "voice": comps.settings.get("KOKORO_VOICE"),
        }
    if comps.brain is not None:
        modules["brain"] = {
            "alive": True,
            "muted": getattr(comps.brain, "muted", False),
            "history_len": len(getattr(comps.brain, "history", [])),
            "in_window": (time.time() - getattr(comps.brain, "last_response_time", 0)) < comps.settings.get("CONVERSATION_WINDOW"),
        }
    if comps.tracker is not None:
        modules["tracker"] = {
            "alive": getattr(comps.tracker, "_running", False),
            "face_present": getattr(comps.tracker, "_face_present", False),
            "ptz": getattr(comps.tracker, "ptz_available", False),
        }
    return {
        "status": "ok",
        "uptime": uptime,
        "modules": modules,
        "latency": comps.latency,
        "session_id": comps.store.session_id,
        "boot": comps.boot,
        "boot_detail": comps.boot_detail,
        "subsystems_ready": comps.subsystems_ready,
    }


def _apply_hot(comps: Components, key: str, value):
    """Push a hot-applicable setting into the running subsystem."""
    if key == "ENERGY_THRESHOLD":
        # No setter — write directly into config module so audio.py sees the new value
        import config as cfg
        setattr(cfg, key, value)
    elif key == "SILENCE_TIMEOUT":
        import config as cfg
        cfg.SILENCE_TIMEOUT = value
    elif key in ("MIN_UTTERANCE_LENGTH", "MAX_UTTERANCE_LENGTH"):
        import config as cfg
        setattr(cfg, key, value)
    elif key == "KOKORO_VOICE":
        import config as cfg
        cfg.KOKORO_VOICE = value
    elif key == "KOKORO_SPEED":
        import config as cfg
        cfg.KOKORO_SPEED = value
    elif key == "SPEAKER_DEVICE":
        import config as cfg
        cfg.SPEAKER_DEVICE = value
    elif key == "SYSTEM_PROMPT":
        import config as cfg
        cfg.SYSTEM_PROMPT = value
    elif key in ("MAX_HISTORY", "MAX_TOKENS", "TEMPERATURE", "CONVERSATION_WINDOW",
                 "WAKE_WORDS", "MUTE_WORDS", "UNMUTE_WORDS", "NEVERMIND_WORDS",
                 "LLM_URL", "LLM_MODEL", "WHISPER_LANGUAGE",
                 "FACE_CONFIDENCE", "PTZ_ENABLED", "PTZ_SPEED", "PTZ_DEADZONE"):
        import config as cfg
        setattr(cfg, key, value)


# ----------------------------------------------------------------------
# Threaded launcher
# ----------------------------------------------------------------------

def run_in_thread(comps: Components, host: str = "127.0.0.1", port: int = 8800) -> threading.Thread:
    """Start uvicorn in a daemon thread. Returns the thread."""
    import uvicorn

    hub = WSHub(comps=comps)

    # Attach to the bus IMMEDIATELY (synchronously) so boot events fired
    # before uvicorn's startup hook runs are still captured into the
    # replay buffer + boot state. The actual WebSocket broadcast remains
    # gated on `hub.loop` being bound, which happens at uvicorn startup.
    hub.attach(comps.bus)
    comps.bus.on("stt_complete", lambda **kw: comps.latency.update(stt_ms=kw.get("latency_ms")))
    comps.bus.on("thinking_complete", lambda **kw: comps.latency.update(llm_ms=kw.get("latency_ms")))
    comps.bus.on("tts_complete", lambda **kw: comps.latency.update(tts_ms=kw.get("latency_ms")))

    app = create_app(comps, hub)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def _run():
        # uvicorn manages its own asyncio loop
        try:
            server.run()
        except Exception as e:
            print(f"[server] uvicorn crashed: {e}")

    t = threading.Thread(target=_run, daemon=True, name="merlin-web")
    t.start()
    return t
