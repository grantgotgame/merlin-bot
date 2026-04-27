"""Layered settings: defaults from config.py + user overrides from JSON.

`config.py` is the source of truth for default values and is never written.
Overrides live in `config_overrides.json` next to merlin.py, so a user can
change voice/threshold/mic via the web UI without editing Python.

Each setting is tagged either "hot" (apply immediately by calling the
subsystem's setter) or "restart" (the user must explicitly restart that
subsystem for the change to take effect).
"""

import json
import os
import threading
from typing import Any

import config as defaults_module

OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_overrides.json")

# field -> ("hot" or "restart", subsystem name)
_TAGS: dict[str, tuple[str, str]] = {
    # audio
    "ENERGY_THRESHOLD": ("hot", "audio"),
    "SILENCE_TIMEOUT": ("hot", "audio"),
    "MIN_UTTERANCE_LENGTH": ("hot", "audio"),
    "MAX_UTTERANCE_LENGTH": ("hot", "audio"),
    "PIXY_MIC_DEVICE": ("restart", "audio"),
    "SAMPLE_RATE": ("restart", "audio"),
    # speaker / voice
    "SPEAKER_DEVICE": ("hot", "voice"),
    "KOKORO_VOICE": ("hot", "voice"),
    "KOKORO_SPEED": ("hot", "voice"),
    # stt
    "WHISPER_MODEL": ("restart", "stt"),
    "WHISPER_DEVICE": ("restart", "stt"),
    "WHISPER_COMPUTE": ("restart", "stt"),
    "WHISPER_LANGUAGE": ("hot", "stt"),
    # brain
    "LLM_URL": ("hot", "brain"),
    "LLM_MODEL": ("hot", "brain"),
    "SYSTEM_PROMPT": ("hot", "brain"),
    "MAX_HISTORY": ("hot", "brain"),
    "MAX_TOKENS": ("hot", "brain"),
    "TEMPERATURE": ("hot", "brain"),
    "WAKE_WORDS": ("hot", "brain"),
    "MUTE_WORDS": ("hot", "brain"),
    "UNMUTE_WORDS": ("hot", "brain"),
    "NEVERMIND_WORDS": ("hot", "brain"),
    "CONVERSATION_WINDOW": ("hot", "brain"),
    # tracker
    "CAMERA_INDEX": ("restart", "tracker"),
    "CAMERA_WIDTH": ("restart", "tracker"),
    "CAMERA_HEIGHT": ("restart", "tracker"),
    "CAMERA_FPS": ("restart", "tracker"),
    "FACE_CONFIDENCE": ("hot", "tracker"),
    "PTZ_ENABLED": ("hot", "tracker"),
    "PTZ_SPEED": ("hot", "tracker"),
    "PTZ_DEADZONE": ("hot", "tracker"),
}


def _read_defaults() -> dict[str, Any]:
    """Pull every UPPER_CASE attribute from config.py."""
    out = {}
    for k in dir(defaults_module):
        if k.isupper() and not k.startswith("_"):
            v = getattr(defaults_module, k)
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                out[k] = v
    return out


class Settings:
    def __init__(self):
        self._lock = threading.RLock()
        self._defaults = _read_defaults()
        self._overrides: dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(OVERRIDES_FILE):
            try:
                with open(OVERRIDES_FILE, encoding="utf-8") as f:
                    self._overrides = json.load(f)
            except Exception as e:
                print(f"[settings] Failed to load overrides: {e}")
                self._overrides = {}

    def _save(self):
        tmp = OVERRIDES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._overrides, f, indent=2)
        os.replace(tmp, OVERRIDES_FILE)

    def get(self, key: str, fallback: Any = None) -> Any:
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]
            return self._defaults.get(key, fallback)

    def all(self) -> dict[str, Any]:
        """Effective settings (defaults merged with overrides) plus tags."""
        with self._lock:
            merged = dict(self._defaults)
            merged.update(self._overrides)
            tagged = {}
            for k, v in merged.items():
                tag, subsystem = _TAGS.get(k, (None, None))
                tagged[k] = {
                    "value": v,
                    "default": self._defaults.get(k),
                    "overridden": k in self._overrides,
                    "tag": tag,
                    "subsystem": subsystem,
                }
            return tagged

    def patch(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply partial updates. Returns per-field {applied|restart_required, subsystem}."""
        result = {}
        with self._lock:
            for key, value in updates.items():
                if key not in self._defaults:
                    result[key] = {"status": "unknown_key"}
                    continue
                self._overrides[key] = value
                tag, subsystem = _TAGS.get(key, ("restart", None))
                result[key] = {
                    "status": "applied" if tag == "hot" else "restart_required",
                    "subsystem": subsystem,
                    "tag": tag,
                }
            self._save()
        return result

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._overrides = {}
            else:
                self._overrides.pop(key, None)
            self._save()
