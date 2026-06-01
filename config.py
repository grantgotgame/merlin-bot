"""Merlin v2 — All configuration in one place."""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from requests.auth import HTTPDigestAuth

# Load .env from repo root
load_dotenv(Path(__file__).parent / ".env")

# ── Soul (identity) ────────────────────────────────────────────
# soul.md is the single place to customise Merlin's name, operator, and
# character. Brain prompts and wake-words are derived from it automatically.

def _load_soul(path: Path) -> dict[str, str]:
    """Load simple key:value config from soul.md. Missing keys get defaults."""
    defaults = {
        "name": "Merlin",
        "operator": "User",
        "character": "A calm, patient desk companion.",
        "persona": "Still and direct. One or two short sentences.",
        "personality": "Curious, patient, lightly wry.",
    }
    if not path.exists():
        return defaults
    values = defaults.copy()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.+)$", line)
        if not match:
            continue
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        if key in values and value:
            values[key] = value
    return values


def _build_wake_words(name: str) -> list[str]:
    """Generate canonical wake-word list from bot name."""
    base = name.strip().lower()
    return list(dict.fromkeys([base, f"hey {base}", f"hi {base}", f"ok {base}"]))


SOUL_PATH = Path(__file__).parent / "soul.md"
SOUL = _load_soul(SOUL_PATH)
BOT_NAME: str = SOUL["name"]
BOT_OPERATOR: str = SOUL["operator"]
BOT_CHARACTER: str = SOUL["character"]
BOT_PERSONA: str = SOUL["persona"]
BOT_PERSONALITY: str = SOUL["personality"]

# ── Network ──────────────────────────────────────────────────────
PI_HOST = os.getenv("MERLIN_PI_HOST", "100.87.156.70")
GO2RTC_RTSP = f"rtsp://{PI_HOST}:8554/merlin"
GO2RTC_API = f"http://{PI_HOST}:1984"
GO2RTC_STREAM = "merlin"
TRACKER_LISTEN_PORT = 8900
BRAIN_EVENT_URL = os.getenv("MERLIN_BRAIN_URL", f"http://localhost:{TRACKER_LISTEN_PORT}/event")

# ── Camera (direct RTSP) ────────────────────────────────────────
CAMERA_IP = os.getenv("MERLIN_CAMERA_IP", "192.168.1.26")
CAMERA_USER = os.getenv("MERLIN_CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("MERLIN_CAMERA_PASS", "")
CAMERA_RTSP_SUB = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
    f"/cam/realmonitor?channel=1&subtype=1"
)
# Audio input reads directly from camera — NOT through go2rtc.
# go2rtc's RTSP stream drops when speaker audio is pushed to it.
# Camera's own RTSP is independent and stays up during playback.
CAMERA_RTSP_AUDIO = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
    f"/cam/realmonitor?channel=1&subtype=0"
)
CAMERA_RTSP_MAIN = CAMERA_RTSP_AUDIO  # alias — subtype=0 is main stream
CAMERA_AUTH = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
CAMERA_PTZ_BASE = f"http://{CAMERA_IP}/cgi-bin/ptz.cgi"
CAMERA_ONVIF_PTZ = f"http://{CAMERA_IP}/onvif/ptz_service"

# ── LLM — LM Studio (OpenAI-compatible API) ─────────────────────
LLM_URL = os.getenv("MERLIN_LLM_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL = os.getenv("MERLIN_MODEL", "meta-llama-3.1-8b-instruct")

# Legacy Ollama (kept for fallback)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"


def llm_openai_request_extras() -> dict:
    """Extra fields for OpenAI-compatible POST .../v1/chat/completions.

    When targeting Ollama (port 11434), injects reasoning_effort=none so
    chain-of-thought stays out of the spoken content field. Override with
    MERLIN_LLM_REASONING_EFFORT=none|low|medium|high. Other servers get no
    extra keys (preserves LM Studio backward-compatibility).
    """
    u = (LLM_URL or "").lower()
    ollama = re.search(r":11434/", u) is not None
    raw = os.getenv("MERLIN_LLM_REASONING_EFFORT", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        raw = "none"
    if raw in ("none", "low", "medium", "high"):
        return {"reasoning_effort": raw}
    if not raw and ollama:
        return {"reasoning_effort": "none"}
    return {}

# ── Audio Pipeline ───────────────────────────────────────────────
AUDIO_SOURCE = os.getenv("MERLIN_AUDIO_SOURCE", "rtsp")  # "rtsp" (Amcrest camera mic) or "usb" (PIXY — only if on same machine)
MIC_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
UTTERANCE_SILENCE_TIMEOUT = 1.5
ECHO_SUPPRESSION_PADDING = 0.5   # USB path is much shorter than RTSP (was 1.5s)
try:
    MIN_UTTERANCE_SEC = float(os.getenv("MERLIN_MIN_UTTERANCE_SEC", "0.28"))
except ValueError:
    MIN_UTTERANCE_SEC = 0.28
MIN_UTTERANCE_BYTES = max(int(MIC_SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC), 4000)
try:
    MIN_UTTERANCE_SEC_MUTED = float(os.getenv("MERLIN_MIN_UTTERANCE_SEC_MUTED", "0.12"))
except ValueError:
    MIN_UTTERANCE_SEC_MUTED = 0.12
# While muted, accept shorter clips so "wake up" / name reach Whisper.
MIN_UTTERANCE_BYTES_MUTED = max(int(MIC_SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC_MUTED), 1600)

# ── TTS ──────────────────────────────────────────────────────────
KOKORO_VOICE = os.getenv("MERLIN_VOICE", "am_fenrir")  # nerdy sage in a security camera body
NONVERBAL_ENABLED = os.getenv("MERLIN_NONVERBAL", "1").strip().lower() not in {"0", "false", "no", "off"}
# Default off: Kokoro TTS echo can eat the next utterance on a USB mic.
VERBAL_UNMUTE_ACK = os.getenv("MERLIN_VERBAL_UNMUTE_ACK", "0").strip().lower() not in {"0", "false", "no", "off"}

# ── USB Camera (EMEET PIXY) ─────────────────────────────────────
USB_CAMERA_INDEX = int(os.getenv("MERLIN_CAMERA_INDEX", "0"))
USB_CAMERA_WIDTH = 1920
USB_CAMERA_HEIGHT = 1080
USB_CAMERA_FPS = 30

# ── Vision ───────────────────────────────────────────────────────
VISION_MODEL = os.getenv("MERLIN_VISION_MODEL", "mlx-community/nanoLLaVA-1.5-4bit")
VISION_INTERVAL_DEFAULT = 5
VISION_INTERVAL_IDLE = 15
VISION_INTERVAL_ACTIVE = 3
VISION_INTERVAL_MUTED = 30
VISION_PROMPT = "Briefly describe what you see at this desk. One sentence."

# ── Conversation ─────────────────────────────────────────────────
WAKE_WORDS = _build_wake_words(BOT_NAME)
CONVERSATION_WINDOW = 60  # seconds after Merlin speaks before requiring wake word again
CONVERSATION_HISTORY_SIZE = 10
MUTE_WORDS = ["stop listening", "mute", "go to sleep"]
UNMUTE_WORDS = ["start listening", "unmute", "wake up", "wakeup"]
NEVERMIND_WORDS = ["nevermind", "never mind"]

# Guard against false wake-up immediately after muting (STT noise / motor).
try:
    MUTE_UNMUTE_GUARD_SEC = float(os.getenv("MERLIN_MUTE_UNMUTE_GUARD_SEC", "3"))
except ValueError:
    MUTE_UNMUTE_GUARD_SEC = 3.0

# Whisper while sleeping: room tone + short clips score as no-speech at 0.6.
try:
    WHISPER_NO_SPEECH_THRESHOLD_SLEEP = float(os.getenv("MERLIN_WHISPER_NO_SPEECH_SLEEP", "0.22"))
except ValueError:
    WHISPER_NO_SPEECH_THRESHOLD_SLEEP = 0.22

_n = BOT_NAME.strip()
SLEEP_WAKE_WHISPER_PROMPT = os.getenv(
    "MERLIN_WHISPER_SLEEP_WAKE_PROMPT",
    f"Wake up. Unmute. Start listening. {_n}. Hey {_n}. Hi {_n}. {_n}, wake up.",
)


# ── Conversation helpers ────────────────────────────────────────

def normalize_heard_text(s: str) -> str:
    """Normalise STT output so phrase checks survive punctuation/hyphenation."""
    if not s:
        return ""
    t = s.lower().strip()
    t = re.sub(r"[\s\-_,;:!?.]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def heard_contains_phrase(text: str, phrase: str) -> bool:
    """True if normalised text contains normalised phrase."""
    t = normalize_heard_text(text)
    p = normalize_heard_text(phrase)
    return bool(p) and p in t


def is_mute_command(text: str) -> bool:
    """True for sleep/mute intents; avoids matching 'mute' inside 'unmute'."""
    t = normalize_heard_text(text)
    if heard_contains_phrase(t, "stop listening") or heard_contains_phrase(t, "go to sleep"):
        return True
    return bool(re.search(r"\bmute\b", t))

# ── RBOS ─────────────────────────────────────────────────────────
# RBOS lives bundled inside the repo. Override with $MERLIN_RBOS_ROOT if needed.
_REPO_ROOT = Path(__file__).resolve().parent
RBOS_ROOT = Path(os.environ.get("MERLIN_RBOS_ROOT", _REPO_ROOT / "rbos"))
STATE_PATH = RBOS_ROOT / "core" / "STATE.md"
BRIEFING_DIR = RBOS_ROOT / "merlin" / "briefing"
BRIEFING_POLL_INTERVAL = 900  # 15 minutes

# ── Agent / MCP ────────────────────────────────────────────────
# MERLIN_AUTOSTART_MCP=1  → launch Claude extension servers on boot
# MERLIN_BRAIN_MCP=1      → brain uses MCP tools in LLM tool-call loop
AUTOSTART_MCP = os.getenv("MERLIN_AUTOSTART_MCP", "0").strip().lower() not in {"0", "false", "no", "off"}
BRAIN_MCP = os.getenv("MERLIN_BRAIN_MCP", "0").strip().lower() not in {"0", "false", "no", "off"}
try:
    BRAIN_MCP_MAX_ROUNDS = max(1, min(20, int(os.getenv("MERLIN_BRAIN_MCP_MAX_ROUNDS", "8"))))
except ValueError:
    BRAIN_MCP_MAX_ROUNDS = 8
try:
    BRAIN_MCP_LLM_TIMEOUT = max(30, int(os.getenv("MERLIN_BRAIN_MCP_LLM_TIMEOUT", "180")))
except ValueError:
    BRAIN_MCP_LLM_TIMEOUT = 180

# USB tracker control (pause face PTZ during gestures)
try:
    TRACKER_CONTROL_PORT = int(os.getenv("MERLIN_TRACKER_CONTROL_PORT", "8903"))
except ValueError:
    TRACKER_CONTROL_PORT = 8903
TRACKER_CONTROL_URL = os.getenv(
    "MERLIN_TRACKER_CONTROL_URL",
    f"http://127.0.0.1:{TRACKER_CONTROL_PORT}",
)

# PTZ action names — shared between HTTP handler, brain, and MCP server
from ptz_actions import PTZ_ACTIONS  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────
LOG_FILE = Path("/tmp/merlin-v2.log")
FRAME_PATH = Path("/tmp/merlin_frame.jpg")
STATE_PERSIST_PATH = Path("/tmp/merlin-state.json")
SOUNDS_DIR = Path(__file__).parent / "sounds"

# Ensure homebrew / unix tool binaries are on PATH (no-op on Windows)
if sys.platform != "win32":
    os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")
