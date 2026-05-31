"""
Merlin sound effects — plays the wood clave .wav palette from sounds/.

All sounds are the pre-generated wooden clave files in sounds/ at the repo
root (one level above this windows/ directory). See sounds/SOUND_DESIGN.md
and sounds/generate_clave.py for the full design rationale.

Design principles:
  - One organic source (wood clave) → shared character across all sounds.
  - Pools: where multiple variations exist, a random file is chosen each call
    so the same sound never plays twice in a row (Vector/Gabaldon principle).
  - No synthesized sine waves. No Windows system beeps.

Conversation sound flow:
  [Merlin engaged]   → listening()   — open.wav     (C→G ascending fifth)
  [TTS finishes]     → ready()       — ready.wav    (E→F "your turn")
  [Mute/nevermind]   → acknowledged() — close.wav   (G→C descending fifth)
  [Face arrives]     → greeting()    — greeting.wav (D-E-F ascending)
  [Boot complete]    → startup()     — startup.wav  (full C-D-E-F-G scale)
  [Error]            → error()       — error.wav    (dissonant low flam)
"""

import random
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from config import SPEAKER_DEVICE

# sounds/ directory is one level above this windows/ directory
_SOUNDS_DIR = Path(__file__).parent.parent / "sounds"


# ── Playback core ─────────────────────────────────────────────────────────────


def _load_wav(path: Path) -> tuple[np.ndarray, int] | None:
    """Load a WAV file and return (float32 samples, sample_rate).

    Handles 16- and 32-bit PCM, mono or stereo (stereo is mixed to mono).
    Returns None on any error so callers can fall through gracefully.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0
        elif sw == 4:
            pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
        else:
            print(f"[sounds] Unsupported sample width {sw} in {path.name}")
            return None

        if n_ch == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)

        return pcm, rate

    except Exception as e:
        print(f"[sounds] Failed to load {path.name}: {e}")
        return None


def _play(path: Path) -> bool:
    """Play a single .wav file synchronously through the configured speaker.

    Returns True on success, False if the file is missing or playback fails.
    Uses the same SPEAKER_DEVICE as voice.py so sounds and TTS share output.
    """
    if not path.exists():
        print(f"[sounds] Missing: {path.name} (run sounds/generate_clave.py to regenerate)")
        return False

    result = _load_wav(path)
    if result is None:
        return False

    samples, rate = result
    try:
        sd.play(samples, samplerate=rate, device=SPEAKER_DEVICE)
        sd.wait()
        return True
    except Exception as e:
        print(f"[sounds] Playback error ({path.name}): {e}")
        return False


def _pool(*filenames: str) -> bool:
    """Pick a random file from a list and play it.

    Shuffles first so the first pick is random — satisfies the "never the
    same sound twice in a row" principle from SOUND_DESIGN.md.
    Returns True if any file played successfully.
    """
    candidates = [_SOUNDS_DIR / f for f in filenames]
    present = [p for p in candidates if p.exists()]
    if not present:
        # None of the pool files exist — fall back silently
        return False
    random.shuffle(present)
    return _play(present[0])


# ── Named sounds ──────────────────────────────────────────────────────────────
#
# These match the event names used by the Mac voice.py (speak_nonverbal) and
# the SOUND_DESIGN.md conversation flow, so the two platforms are in sync.


def listening():
    """C → G ascending fifth: "I'm engaged, I heard you."

    Plays right after brain.process() confirms Merlin will respond.
    Ideal placement would be before the LLM call (so the user hears the cue
    during the thinking wait), but that requires splitting brain.process().
    See TODO note in merlin.py _tick().
    """
    _play(_SOUNDS_DIR / "open.wav")


def ready():
    """E → F: "Your turn to speak." Plays after TTS finishes.

    This is the handoff cue that closes each response cycle.
    """
    _play(_SOUNDS_DIR / "ready.wav")


def acknowledged():
    """G → C descending fifth: "Understood, going quiet."

    Used for mute and nevermind commands — mirrors the CLOSE event in
    SOUND_DESIGN.md. The descending fifth is the exact inverse of listening().
    """
    _play(_SOUNDS_DIR / "close.wav")


def greeting():
    """D-E-F ascending inner notes: "Hello, I see you."

    Plays when face arrives and Merlin speaks a verbal greeting.
    Uses the inner notes (not the sacred fifth) since this is inside a
    context, not a hard state-change.
    """
    _play(_SOUNDS_DIR / "greeting.wav")


def startup():
    """C-D-E-F-G full ascending scale: all five notes, boot complete.

    The only named preset where all five notes play in sequence. Signals
    that Merlin has finished loading and is ready to listen.
    """
    _play(_SOUNDS_DIR / "startup.wav")


def error():
    """Low dissonant flam (C4+D4): something went wrong.

    Designed as a grace-note pair — slightly ugly, clearly not a normal cue.
    Used when subsystems fail or an unrecoverable error occurs.
    """
    _play(_SOUNDS_DIR / "error.wav")


def sad():
    """Descending arc: empathetic/commiserating tone."""
    _play(_SOUNDS_DIR / "sad.wav")


def thinking():
    """F-F-F rhythmic tapping: "I'm processing."

    Draws from a small pool (C/E/G rhythmic variants) so it varies across
    calls. Currently called AFTER brain.process() returns (i.e., after
    thinking is already done), which is incorrect timing — see TODO in
    merlin.py _tick(). Do not add latency by calling this in the hot path
    until the timing is fixed.
    """
    _pool("n3_rhy_C.wav", "n3_rhy_E.wav", "n3_rhy_G.wav")
