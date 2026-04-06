"""Merlin v2 — Voice output: Kokoro TTS + speaker EQ + go2rtc push."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

from event_bus import EventBus
import config

log = logging.getLogger("merlin.voice")


def apply_speaker_eq(audio_bytes: bytes) -> bytes:
    """Apply EQ optimized for the Amcrest camera's tiny speaker."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0",
             "-af", (
                 "highpass=f=200,"
                 "lowpass=f=3800,"
                 "equalizer=f=300:width_type=o:width=2:g=-3,"
                 "equalizer=f=2500:width_type=o:width=2:g=4,"
                 "equalizer=f=3200:width_type=o:width=2:g=2,"
                 "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2,"
                 "loudnorm=I=-16:LRA=7:TP=-1.5"
             ),
             "-f", "mp3", "pipe:1"],
            input=audio_bytes, capture_output=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return audio_bytes
    except Exception:
        return audio_bytes


def get_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of audio in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", "pipe:0", "-show_entries", "format=duration",
             "-v", "quiet", "-of", "csv=p=0"],
            input=audio_bytes, capture_output=True, timeout=5,
        )
        return float(result.stdout.strip())
    except Exception:
        # Fallback estimate: MP3 at ~6KB/s
        return max(len(audio_bytes) / 6000, 1.0)


class Voice:
    """Voice output module. Implements the Module contract."""

    def __init__(self):
        self._bus = None
        self._tts_model = None
        self._lock = threading.Lock()

    def start(self, bus: EventBus, cfg=None) -> None:
        self._bus = bus
        bus.on("speak", self._on_speak)
        bus.on("speak_nonverbal", self._on_speak_nonverbal)
        self._load_tts()

    def stop(self) -> None:
        if self._bus:
            self._bus.off("speak", self._on_speak)
            self._bus.off("speak_nonverbal", self._on_speak_nonverbal)

    def is_alive(self) -> bool:
        return True  # Voice is event-driven (no background thread to die)

    def _load_tts(self):
        """Load Kokoro TTS model."""
        try:
            from mlx_audio.tts import generate as _  # test import
            log.info(f"Kokoro TTS ready (voice: {config.KOKORO_VOICE})")
        except ImportError:
            log.warning("mlx-audio TTS not available — voice will be silent")

    def _on_speak(self, text: str = "") -> None:
        """Handle speak event — generate TTS and push to speaker."""
        if not text:
            return
        threading.Thread(
            target=self._speak_thread, args=(text,), daemon=True, name="speak"
        ).start()

    def _on_speak_nonverbal(self, sound: str = "") -> None:
        """Play a pre-recorded sound file."""
        if not sound:
            return
        sound_path = config.SOUNDS_DIR / f"{sound}.mp3"
        if sound_path.exists():
            threading.Thread(
                target=self._play_file, args=(sound_path,), daemon=True, name="nonverbal"
            ).start()
        else:
            log.warning(f"Sound not found: {sound_path}")

    def _speak_thread(self, text: str) -> None:
        """Generate TTS and push to camera speaker. Runs in a thread."""
        with self._lock:  # only one utterance at a time
            try:
                audio = self._generate_tts(text)
                if not audio:
                    log.warning(f"TTS failed, would say: {text}")
                    self._bus.emit("speak_failed")
                    return

                # Skip EQ — Mac speakers don't need camera speaker optimization
                self._bus.emit("speaking_started")
                self._push_to_speaker(audio)  # afplay blocks until done
                self._bus.emit("speaking_finished")

            except Exception:
                log.exception("Speak error")
                self._bus.emit("speak_failed")
                self._bus.emit("speaking_finished")

    def _play_file(self, path: Path) -> None:
        """Play a pre-recorded file through the camera speaker."""
        with self._lock:
            try:
                audio = path.read_bytes()
                self._bus.emit("speaking_started")
                self._push_to_speaker(audio)  # afplay blocks until done
                self._bus.emit("speaking_finished")
            except Exception:
                log.exception(f"Play file error: {path}")
                self._bus.emit("speaking_finished")

    def _generate_tts(self, text: str) -> bytes | None:
        """Generate speech audio from text using Kokoro via mlx-audio."""
        try:
            if self._tts_model is None:
                from mlx_audio.tts.generate import load_model
                self._tts_model = load_model("prince-canuma/Kokoro-82M")
                log.info("Kokoro model loaded")

            # Generate audio — returns a generator of GenerationResult chunks
            import numpy as np
            import mlx.core as mx

            # Clean text for TTS — strip newlines, extra spaces, special chars
            clean_text = " ".join(text.replace("\n", " ").split())
            clean_text = clean_text.strip()
            if not clean_text:
                return None

            audio_chunks = []
            sample_rate = 24000
            for chunk in self._tts_model.generate(text=clean_text, voice=config.KOKORO_VOICE):
                audio_arr = np.array(chunk.audio, dtype=np.float32) if not isinstance(chunk.audio, np.ndarray) else chunk.audio
                audio_chunks.append(audio_arr)
                sample_rate = chunk.sample_rate

            if not audio_chunks:
                return None

            audio_data = np.concatenate(audio_chunks)
            # Convert float32 to int16 PCM, then to MP3
            pcm = (audio_data * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            result = subprocess.run(
                ["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
                 "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                input=pcm, capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                log.info(f"TTS generated ({len(result.stdout)} bytes)")
                return result.stdout

            log.warning("ffmpeg MP3 conversion failed")
            return None

        except ImportError:
            log.warning("mlx-audio not available for TTS")
            return None
        except Exception:
            log.exception("TTS generation error")
            return None

    def _push_to_speaker(self, audio_bytes: bytes) -> None:
        """Play audio on Nate's Mac speakers via afplay.

        Simple, reliable, instant. No SCP, no Pi, no go2rtc speaker push.
        Future: switch back to camera speaker when hardware is better.
        """
        import uuid
        filename = f"merlin_speak_{uuid.uuid4().hex[:8]}.mp3"
        local_path = f"/tmp/{filename}"

        try:
            with open(local_path, "wb") as f:
                f.write(audio_bytes)

            # afplay is macOS built-in, plays immediately, blocks until done
            result = subprocess.run(
                ["afplay", local_path],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                log.warning(f"afplay failed: {result.stderr.decode()[:100]}")
            else:
                log.info("Speaker: played via Mac speakers")

        except subprocess.TimeoutExpired:
            log.warning("afplay timed out")
        except Exception:
            log.exception("Speaker error")
        finally:
            try:
                Path(local_path).unlink()
            except Exception:
                pass


# ── Standalone test ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[voice] %(message)s")
    bus = EventBus()
    bus.on("speaking_started", lambda: print(">>> Speaking..."))
    bus.on("speaking_finished", lambda: print(">>> Done speaking."))

    voice = Voice()
    voice.start(bus)

    import sys
    text = " ".join(sys.argv[1:]) or "Morning."
    print(f'Saying: "{text}"')
    bus.emit("speak", text=text)

    time.sleep(10)  # wait for playback
    print("Test complete.")
