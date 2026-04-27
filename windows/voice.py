"""Text-to-speech via Kokoro ONNX, played through the default audio output (BT speaker)."""

import time
import numpy as np
import sounddevice as sd
import config


class Voice:
    def __init__(self, bus=None):
        self.bus = bus
        self.tts = None
        self.sample_rate = 24000
        print("[voice] Loading Kokoro TTS...")
        try:
            import kokoro_onnx
            import os

            model_file = "kokoro-v1.0.onnx"
            voices_file = "voices-v1.0.bin"

            if not os.path.exists(model_file) or not os.path.exists(voices_file):
                print("[voice] Kokoro model files not found.")
                print(f"[voice] Download these two files into the merlin folder:")
                print(f"[voice]   1. {model_file}")
                print(f"[voice]   2. {voices_file}")
                print(f"[voice] From: https://github.com/thewh1teagle/kokoro-onnx/releases")
                print("[voice] TTS disabled until model files are present.")
                return

            self.tts = kokoro_onnx.Kokoro(model_file, voices_file)
            print(f"[voice] Kokoro loaded. Voice: {config.KOKORO_VOICE}")
        except ImportError:
            print("[voice] kokoro-onnx not installed. Run: pip install kokoro-onnx")
        except Exception as e:
            print(f"[voice] Failed to load Kokoro: {e}")

    def _emit(self, event, **kwargs):
        if self.bus is not None:
            try:
                self.bus.emit(event, **kwargs)
            except Exception:
                pass

    def speak(self, text):
        """Generate speech and play it through the BT speaker."""
        if not self.tts or not text:
            return

        voice = config.KOKORO_VOICE
        speed = config.KOKORO_SPEED
        device = config.SPEAKER_DEVICE

        t0 = time.time()
        self._emit("tts_start", text=text, voice=voice, speed=speed)
        try:
            samples, sample_rate = self.tts.create(text, voice=voice, speed=speed)
            self.sample_rate = int(sample_rate)
            duration = float(len(samples)) / float(sample_rate)
            # Emit a small downsampled envelope so the orb can lip-sync.
            try:
                arr = np.asarray(samples, dtype=np.float32)
                bins = 64
                if len(arr) >= bins:
                    chunks = np.array_split(np.abs(arr), bins)
                    envelope = [float(c.mean()) for c in chunks]
                else:
                    envelope = [float(np.abs(arr).mean())]
                self._emit("tts_envelope", envelope=envelope, duration=duration)
            except Exception:
                pass

            sd.play(samples, samplerate=sample_rate, device=device)
            sd.wait()
            self._emit("tts_complete",
                       latency_ms=int((time.time() - t0) * 1000),
                       duration=duration)
        except Exception as e:
            print(f"[voice] Playback error: {e}")
            self._emit("tts_failed", error=str(e))
