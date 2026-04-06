"""Merlin v2 — USB Audio Stream Manager.

Drop-in replacement for the RTSP StreamManager in audio_pipeline.py.
Uses sounddevice to capture from the EMEET PIXY's USB microphone.

Usage in audio_pipeline.py:
    # Replace:  from audio_pipeline import StreamManager  # RTSP
    # With:     from audio_usb import USBStreamManager as StreamManager

Or set config.AUDIO_SOURCE = "usb" to auto-select.
"""

import logging
import queue
import threading
import time
import os

import numpy as np

log = logging.getLogger("merlin.audio")

# Default sample rate — must match what Silero VAD and Parakeet expect
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms at 16kHz — matches RTSP StreamManager


def find_pixy_audio():
    """Find the EMEET PIXY audio input device index."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev.get("name", "").upper()
            if ("EMEET" in name or "PIXY" in name) and dev["max_input_channels"] > 0:
                return i
        # No EMEET found — use default input
        default = sd.default.device[0]
        if default is not None and default >= 0:
            log.warning(f"EMEET not found, using default input device {default}")
            return int(default)
    except Exception as e:
        log.error(f"Cannot enumerate audio devices: {e}")
    return None


class USBStreamManager:
    """Captures PCM audio from USB mic via sounddevice.

    API-compatible with the RTSP StreamManager:
    - start() / stop()
    - read_chunks() generator yielding PCM byte chunks
    """

    CHUNK_SAMPLES = CHUNK_SAMPLES

    def __init__(self, device_index=None):
        self._device_index = device_index
        self._running = False
        self._stream = None
        self._queue = queue.Queue(maxsize=200)  # ~6 seconds of buffered audio
        self._overflow_count = 0

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            if "overflow" in str(status).lower():
                self._overflow_count += 1
                if self._overflow_count % 100 == 1:
                    log.debug(f"Audio overflow ({self._overflow_count} total)")
            else:
                log.debug(f"Audio status: {status}")

        # Convert float32 to int16 PCM bytes (matches RTSP StreamManager output)
        pcm_int16 = (indata[:, 0] * 32767).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass  # Drop oldest — prevents buildup

    def read_chunks(self):
        """Generator yielding PCM byte chunks. Compatible with RTSP StreamManager."""
        import sounddevice as sd

        while self._running:
            # Find device if not set
            if self._device_index is None:
                self._device_index = find_pixy_audio()
                if self._device_index is None:
                    log.error("No USB audio device found — waiting 5s")
                    time.sleep(5)
                    continue

            try:
                device_info = sd.query_devices(self._device_index)
                dev_name = device_info.get("name", f"device {self._device_index}")
                log.info(f"USB audio: opening {dev_name}")

                self._stream = sd.InputStream(
                    device=self._device_index,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=self.CHUNK_SAMPLES,
                    callback=self._audio_callback,
                )
                self._stream.start()
                log.info("USB audio: streaming")
                self._overflow_count = 0

                # Yield chunks from the queue
                while self._running and self._stream.active:
                    try:
                        pcm = self._queue.get(timeout=1.0)
                        yield pcm
                    except queue.Empty:
                        continue

                log.warning("USB audio stream ended")

            except Exception:
                log.exception("USB audio error")
                self._device_index = None  # Re-detect on next attempt

            finally:
                if self._stream:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None

            if self._running:
                log.info("Reconnecting USB audio in 2s")
                time.sleep(2)


# ── Standalone Test ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[audio-usb] %(message)s")

    print("Testing USB audio capture...")
    print(f"Looking for EMEET PIXY...")

    idx = find_pixy_audio()
    if idx is None:
        print("ERROR: No audio input device found")
        exit(1)

    import sounddevice as sd
    info = sd.query_devices(idx)
    print(f"Using: {info['name']} (index {idx})")

    mgr = USBStreamManager(device_index=idx)
    mgr.start()

    print("Capturing 3 seconds of audio...")
    chunk_count = 0
    rms_total = 0

    start = time.time()
    for pcm in mgr.read_chunks():
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        rms_total += rms
        chunk_count += 1

        if chunk_count % 50 == 0:
            print(f"  Chunk {chunk_count}: RMS={rms:.0f}")

        if time.time() - start > 3.0:
            break

    mgr.stop()

    avg_rms = rms_total / chunk_count if chunk_count > 0 else 0
    print(f"\nCaptured {chunk_count} chunks in 3 seconds")
    print(f"Average RMS: {avg_rms:.0f}")
    print(f"{'Audio is working!' if avg_rms > 10 else 'Very quiet — check mic connection'}")
