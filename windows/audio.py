"""
Audio capture from EMEET Piko mic with energy-based voice activity detection.

Device selection:
  WASAPI is preferred because the Windows Audio Engine applies Acoustic Echo
  Cancellation (AEC) automatically, removing TTS speaker output from the mic
  signal. The InputStream is opened in AudioPipeline.__init__() — which always
  runs on the main thread — to satisfy WASAPI's COM single-threaded apartment
  requirement. (Opening from a daemon thread causes WdmSyncIoctl errors.)
  DirectSound and MME are used as fallbacks if WASAPI fails.

Silence detection uses an exponential moving average (EMA) of the RMS rather
than the raw instantaneous value. This makes the silence gate immune to brief
noise spikes (keyboard click, fan burst) that would otherwise keep resetting
the silence timer and preventing utterances from being emitted.
"""

import os
import sys
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from config import (
    PIXY_MIC_DEVICE, SAMPLE_RATE, CHANNELS,
    ENERGY_THRESHOLD, SILENCE_TIMEOUT,
    MIN_UTTERANCE_LENGTH, MAX_UTTERANCE_LENGTH,
)

_MIC_CHECK_WINDOW = 10.0   # seconds before warning about no speech detected
_NO_FRAMES_TIMEOUT = 5.0   # seconds before watchdog fires

# Onset threshold multiplier for hysteresis VAD.
# The onset threshold = ENERGY_THRESHOLD * _ONSET_MULTIPLIER.
# Frames in the "neutral zone" (between ENERGY_THRESHOLD and the onset
# threshold) are appended to the buffer but don't affect the silence timer —
# the timer runs through them uninterrupted. This prevents camera USB
# interference and brief ambient spikes from resetting silence detection.
_ONSET_MULTIPLIER = 2.0


def _resample(audio, src_rate, dst_rate):
    """Resample float32 audio using linear interpolation. No extra dependencies."""
    if src_rate == dst_rate:
        return audio
    target_len = int(round(len(audio) * dst_rate / src_rate))
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _check_windows_mic_privacy():
    """
    Read the Windows microphone privacy registry key.
    Returns True if desktop apps are allowed, False if blocked, None if unknown.
    The 'NonPackaged' subkey covers unpackaged desktop apps like python.exe.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion"
            r"\CapabilityAccessManager\ConsentStore\microphone\NonPackaged",
        )
        value, _ = winreg.QueryValueEx(key, "Value")
        winreg.CloseKey(key)
        return value == "Allow"
    except FileNotFoundError:
        return None   # Key absent -> not explicitly blocked (usually allowed)
    except Exception:
        return None


class AudioPipeline:
    def __init__(self):
        # Warn immediately if Windows is blocking mic access for desktop apps.
        allowed = _check_windows_mic_privacy()
        if allowed is False:
            print("[audio] (!) Windows is blocking microphone access for desktop apps!")
            print("[audio]    Fix: Settings -> Privacy & Security -> Microphone")
            print("[audio]          -> 'Let desktop apps access your microphone' -> On")

        self.mic_device, self._api_name = self._find_pixy_mic()
        dev_info = sd.query_devices(self.mic_device)
        self._mic_rate = int(dev_info["default_samplerate"])

        self.speech_queue = queue.Queue()
        self._buffer = []
        self._accumulating = False
        self._silence_start = None
        self._running = False
        self._suppressed = threading.Event()
        self._suppress_deadline = 0.0
        self._suppress_safety = False   # True = full safety timeout, False = normal settle
        # Diagnostic counters
        self._start_time = None
        self._frames_read = 0
        self._peak_rms = 0.0
        self._mic_check_done = False
        self._utterances_queued = 0

        # Open AND start the InputStream immediately on the main thread.
        # WASAPI requires COM STA context; __init__ always runs on the main
        # thread in merlin.py, satisfying that requirement.
        #
        # Critically: the stream must be STARTED before FaceTracker opens its
        # DirectShow camera. DirectShow's COM initialisation can break an
        # already-created-but-not-yet-started WASAPI stream, causing it to
        # deliver 0 frames. Starting first (then loading other components)
        # prevents this interference.
        #
        # Callbacks that arrive before start() sets _running=True are silently
        # discarded — the pipeline doesn't accumulate or emit during that window.
        self._stream = self._open_stream()
        self._stream.start()   # begin callbacks immediately

    # ------------------------------------------------------------------
    # Device selection — WASAPI first (for AEC), then DirectSound, then MME
    # ------------------------------------------------------------------

    def _find_pixy_mic(self):
        """
        Auto-detect EMEET Piko mic and return (device_index, api_name).

        Preference order:
          1. WASAPI  — Windows Audio Engine applies AEC, removing TTS echo
          2. DirectSound — no AEC but reliable from any thread
          3. MME — last resort
        """
        if PIXY_MIC_DEVICE is not None:
            info = sd.query_devices(PIXY_MIC_DEVICE)
            api = sd.query_hostapis(info["hostapi"])["name"]
            return PIXY_MIC_DEVICE, api

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        def _is_emeet(name):
            n = name.lower()
            return "emeet" in n or "pixy" in n or "piko" in n

        def _api_name(d):
            return hostapis[d["hostapi"]]["name"]

        # Gather all EMEET input devices
        emeet = [
            (i, d) for i, d in enumerate(devices)
            if d["max_input_channels"] > 0 and _is_emeet(d["name"])
        ]

        if not emeet:
            default = sd.default.device[0]
            print(f"[audio] EMEET mic not found. Using default input [{default}].")
            info = sd.query_devices(default)
            return default, sd.query_hostapis(info["hostapi"])["name"]

        # WASAPI first (gives us Windows AEC), then DirectSound, then MME
        for preference in ("wasapi", "directsound", "mme"):
            for i, d in emeet:
                if preference in _api_name(d).lower():
                    api = _api_name(d)
                    print(f"[audio] Found EMEET mic ({d['name']} via {api}): device [{i}]")
                    return i, api

        # Last resort: first EMEET device regardless of API
        i, d = emeet[0]
        api = _api_name(d)
        print(f"[audio] Found EMEET mic: [{i}] {d['name']}")
        return i, api

    # ------------------------------------------------------------------
    # Stream management — opened on main thread, started/stopped later
    # ------------------------------------------------------------------

    def _open_stream(self):
        """
        Create (but do not start) the PortAudio InputStream.
        Tries WASAPI first; falls back to DirectSound if it raises.
        Must be called from the MAIN THREAD so WASAPI COM init succeeds.
        """
        blocksize = int(self._mic_rate * 0.03)   # 30 ms

        try:
            stream = sd.InputStream(
                device=self.mic_device,
                samplerate=self._mic_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=blocksize,
                callback=self._audio_callback,
            )
            return stream
        except Exception as e:
            # If WASAPI failed (e.g. device locked), try DirectSound fallback
            if "wasapi" in self._api_name.lower():
                print(f"[audio] WASAPI open failed ({e}), trying DirectSound...")
                devices = sd.query_devices()
                hostapis = sd.query_hostapis()
                for i, d in enumerate(devices):
                    nm = d["name"].lower()
                    api = hostapis[d["hostapi"]]["name"].lower()
                    if (d["max_input_channels"] > 0
                            and any(k in nm for k in ("emeet", "pixy", "piko"))
                            and "directsound" in api):
                        self.mic_device = i
                        self._api_name = hostapis[d["hostapi"]]["name"]
                        self._mic_rate = int(d["default_samplerate"])
                        print(f"[audio] Fallback to DirectSound device [{i}]")
                        return sd.InputStream(
                            device=i,
                            samplerate=self._mic_rate,
                            channels=CHANNELS,
                            dtype="float32",
                            blocksize=int(self._mic_rate * 0.03),
                            callback=self._audio_callback,
                        )
            raise

    # ------------------------------------------------------------------
    # Callback-based capture
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback — called for every audio block."""
        if not self._running:
            return  # Discard frames that arrive before start() is called
        if status:
            print(f"[audio] {status}", end="\r")

        self._frames_read += 1
        self._process_frame(indata, frames)

    def _watchdog(self):
        """
        After NO_FRAMES_TIMEOUT seconds, check if frames are flowing.
        If not, diagnose: re-check privacy settings and list device alternatives.
        """
        time.sleep(_NO_FRAMES_TIMEOUT)
        if not self._running or self._frames_read > 0:
            return

        print(f"\n[audio] (!) No audio frames after {_NO_FRAMES_TIMEOUT:.0f}s "
              f"(device [{self.mic_device}] at {self._mic_rate}Hz).")

        allowed = _check_windows_mic_privacy()
        if allowed is False:
            print("[audio] (!) CAUSE: Windows microphone privacy is blocking python.exe")
            print("[audio]    Fix -> Settings -> Privacy & Security -> Microphone")
            print("[audio]          -> 'Let desktop apps access your microphone' -> On")
            print("[audio]    Then restart Merlin.")
        elif allowed is None:
            print("[audio] (!) Could not read Windows privacy key -- check manually:")
            print("[audio]    Settings -> Privacy & Security -> Microphone -> all toggles On")
        else:
            print("[audio] (!) Privacy settings look OK. Possible causes:")
            print("[audio]    - Another app holds exclusive mic access (Teams, Zoom, etc.)")
            print("[audio]    - Wrong device index -- try setting PIXY_MIC_DEVICE in config.py")
            print("[audio]    Available input devices:")
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    api = sd.query_hostapis(d["hostapi"])["name"]
                    print(f"[audio]      [{i}] {d['name']} ({api})")

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def _process_frame(self, indata, frames):
        """Diagnostics, suppression gate, hysteresis VAD."""
        audio = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if os.environ.get("MERLIN_DEBUG_AUDIO"):
            print(f"[audio] rms={rms:.4f}", end="\r")

        # Mic-level diagnostic — runs regardless of suppression state.
        if not self._mic_check_done and self._start_time is not None:
            self._peak_rms = max(self._peak_rms, rms)
            elapsed = time.time() - self._start_time
            if elapsed >= _MIC_CHECK_WINDOW:
                self._mic_check_done = True
                if self._utterances_queued == 0:
                    onset_thr = ENERGY_THRESHOLD * _ONSET_MULTIPLIER
                    print(
                        f"\n[audio] (!) No speech detected in {_MIC_CHECK_WINDOW:.0f}s "
                        f"(frames={self._frames_read}, peak_rms={self._peak_rms:.4f}, "
                        f"onset_threshold={onset_thr:.3f})"
                    )
                    if self._peak_rms < onset_thr:
                        print("[audio] (!) Mic too quiet -- check Windows Sound Settings "
                              "-> EMEET Piko -> Input volume = 100%")
                    else:
                        print("[audio] (!) Mic level OK. Try speaking for 1-2 seconds "
                              "and say 'merlin'.")

        # Echo suppression with auto-timeout safety net.
        if self._suppressed.is_set():
            if time.time() > self._suppress_deadline:
                self._suppressed.clear()
                # Only log when the safety timeout fires (unexpected hang).
                # Normal settle expirations (_suppress_safety=False) are silent.
                if self._suppress_safety:
                    print("[audio] (!) Suppress safety timeout fired -- mic re-enabled")
                self._suppress_safety = False
            else:
                return

        # Hysteresis VAD — three zones:
        #
        #   rms > onset_threshold  → LOUD: start/continue accumulating.
        #                            Silence timer is reset only on NEW onset.
        #
        #   silence_thr < rms <= onset_thr  → NEUTRAL: already accumulating?
        #                            Append to buffer but leave silence timer
        #                            untouched. Camera USB spikes land here;
        #                            they neither advance nor reset the timer.
        #
        #   rms <= silence_threshold → QUIET: advance silence timer.
        #                            After SILENCE_TIMEOUT, emit utterance.
        #
        # Using two thresholds prevents brief noise bursts from resetting
        # silence detection without using a slow EMA that adds latency.

        onset_threshold = ENERGY_THRESHOLD * _ONSET_MULTIPLIER

        if rms > onset_threshold:
            # Loud onset
            if not self._accumulating:
                self._accumulating = True
                self._buffer = []
                self._silence_start = None
            self._buffer.append(audio)

        elif self._accumulating and rms > ENERGY_THRESHOLD:
            # Neutral zone — just buffer, don't touch silence timer
            self._buffer.append(audio)

        elif self._accumulating:
            # Quiet — advance silence timer
            self._buffer.append(audio)
            if self._silence_start is None:
                self._silence_start = time.time()
            elif time.time() - self._silence_start > SILENCE_TIMEOUT:
                self._emit_utterance()

        # Force-cut very long utterances
        if self._accumulating and self._buffer:
            duration = len(self._buffer) * frames / self._mic_rate
            if duration >= MAX_UTTERANCE_LENGTH:
                self._emit_utterance()

    def _emit_utterance(self):
        """Resample buffered audio to SAMPLE_RATE Hz and push to speech queue."""
        if self._buffer:
            raw = np.concatenate(self._buffer)
            utterance = _resample(raw, self._mic_rate, SAMPLE_RATE)
            duration = len(utterance) / SAMPLE_RATE
            if duration >= MIN_UTTERANCE_LENGTH:
                self.speech_queue.put(utterance)
                self._utterances_queued += 1
        self._buffer = []
        self._accumulating = False
        self._silence_start = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        # Stream is already running (started in __init__).
        # Flipping _running=True tells _audio_callback to begin real work.
        self._running = True
        self._start_time = time.time()
        threading.Thread(target=self._watchdog, daemon=True).start()
        print(f"[audio] Capturing from device [{self.mic_device}] "
              f"({self._api_name}) at {self._mic_rate}Hz")
        if self._mic_rate != SAMPLE_RATE:
            print(f"[audio] Resampling {self._mic_rate}Hz -> {SAMPLE_RATE}Hz for STT")

    def stop(self):
        self._running = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def get_utterance(self, timeout=0.1):
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def suppress(self, timeout=15.0):
        """Suppress mic input while Merlin is speaking. Auto-clears after timeout."""
        self._suppress_deadline = time.time() + timeout
        self._suppress_safety = True   # flag: log if this timeout fires naturally
        self._suppressed.set()

    def unsuppress(self, settle=1.0):
        """
        Resume mic input after Merlin finishes speaking.

        settle: extra seconds to stay suppressed, giving room echo and mic AGC
        time to settle before VAD starts accumulating again.  Default 1.0s.
        Pass settle=0 to unsuppress immediately (e.g. when no TTS was played).
        """
        self._buffer = []
        self._accumulating = False
        self._silence_start = None
        if settle > 0:
            # Extend deadline; _process_frame auto-clears when it expires.
            self._suppress_deadline = time.time() + settle
            self._suppress_safety = False   # normal settle — no log on expiry
            self._suppressed.set()
        else:
            self._suppress_safety = False
            self._suppressed.clear()
