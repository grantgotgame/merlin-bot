"""
Speech-to-text via faster-whisper with CUDA acceleration.

On Windows, Python 3.8+ no longer searches site-packages for DLLs. NVIDIA pip
packages install their DLLs into site-packages/nvidia/<pkg>/bin/. We prepend
those directories to PATH (and also call os.add_dll_directory) before importing
ctranslate2/faster_whisper, so LoadLibrary finds them however it searches.

This registration must happen at module load time — before the faster_whisper
import — so it runs before ctranslate2 is initialised.
"""

import os
import sys
import site
import subprocess
import threading
import numpy as np


def _register_nvidia_dll_dirs():
    """
    Find every site-packages/nvidia/<pkg>/bin directory and register it two ways:
      1. os.add_dll_directory  — for extensions that use LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
      2. prepend to PATH       — for ctranslate2's internal LoadLibrary calls, which use
                                 the standard search order (application dir -> PATH -> System32)
    Searches both system site-packages and the user site-packages, since pip
    falls back to --user installation when the system dir isn't writable.
    Safe to call multiple times; duplicate PATH entries are skipped.
    """
    if sys.platform != "win32":
        return
    search_roots = list(site.getsitepackages())
    try:
        user_site = site.getusersitepackages()
        if user_site and user_site not in search_roots:
            search_roots.append(user_site)
    except Exception:
        pass
    found = []
    for sp in search_roots:
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            for subdir in ("bin", "lib"):
                d = os.path.join(nvidia_root, pkg, subdir)
                if os.path.isdir(d):
                    found.append(d)
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
    if found:
        current = os.environ.get("PATH", "")
        new_dirs = [d for d in found if d not in current]
        if new_dirs:
            os.environ["PATH"] = os.pathsep.join(new_dirs) + os.pathsep + current


_register_nvidia_dll_dirs()

# Now safe to import — ctranslate2 will find CUDA DLLs via PATH when it needs them.
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE, WHISPER_LANGUAGE

_CUDA_PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
_GPU_LOAD_TIMEOUT = 60  # seconds before we give up waiting for GPU model load


def _is_missing_lib_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "not found", "cannot be loaded"))


def _load_whisper(device, compute_type):
    """
    Load WhisperModel, with a timeout when targeting GPU.
    GPU model loads can hang indefinitely if CUDA initialisation stalls —
    a daemon thread lets us detect that and fall back gracefully.
    """
    model_box, err_box = [None], [None]

    def _load():
        try:
            model_box[0] = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        except Exception as e:
            err_box[0] = e

    if device == "cpu":
        _load()  # CPU is fast; no timeout needed
    else:
        t = threading.Thread(target=_load, daemon=True)
        t.start()
        t.join(timeout=_GPU_LOAD_TIMEOUT)
        if t.is_alive():
            raise TimeoutError(
                f"GPU model load timed out after {_GPU_LOAD_TIMEOUT}s "
                "(likely a missing or incompatible CUDA library)"
            )

    if err_box[0]:
        raise err_box[0]
    return model_box[0]


class STT:
    def __init__(self, bus=None):
        self.bus = bus
        print(f"[stt] Loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
        self._on_cpu = (WHISPER_DEVICE == "cpu")

        try:
            self.model = _load_whisper(WHISPER_DEVICE, WHISPER_COMPUTE)
        except Exception as e:
            if not self._on_cpu:
                print(f"[stt] GPU load failed: {e}")
                self._switch_to_cpu()
            else:
                raise

        # Smoke test: ctranslate2 allocates GPU resources lazily, so the model
        # can load without error even when a CUDA DLL is missing. Force a real
        # inference now to surface any remaining issues at boot.
        if not self._on_cpu:
            self._smoke_test()

        print(f"[stt] Whisper loaded ({'CPU' if self._on_cpu else 'GPU'}).")

    def _bounded_inference(self, audio, timeout=30):
        """Run inference on a daemon thread with a hard timeout. ctranslate2's
        LoadLibrary path can hang silently if a CUDA DLL is wedged; we don't
        want that to block the whole boot."""
        result_box, err_box = [None], [None]

        def _go():
            try:
                result_box[0] = self._run_inference(audio)
            except Exception as e:
                err_box[0] = e

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise TimeoutError(f"GPU inference timed out after {timeout}s")
        if err_box[0]:
            raise err_box[0]
        return result_box[0]

    def _smoke_test(self):
        """Force a real GPU inference at boot; auto-install and retry if it fails.
        Bounded by a timeout so a wedged DLL load can't hang Merlin forever."""
        try:
            self._bounded_inference(np.zeros(1600, dtype=np.float32), timeout=30)
        except Exception as e:
            print(f"[stt] GPU smoke test failed: {e}")
            if _is_missing_lib_error(e) and self._auto_install_cuda():
                print("[stt] Reloading model on GPU...")
                try:
                    self.model = _load_whisper(WHISPER_DEVICE, WHISPER_COMPUTE)
                    self._bounded_inference(np.zeros(1600, dtype=np.float32), timeout=30)
                    print("[stt] GPU working.")
                    return
                except Exception as e2:
                    print(f"[stt] GPU still failing after install: {e2}")
            self._switch_to_cpu()

    def _auto_install_cuda(self):
        """pip-install missing NVIDIA packages, then re-register DLL paths."""
        print(f"[stt] Installing missing GPU libraries: {', '.join(_CUDA_PACKAGES)}")
        print("[stt] Downloading from PyPI — this takes about a minute...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + _CUDA_PACKAGES,
                timeout=300,
            )
            if result.returncode != 0:
                print("[stt] pip install returned a non-zero exit code.")
                return False
        except subprocess.TimeoutExpired:
            print("[stt] Install timed out after 5 minutes.")
            return False
        except Exception as e:
            print(f"[stt] Auto-install error: {e}")
            return False

        _register_nvidia_dll_dirs()  # pick up newly installed DLL dirs
        print("[stt] GPU libraries installed and registered.")
        return True

    def _switch_to_cpu(self):
        """Last resort: swap the model to CPU."""
        print("[stt] Falling back to CPU (int8).")
        self.model = _load_whisper("cpu", "int8")
        self._on_cpu = True

    def _run_inference(self, audio):
        """Transcribe audio and consume the lazy segment generator."""
        segments, _ = self.model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            vad_filter=False,
        )
        return list(segments)

    def _emit(self, event, **kwargs):
        if self.bus is not None:
            try:
                self.bus.emit(event, **kwargs)
            except Exception:
                pass

    def transcribe(self, audio):
        """
        Transcribe a numpy float32 audio array to text.
        Returns the transcribed string, or None if nothing recognised.
        """
        import time as _t
        self._emit("stt_start")
        t0 = _t.time()
        try:
            segments = self._run_inference(audio)
            text = " ".join(s.text for s in segments).strip()
            text = text if text else None
            self._emit("stt_complete", text=text or "", latency_ms=int((_t.time() - t0) * 1000))
            return text
        except Exception as e:
            # Belt-and-suspenders: if a GPU error slips through after boot
            # (e.g. driver reset), fall back to CPU and retry once.
            if not self._on_cpu:
                print(f"[stt] GPU transcription failed mid-session: {e}")
                self._switch_to_cpu()
                try:
                    segments = self._run_inference(audio)
                    text = " ".join(s.text for s in segments).strip()
                    return text if text else None
                except Exception as e2:
                    print(f"[stt] CPU fallback also failed: {e2}")
                    return None
            print(f"[stt] Transcription error: {e}")
            return None
