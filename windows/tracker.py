"""
Face detection via YuNet + PTZ tracking via OpenCV DirectShow.
Runs in its own thread. Fires callbacks on face_arrived / face_lost.
"""

import cv2
import os
import subprocess
import time
import urllib.request
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    FACE_CONFIDENCE, YUNET_MODEL,
    PTZ_ENABLED, PTZ_SPEED, PTZ_DEADZONE,
)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceTracker:
    def __init__(self, bus=None):
        self.bus = bus
        self._running = False
        self._face_present = False
        self._face_lost_time = None

        # JPEG of the most recent frame (for MJPEG endpoint)
        self.latest_jpeg: bytes | None = None
        self._last_jpeg_t = 0.0
        self._last_box_emit = 0.0

        # Callbacks — set by merlin.py
        self.on_face_arrived = None
        self.on_face_lost = None

        self._download_failed = False
        self._ensure_model()
        self._init_camera()
        if not self._download_failed:
            self._init_detector()
            self._test_ptz()
        else:
            self.detector = None
            self.ptz_available = False

    def _ensure_model(self):
        """Download YuNet face detection model if missing."""
        if os.path.exists(YUNET_MODEL):
            return
        print("[tracker] Downloading YuNet face detection model...")
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(YUNET_URL)
            with urllib.request.urlopen(req, context=ctx) as resp, open(YUNET_MODEL, "wb") as f:
                f.write(resp.read())
            print("[tracker] YuNet model downloaded.")
        except Exception as e:
            print(f"[tracker] Auto-download failed: {e}")
            print(f"[tracker] Download manually from:")
            print(f"[tracker]   {YUNET_URL}")
            print(f"[tracker] Save as: {YUNET_MODEL} in the merlin folder.")
            print("[tracker] Face tracking disabled until model file is present.")
            self._download_failed = True

    def _find_camera(self):
        """
        Auto-detect EMEET/PIXY/Piko camera index.
        Uses Windows WMI enumeration order (matches DirectShow index).
        Falls back to first working camera, then to CAMERA_INDEX / 0.
        """
        if CAMERA_INDEX is not None:
            return CAMERA_INDEX

        # Ask Windows which cameras are connected, in registration order.
        # WMI enumeration order == DirectShow index on most Windows setups.
        try:
            result = subprocess.run(
                ["powershell", "-NonInteractive", "-Command",
                 "Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue "
                 "| Where-Object { $_.Status -eq 'OK' } "
                 "| Select-Object -ExpandProperty FriendlyName"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
                for idx, name in enumerate(names):
                    if any(k in name.lower() for k in ("emeet", "pixy", "piko")):
                        print(f"[tracker] Auto-detected EMEET camera: '{name}' (index {idx})")
                        return idx
        except Exception as e:
            print(f"[tracker] WMI camera probe failed: {e}")

        # Fallback: probe cv2 indices and return first that opens
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                print(f"[tracker] Camera auto-detected at index {i}")
                return i
            cap.release()

        print("[tracker] No camera detected -- defaulting to index 0")
        return 0

    def _init_camera(self):
        """Open the EMEET camera via DirectShow."""
        idx = self._find_camera()
        print(f"[tracker] Opening camera index {idx}...")
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[tracker] Camera opened: {w}x{h}")
        else:
            print("[tracker] WARNING: Camera failed to open!")
            print("[tracker] Try changing CAMERA_INDEX in config.py (0, 1, or 2).")

    def _init_detector(self):
        """Initialize YuNet face detector."""
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL,
            "",
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            FACE_CONFIDENCE,
            0.3,   # NMS threshold
            5000,  # top_k
        )

    def _test_ptz(self):
        """Check if PTZ controls work via DirectShow."""
        self.ptz_available = False
        if not PTZ_ENABLED:
            print("[tracker] PTZ disabled in config.")
            return

        if not self.cap.isOpened():
            return

        try:
            pan = self.cap.get(cv2.CAP_PROP_PAN)
            tilt = self.cap.get(cv2.CAP_PROP_TILT)
            # Try a write — if it doesn't throw, PTZ is available
            self.cap.set(cv2.CAP_PROP_PAN, pan)
            self.ptz_available = True
            print(f"[tracker] PTZ available (pan={pan}, tilt={tilt})")
        except Exception:
            pass

        if not self.ptz_available:
            print("[tracker] PTZ not available via DirectShow — face tracking is visual only.")
            print("[tracker] Camera will still detect faces but won't physically follow them.")

    def _move_ptz(self, face_cx, face_cy, frame_w, frame_h):
        """Nudge PTZ toward the detected face center."""
        if not self.ptz_available:
            return

        # Normalized offset from frame center (-1 to 1)
        offset_x = (face_cx - frame_w / 2) / (frame_w / 2)
        offset_y = (face_cy - frame_h / 2) / (frame_h / 2)

        # Ignore small offsets (deadzone)
        if abs(offset_x) < PTZ_DEADZONE and abs(offset_y) < PTZ_DEADZONE:
            return

        try:
            cur_pan = self.cap.get(cv2.CAP_PROP_PAN)
            cur_tilt = self.cap.get(cv2.CAP_PROP_TILT)
            self.cap.set(cv2.CAP_PROP_PAN, cur_pan + offset_x * PTZ_SPEED * 10)
            self.cap.set(cv2.CAP_PROP_TILT, cur_tilt - offset_y * PTZ_SPEED * 10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop — run this in a daemon thread
    # ------------------------------------------------------------------

    def _emit(self, event, **kwargs):
        if self.bus is not None:
            try:
                self.bus.emit(event, **kwargs)
            except Exception:
                pass

    def _maybe_publish_jpeg(self, frame, face_box=None):
        """Encode the current frame to JPEG ~15 fps for the MJPEG stream."""
        now = time.time()
        if now - self._last_jpeg_t < 0.066:
            return
        self._last_jpeg_t = now
        try:
            annotated = frame
            if face_box is not None:
                fx, fy, fw, fh = face_box
                annotated = frame.copy()
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (80, 220, 120), 2)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.latest_jpeg = bytes(buf)
        except Exception:
            pass

    def run(self):
        """Continuous face tracking loop."""
        self._running = True
        face_lost_timeout = 8.0
        print("[tracker] Face tracking active.")

        while self._running:
            if not self.cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if self.detector is None:
                self._maybe_publish_jpeg(frame)
                time.sleep(1)
                continue

            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            face_box = None
            if faces is not None and len(faces) > 0:
                best = max(faces, key=lambda f: f[-1])
                fx, fy, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
                conf = float(best[-1])
                cx, cy = fx + fw // 2, fy + fh // 2
                face_box = (fx, fy, fw, fh)

                self._move_ptz(cx, cy, w, h)

                # Throttle face_box emissions to ~5 Hz
                now = time.time()
                if now - self._last_box_emit > 0.2:
                    self._last_box_emit = now
                    self._emit("face_box",
                               x=fx, y=fy, w=fw, h=fh,
                               frame_w=w, frame_h=h, conf=conf)

                if not self._face_present:
                    self._face_present = True
                    self._face_lost_time = None
                    self._emit("face_arrived")
                    if self.on_face_arrived:
                        self.on_face_arrived()
            else:
                if self._face_present:
                    if self._face_lost_time is None:
                        self._face_lost_time = time.time()
                    elif time.time() - self._face_lost_time > face_lost_timeout:
                        self._face_present = False
                        self._face_lost_time = None
                        self._emit("face_lost")
                        if self.on_face_lost:
                            self.on_face_lost()

            self._maybe_publish_jpeg(frame, face_box)
            time.sleep(1.0 / CAMERA_FPS)

    def stop(self):
        """Stop tracking and release camera."""
        self._running = False
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
