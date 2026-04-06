#!/usr/bin/env python3
"""
Merlin Face Tracker v3 — YuNet detection + ContinuousMove velocity control.

Pi reflex loop. No LLM. Camera frame = feedback signal.
YuNet face detection (70ms, multi-angle, 93%+ confidence) replaces Haar cascade.
ContinuousMove sends smooth velocity proportional to face offset.
Threaded frame grabber ensures fresh frames (no stale buffer).

Run:  python3 tracker.py
Stop: Ctrl+C (returns camera to home)
"""

import cv2
import csv
import requests
from requests.auth import HTTPDigestAuth
import re
import time
import os
import sys
import signal
import threading
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────
from config import (CAMERA_IP, CAMERA_USER, CAMERA_PASS,
                    CAMERA_RTSP_SUB, CAMERA_ONVIF_PTZ)

RTSP_SUB = CAMERA_RTSP_SUB
ONVIF_PTZ = CAMERA_ONVIF_PTZ
PROFILE_TOKEN = "MediaProfile00000"

# YuNet model
YUNET_MODEL = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

# Tracking parameters
DEADBAND = 0.03
SPEED_FAST = 5.0            # snap hard for big moves
SPEED_FINE = 0.7            # gentle for small corrections
FINE_ZONE = 0.20            # enter fine mode earlier
MIN_VELOCITY = 0.12
FACE_LOST_TIMEOUT = 8.0

# Smoothing + PD control
SMOOTH_ALPHA = 0.7          # responsive but filters single-frame noise
KP = 1.0
KD = 0.7                    # heavy braking to kill oscillation
VELOCITY_RAMP = 0.8         # fast acceleration kept
VELOCITY_THRESHOLD = 0.03   # smaller = more responsive corrections, less coasting

# Axis mapping (empirically verified for this camera):
# ONVIF x=+0.1 → camera pans LEFT → face moves RIGHT in frame
# ONVIF y=+0.1 → camera tilts UP → face moves DOWN in frame
# So: face right of center → send POSITIVE x (pan left, face shifts left toward center)
#     face above center → send POSITIVE y (tilt up, face shifts down toward center)
# x=+0.3 → camera pans LEFT → face at right of frame shifts LEFT toward center → correct
# y=+0.3 → camera tilts UP → face below center shifts DOWN → wrong, need negative
PAN_SIGN = 1.0
TILT_SIGN = -1.0
# Wait — re-derive:
# face_y > 0.5 means face is below center. Need camera to tilt DOWN to bring face up.
# y=+0.1 tilts UP (wrong direction). y=-0.1 tilts DOWN (right direction).
# So: offset_y = (face_y - 0.5) needs to be sent as NEGATIVE to tilt down.
# TILT_SIGN = -1.0 ← correct

# ── Brain Notification Bridge ──────────────────────────────────

BRAIN_URL = getattr(config, 'BRAIN_EVENT_URL', "http://localhost:8900/event")
_last_notified = None

def notify_brain(event_type):
    global _last_notified
    if event_type == _last_notified:
        return
    _last_notified = event_type
    try:
        requests.post(BRAIN_URL, json={"type": event_type}, timeout=1)
    except Exception:
        pass


# State
running = True


# ── ONVIF PTZ ──────────────────────────────────────────────────

session = requests.Session()
session.auth = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
session.headers.update({"Content-Type": "application/soap+xml"})


def _soap(body):
    try:
        return session.post(ONVIF_PTZ, timeout=0.5, data=(
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
            ' xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"'
            ' xmlns:tt="http://www.onvif.org/ver10/schema">'
            f'<s:Body>{body}</s:Body></s:Envelope>'
        ))
    except Exception:
        return None


def ptz_move(pan_vel, tilt_vel):
    """ContinuousMove — smooth velocity control."""
    _soap(
        f'<tptz:ContinuousMove>'
        f'<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>'
        f'<tptz:Velocity><tt:PanTilt x="{pan_vel:.4f}" y="{tilt_vel:.4f}"/></tptz:Velocity>'
        f'</tptz:ContinuousMove>'
    )


def ptz_stop():
    _soap(
        f'<tptz:Stop>'
        f'<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>'
        f'<tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>false</tptz:Zoom>'
        f'</tptz:Stop>'
    )


def ptz_home():
    _soap(
        f'<tptz:GotoPreset>'
        f'<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>'
        f'<tptz:PresetToken>1</tptz:PresetToken>'
        f'</tptz:GotoPreset>'
    )


# ── Fresh Frame Grabber ────────────────────────────────────────

class FreshFrameGrabber:
    """Background thread drains RTSP buffer. get() returns latest frame."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()


# ── Face Detection (YuNet) ─────────────────────────────────────

yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (640, 480), 0.5, 0.3, 5000)


DETECT_SIZE = (320, 240)  # 8ms vs 50ms at 640x480

def detect_face(frame):
    """Detect largest face via YuNet at reduced resolution. Returns (cx, cy) normalized 0-1, or None."""
    small = cv2.resize(frame, DETECT_SIZE)
    yunet.setInputSize(DETECT_SIZE)
    _, faces = yunet.detect(small)

    if faces is None or len(faces) == 0:
        return None

    # Pick highest confidence face
    best = max(range(len(faces)), key=lambda i: faces[i][14])
    f = faces[best]
    cx = (f[0] + f[2] / 2) / DETECT_SIZE[0]
    cy = (f[1] + f[3] / 2) / DETECT_SIZE[1]
    return (cx, cy)


# ── Performance Logger ─────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


class TrackingLogger:
    """Logs every tracking cycle to CSV for analysis and self-tuning."""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.path = os.path.join(LOG_DIR, f"tracking-{date_str}.csv")
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)
        # Write header if file is empty
        if os.path.getsize(self.path) == 0:
            self.writer.writerow([
                "timestamp", "face_x", "face_y", "err_x", "err_y",
                "pan_vel", "tilt_vel", "speed_mode", "detect_ms",
            ])
        self.session_start = time.monotonic()
        self.moves = 0
        self.overshoots = 0
        self.prev_err_x = 0
        print(f"[tracker] Logging to {self.path}")

    def log(self, face_x, face_y, err_x, err_y, pan_vel, tilt_vel, speed_mode, detect_ms):
        self.writer.writerow([
            f"{time.monotonic() - self.session_start:.2f}",
            f"{face_x:.3f}", f"{face_y:.3f}",
            f"{err_x:.3f}", f"{err_y:.3f}",
            f"{pan_vel:.3f}", f"{tilt_vel:.3f}",
            speed_mode, f"{detect_ms:.1f}",
        ])
        self.moves += 1
        # Detect overshoot: error sign flipped from previous
        if self.prev_err_x * err_x < 0 and abs(err_x) > DEADBAND:
            self.overshoots += 1
        self.prev_err_x = err_x
        # Flush every 50 rows
        if self.moves % 50 == 0:
            self.file.flush()

    def summary(self):
        elapsed = time.monotonic() - self.session_start
        rate = self.moves / elapsed if elapsed > 0 else 0
        print(f"[tracker] Session: {elapsed:.0f}s, {self.moves} moves, {rate:.1f}/s, {self.overshoots} overshoots")

    def close(self):
        self.summary()
        self.file.close()


# ── Main Tracking Loop ─────────────────────────────────────────

def main():
    global running

    def shutdown(sig, frame):
        global running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Pre-warm ONVIF
    _soap(f'<tptz:GetStatus><tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken></tptz:GetStatus>')

    # Open RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    grabber = FreshFrameGrabber(RTSP_SUB)
    time.sleep(2)

    if grabber.get() is None:
        print("[tracker] ERROR: No frames")
        return

    print(f"[tracker] YuNet + ContinuousMove tracker")
    print(f"[tracker] Deadband={DEADBAND}, fast={SPEED_FAST}, fine={SPEED_FINE}, zone={FINE_ZONE}")
    print("[tracker] Running.")

    logger = TrackingLogger()
    face_lost_since = None
    is_tracking = False
    is_moving = False
    frame_count = 0
    last_log = 0

    # Smoothing state
    smooth_x = 0.5  # start at center
    smooth_y = 0.5
    prev_err_x = 0.0
    prev_err_y = 0.0
    current_pan_vel = 0.0
    current_tilt_vel = 0.0
    _last_sent_pan = 0.0
    _last_sent_tilt = 0.0

    try:
        while running:
            frame = grabber.get()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            t_detect = time.monotonic()
            face = detect_face(frame)
            detect_ms = (time.monotonic() - t_detect) * 1000

            if face is not None:
                raw_x, raw_y = face
                face_lost_since = None

                if not is_tracking:
                    # First detection — snap smoothing to current position
                    smooth_x = raw_x
                    smooth_y = raw_y
                    prev_err_x = raw_x - 0.5
                    prev_err_y = raw_y - 0.5
                    print(f"[tracker] Face acquired ({raw_x:.2f}, {raw_y:.2f})")
                    is_tracking = True
                    notify_brain("face_arrived")

                # 1. Exponential smoothing — filter jitter
                smooth_x = SMOOTH_ALPHA * raw_x + (1 - SMOOTH_ALPHA) * smooth_x
                smooth_y = SMOOTH_ALPHA * raw_y + (1 - SMOOTH_ALPHA) * smooth_y

                # 2. Calculate error from center
                err_x = smooth_x - 0.5
                err_y = smooth_y - 0.5

                # 3. Dead zone — stop when centered
                if abs(err_x) < DEADBAND and abs(err_y) < DEADBAND:
                    if is_moving:
                        ptz_stop()
                        is_moving = False
                        current_pan_vel = 0.0
                        current_tilt_vel = 0.0
                    prev_err_x = err_x
                    prev_err_y = err_y
                else:
                    # 4. PD controller — proportional + derivative braking
                    d_err_x = err_x - prev_err_x  # how fast error is changing
                    d_err_y = err_y - prev_err_y

                    # Two-speed gain: fast acquisition, gentle fine-tuning
                    dist = max(abs(err_x), abs(err_y))
                    speed = SPEED_FINE if dist < FINE_ZONE else SPEED_FAST

                    # PD output: proportional drives toward target, derivative brakes overshoot
                    target_pan = PAN_SIGN * (KP * err_x * speed + KD * d_err_x * speed)
                    target_tilt = TILT_SIGN * (KP * err_y * speed + KD * d_err_y * speed)

                    # 5. Velocity ramping — limit acceleration for smooth motion
                    pan_delta = target_pan - current_pan_vel
                    tilt_delta = target_tilt - current_tilt_vel

                    if abs(pan_delta) > VELOCITY_RAMP:
                        pan_delta = VELOCITY_RAMP if pan_delta > 0 else -VELOCITY_RAMP
                    if abs(tilt_delta) > VELOCITY_RAMP:
                        tilt_delta = VELOCITY_RAMP if tilt_delta > 0 else -VELOCITY_RAMP

                    current_pan_vel += pan_delta
                    current_tilt_vel += tilt_delta

                    # Apply minimum velocity floor
                    pan_vel = current_pan_vel
                    tilt_vel = current_tilt_vel
                    if 0 < abs(pan_vel) < MIN_VELOCITY:
                        pan_vel = MIN_VELOCITY if pan_vel > 0 else -MIN_VELOCITY
                    if 0 < abs(tilt_vel) < MIN_VELOCITY:
                        tilt_vel = MIN_VELOCITY if tilt_vel > 0 else -MIN_VELOCITY

                    # Clamp
                    pan_vel = max(-0.8, min(0.8, pan_vel))
                    tilt_vel = max(-0.8, min(0.8, tilt_vel))

                    # Suppress small axes
                    if abs(err_x) < DEADBAND:
                        pan_vel = 0.0
                    if abs(err_y) < DEADBAND:
                        tilt_vel = 0.0

                    # Only send PTZ command if velocity changed significantly
                    # Prevents motor micro-stutters from constant command restarts
                    pan_changed = abs(pan_vel - _last_sent_pan) > VELOCITY_THRESHOLD
                    tilt_changed = abs(tilt_vel - _last_sent_tilt) > VELOCITY_THRESHOLD
                    if pan_changed or tilt_changed or not is_moving:
                        ptz_move(pan_vel, tilt_vel)
                        _last_sent_pan = pan_vel
                        _last_sent_tilt = tilt_vel
                    is_moving = True

                    speed_mode = "fine" if dist < FINE_ZONE else "fast"
                    logger.log(smooth_x, smooth_y, err_x, err_y, pan_vel, tilt_vel, speed_mode, detect_ms)

                    prev_err_x = err_x
                    prev_err_y = err_y

                now = time.monotonic()
                if now - last_log > 2.0:
                    print(f"[tracker] face=({smooth_x:.2f},{smooth_y:.2f}) err=({err_x:+.2f},{err_y:+.2f}) vel=({current_pan_vel:+.2f},{current_tilt_vel:+.2f}) {'fine' if dist < FINE_ZONE else 'fast'}")
                    last_log = now

            else:
                if is_tracking:
                    if face_lost_since is None:
                        face_lost_since = time.monotonic()
                        ptz_stop()
                        is_moving = False
                        current_pan_vel = 0.0
                        current_tilt_vel = 0.0

                    if time.monotonic() - face_lost_since > FACE_LOST_TIMEOUT:
                        print("[tracker] Face lost → home")
                        ptz_home()
                        is_tracking = False
                        face_lost_since = None
                        notify_brain("face_lost")

            # ~15 FPS detection loop
            time.sleep(0.01)

    finally:
        print("[tracker] Shutting down...")
        ptz_stop()
        ptz_home()
        logger.close()
        grabber.release()


if __name__ == "__main__":
    main()
