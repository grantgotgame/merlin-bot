"""Merlin v2 — Integration test: audio → brain → voice.

Say "Hey Merlin" and hear a response through the camera speaker.
Ctrl+C to stop.
"""

import logging
import time

from event_bus import EventBus
from audio_pipeline import AudioPipeline
from voice import Voice
from brain import Brain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

bus = EventBus()

audio = AudioPipeline()
voice = Voice()
brain = Brain()

audio.start(bus)
voice.start(bus)
brain.start(bus)

print("\n" + "=" * 50)
print("Merlin v2 — Audio Loop Test")
print("Say 'Hey Merlin' followed by anything.")
print("Ctrl+C to stop.")
print("=" * 50 + "\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    audio.stop()
    print("\nStopped.")
