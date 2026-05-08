# Merlin — Your Executive Functioning Prosthetic

Merlin is the physical body of your RBOS system. A camera on your desk that can see you, hear you, and talk to you — with your full RBOS context.

## How Merlin Connects to RBOS

Merlin reads three JSON files from `merlin/briefing/` to know your current state. These get rebuilt by the checkpoint skill at every session close.

```
merlin/briefing/state.json   ← Energy, The Thing, shift, mode
merlin/briefing/today.json   ← Schedule, what shipped, open loops
merlin/briefing/context.json ← Mood history, crash risk signals
```

Merlin also reads `merlin/briefing.md` for a human-readable summary.

## Hardware

| Component | What | Cost |
|-----------|------|------|
| Camera | EMEET PIXY (PTZ webcam) | $95-150 |
| Tracking | Raspberry Pi 5 (face detection, PTZ control) | $80-200 |
| Brain | Your Mac or PC (16GB+ RAM) | (you have this) |
| LLM | Gemma 4 via LM Studio (local, free) | $0 |
| STT | Whisper (local, free) | $0 |
| TTS | Kokoro (local, free) | $0 |

**Minimum setup:** Just the camera + your computer. The Pi adds face tracking.
**Total: $0-350** depending on what you already own.

## Architecture

```
Camera (sees you, hears you, speaks to you)
    ↕
Raspberry Pi 5 (face tracking, voice ID, idle behavior)
    ↕
Your Mac/PC — The Brain
    ├ audio_pipeline.py  — VAD → STT
    ├ brain.py           — LLM + RBOS context
    ├ voice.py           — TTS → camera speaker
    ├ vision.py          — scene description
    └ event_bus.py       — connects everything
```

Everything runs locally. Nothing leaves your network.

## Personality

Merlin's personality lives in the top-level `personality/` folder of the merlin-bot repo. The default is minimal:
- Respond with the same energy and length you're spoken to with
- No lectures, no motivation, no guilt
- Observe and reflect

Customize this to make Merlin yours.

## Getting Started

Merlin and RBOS now ship in the same repo. See the top-level `README.md` and `windows/install.ps1` to bring up the bot — it auto-resolves these briefing files via `config.RBOS_ROOT`.
