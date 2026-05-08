# Merlin-Bot — Claude Code Adapter

Merlin runs RBOS by default. The RBOS folder structure is bundled in this repo at `rbos/`.

When you open this repo in Claude Code, treat `rbos/CLAUDE.md` as the operating adapter:

1. Read `rbos/CLAUDE.md`
2. Read `rbos/core/BOOTSTRAP.md`
3. Read `rbos/core/STATE.md`
4. Read `rbos/core/RUNTIME.md`
5. Follow BOOTSTRAP.md's Perception Pass protocol

All RBOS path conventions in `rbos/CLAUDE.md` are relative to the `rbos/` folder. Code (Python modules, audio pipeline, web UI, agent) lives at the merlin-bot repo root.

## Bot vs OS

- **Bot code** — `main.py`, `brain.py`, `audio_pipeline.py`, `agent/`, `windows/`, `personality/`, `sounds/` (this is Merlin)
- **OS files** — `rbos/core/`, `rbos/skills/`, `rbos/memory/`, `rbos/merlin/briefing/`, etc. (this is RBOS)

The bot reads the OS via `config.RBOS_ROOT` (defaults to `<repo>/rbos`, override with `$MERLIN_RBOS_ROOT`).
