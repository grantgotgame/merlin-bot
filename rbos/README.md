# RBOS — Rebel-Builder Operating System

A life operating system built on plain text files and AI agents.

## What This Is

RBOS is a structured markdown file system that an AI coding agent (like Claude Code) can read and write. It holds your state, your plans, your rituals, your memory — everything you need to run your life, organized so both you and the AI can work with it.

If you have executive functioning deficit, this is designed for you. The AI does the structural work. You do the living.

## Quick Start

1. Open this folder in Claude Code (or your preferred AI coding agent)
2. The agent will read `CLAUDE.md` automatically — that's the entrypoint
3. Tell the agent your name, your energy level, and what you're working on
4. The agent will help you fill in `core/STATE.md` and get started

## Directory Structure

```
core/           ← Operating files (STATE, RUNTIME, BOOTSTRAP)
briefing/       ← One file per day — your morning briefing
lifestyle/      ← Rituals, health, journal
health/         ← Health monitoring
skills/         ← Agent workflows (checkpoint, shift-change, etc.)
memory/         ← Long-term context, people, projects
inbox/          ← Unsorted input — drop anything here
queue/          ← Job queue for night shift
workspace/      ← Active drafts & working documents
log/            ← Daily entries, sprint logs
product/        ← Product docs, roadmap
archive/        ← Old versions, completed work
scripts/        ← Shell scripts, utilities
```

## Naming Convention

- **System files** (stable, canonical): `UPPER_CASE.md`
- **Content files** (dated): `YYYY-MM-DD_kebab-case.md`
- **Working docs** (undated): `kebab-case.md`

## The Philosophy

Rebel — because if you have executive functioning deficit, the default way of doing things doesn't work. You have to change things.

Builder — because we build our way out. That's what people like us do.
