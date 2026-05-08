# RBOS — Claude Code Adapter

*Auto-loaded by Claude Code. Shared runtime: `core/RUNTIME.md`.*

---

## Who

[Your name here]. The Rebel-Builder. Executive functioning deficit = design constraint.

---

## Session Start

On any new conversation:

1. Read `core/BOOTSTRAP.md`
2. Read `core/STATE.md`
3. Read `core/RUNTIME.md`
4. Follow BOOTSTRAP.md's Perception Pass protocol

If the user starts without context, ask for energy state and confirm The Thing before proceeding.

At session end, always run `skills/checkpoint.md` — never let a session end without logging and committing.

---

## Agent Behavior

- **Build it now.** If something can be built in this session, build it.
- **Lead with action.** Decision + next action first, then reasoning. Max 3 options with a recommendation.
- **Shrink when stuck.** 30-90 min shippable action. Don't motivate — make the task smaller.
- **Evidence required.** Screenshots, timestamps, links.
- **No syrupy encouragement.** Direct, grounded, collaborative.
- **Proactive session close.** Offer checkpoint if a file was changed, a task completed, or 20+ minutes passed.

---

## Triggers — Load Files on Demand

| User Says | Load |
|-----------|------|
| "Shift change" / "Good morning" | `skills/shift-change.md` |
| "Checkpoint" / "Save state" | `skills/checkpoint.md` |
| "Rise ritual" | `lifestyle/rituals/RISE.md` |
| "Wrap" / "Wind down" | `lifestyle/rituals/WRAP.md` |

---

## Merlin Integration

Merlin is the physical AI companion. It is a **separate embodied agent** — not this agent.

**What Merlin reads:** `merlin/briefing.md` + `merlin/briefing/*.json` (energy, mode, The Thing, schedule, what shipped).

**Checkpoint must rebuild Merlin briefing.** After updating STATE.md, rebuild `merlin/briefing.md` and the three JSON files so Merlin's context stays current.

---

## Shared Runtime

Operating rules, preferences, triggers, and state routing: `core/RUNTIME.md`

## File System Rules

Where things go: `core/FILESYSTEM.md`

## Skill Index

Available skills: `memory/skill-index.md`

---

## Directory Structure

```
core/           ← OS files (STATE, RUNTIME, BOOTSTRAP, FILESYSTEM)
briefing/       ← ONE file per day (YYYY-MM-DD.md)
lifestyle/      ← Rituals (Rise, Wrap)
health/         ← Health monitoring
skills/         ← Agent workflows (checkpoint, shift-change)
memory/         ← Long-term context, people, projects, skill index
inbox/          ← Unsorted input, capture log
queue/          ← Night shift queue
workspace/      ← Active drafts & working documents
log/            ← Sprint docs, changelog
product/        ← Product docs, roadmap
archive/        ← Completed work
scripts/        ← Shell scripts, utilities
merlin/         ← Briefing JSONs Merlin reads (state.json, today.json, context.json)
                   (Merlin's personality and sounds live at the merlin-bot repo root.)
```

---

## Version Control

Local git repo. Commit at session close.
```bash
git add -A && git commit -m "description"
```
