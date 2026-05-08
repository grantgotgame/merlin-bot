# Skill Index

Quick reference for all available skills. The agent reads this to know what's available.

| Skill | File | Trigger | What it does |
|-------|------|---------|-------------|
| Checkpoint | `skills/checkpoint.md` | "Checkpoint" / "Save state" | Git commit + open loop capture + state update + rebuild Merlin briefing |
| Shift Change | `skills/shift-change.md` | "Shift change" / "Good morning" | Open a new shift, confirm energy + Thing |
| Rise | `lifestyle/rituals/RISE.md` | "Rise ritual" | Morning boot sequence (meds, water, move, eat, arrive) |
| Wrap | `lifestyle/rituals/WRAP.md` | "Wrap" / "Wind down" | End of day (checkpoint, set tomorrow's Thing, prep space, wind down) |

## Creating New Skills

1. Create a `.md` file in `skills/` (or `lifestyle/rituals/` for rituals)
2. Include: trigger phrase, step-by-step protocol, definition of "done"
3. Add it to this index
4. Tell the agent about it — or it'll find it here next session
