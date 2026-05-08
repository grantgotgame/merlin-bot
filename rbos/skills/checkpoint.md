# Skill: Checkpoint

*Save the game. Keep playing — or walk away. ~2 minutes.*

---

## Triggers

- "Checkpoint" / "Save state" / "Quick save"
- "Close session" / "Wrap it up"
- End of any productive session

---

## The Two Non-Negotiables

**If checkpoint does nothing else, it MUST do these two things:**

### A. Git Commit
Save all file changes. `git add -A && git commit`. This preserves state so any future agent can see where we left off.

### B. Open Loop Capture
Sweep the full conversation. Every "we should," every bug, every decision — capture it. Inbox, task manager, or sprint doc. One loop = one capture. If it's not captured, it disappears.

---

## The Full Process (when time permits)

**No questions. Extract from context. Do it all. Report when done.**

### Step 1: Update STATE.md

- **Sprint doc first:** Write session details to the active sprint doc (`log/SPRINT_W[N].md`) under today's section.
- **Last Active second:** Update the timestamp with a SHORT summary (one line). This is a headline, not a record.

### Step 2: Rebuild Merlin Briefing

Rebuild `merlin/briefing.md` and the three JSON files so Merlin's context stays current.

**Sources:** `core/STATE.md` + active sprint doc.

**Files to write:**
- `merlin/briefing.md` — human-readable summary (energy, Thing, what shipped, what's next, reminders)
- `merlin/briefing/state.json` — energy, The Thing, shift, mode
- `merlin/briefing/today.json` — schedule, shipped, open loops
- `merlin/briefing/context.json` — mood history, crash risk signals

Keep briefing.md under 50 lines. Pull real data. Do not fabricate.

### Step 3: Log Updates

Append a session block to `log/CHANGELOG.md`:
```
## [Date]
**Session — [start] → [end]:**
**What Shipped:** [items]
```

### Step 4: Capture

Add any new open loops to `inbox/CAPTURE_LOG.md`.

### Step 5: Git Commit
```bash
git add -A && git commit -m "checkpoint: [1-line summary]"
```

### Step 6: Report

Short. Three lines max:
```
**Checkpoint saved.** [what was captured]
**Next:** [what's happening next]
**All clear.** / **Blocked on X.**
```

---

## Rules

1. **Under 2 minutes** for regular checkpoint.
2. **No confirmation needed.** The trigger word is the confirmation.
3. **Don't interrupt flow.** Save and report. That's it.
4. **Idempotent.** Running twice shouldn't create duplicate entries.
