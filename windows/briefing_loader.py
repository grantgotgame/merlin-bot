"""RBOS briefing loader — single source of truth for both brains.

Reads `rbos/merlin/briefing/{state,today,context}.json` and falls back to
`rbos/core/STATE.md` parsing. Returns a plain text block ready to drop
into a system prompt.

The wizard principle here: Merlin always knows what's going on with the
Hero — energy, focus, schedule, recent mood — without the Hero having to
re-explain every conversation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("merlin.briefing")


def repo_root() -> Path:
    """Resolve the merlin-bot repo root from this module's location."""
    here = Path(__file__).resolve().parent
    # If we're at repo root, here/rbos exists. If we're under windows/,
    # here.parent/rbos exists.
    if (here / "rbos").exists():
        return here
    if (here.parent / "rbos").exists():
        return here.parent
    # Best-effort fallback: assume we're at repo root.
    return here


def rbos_root() -> Path:
    """Allow MERLIN_RBOS_ROOT override for tests / alternate deployments."""
    override = os.environ.get("MERLIN_RBOS_ROOT")
    if override:
        return Path(override)
    return repo_root() / "rbos"


def briefing_dir() -> Path:
    return rbos_root() / "merlin" / "briefing"


def state_md_path() -> Path:
    return rbos_root() / "core" / "STATE.md"


def _parse_state_md(text: str) -> dict:
    """Pull the_thing/energy/mode/shift out of STATE.md. Empty values dropped."""
    out: dict = {}
    for line in text.split("\n"):
        for key, header in (
            ("the_thing", "**The Thing:**"),
            ("energy", "**Energy:**"),
            ("mode", "**Mode:**"),
            ("shift", "**Current Shift:**"),
        ):
            if line.startswith(header):
                val = line.replace(header, "").strip()
                if val and not (val.startswith("[") and val.endswith("]")):
                    out[key] = val
    return out


def _rebuild_briefing_from_state() -> None:
    """Cold-start safety net: if briefing JSONs are missing or stale vs
    STATE.md, synthesize minimal versions so Merlin always has context. The
    canonical writer is still rbos/skills/checkpoint.md — this only fires
    when checkpoint hasn't run yet."""
    sp = state_md_path()
    if not sp.exists():
        return
    bd = briefing_dir()
    state_file = bd / "state.json"
    today_file = bd / "today.json"
    context_file = bd / "context.json"
    state_mtime = sp.stat().st_mtime
    needs_rebuild = (
        not state_file.exists()
        or not today_file.exists()
        or not context_file.exists()
        or state_file.stat().st_mtime < state_mtime
    )
    if not needs_rebuild:
        return
    try:
        parsed = _parse_state_md(sp.read_text(encoding="utf-8"))
        bd.mkdir(parents=True, exist_ok=True)
        state_doc = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "the_thing": parsed.get("the_thing", ""),
            "energy": parsed.get("energy", "green"),
            "shift": parsed.get("shift", ""),
            "mode": parsed.get("mode", ""),
            "week_focus": "",
        }
        state_file.write_text(json.dumps(state_doc, indent=2), encoding="utf-8")
        if not today_file.exists():
            today_file.write_text(json.dumps({"schedule": [], "shipped": [], "open_loops": []}, indent=2), encoding="utf-8")
        if not context_file.exists():
            context_file.write_text(json.dumps({"mood_history": [], "crash_risk_signals": [], "notes": ""}, indent=2), encoding="utf-8")
        log.info(f"Rebuilt briefing JSONs from {sp}")
    except Exception as e:
        log.warning(f"Briefing rebuild failed: {e}")


def load_briefing_context() -> str:
    """Return a plain-text block summarising what Merlin knows about the
    Hero, ready to drop into a system prompt. Empty string if no briefing
    data is available."""
    _rebuild_briefing_from_state()
    parts: list = []
    bd = briefing_dir()
    state_file = bd / "state.json"
    today_file = bd / "today.json"
    context_file = bd / "context.json"

    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if data.get("the_thing"):
                parts.append(f"Today's focus: {data['the_thing']}")
            if data.get("energy"):
                parts.append(f"Energy: {data['energy']}")
            if data.get("mode"):
                parts.append(f"Mode: {data['mode']}")
            if data.get("shift"):
                parts.append(f"Shift: {data['shift']}")
            if data.get("week_focus"):
                parts.append(f"This week: {data['week_focus']}")
        except Exception as e:
            log.debug(f"state.json read error: {e}")

    if today_file.exists():
        try:
            data = json.loads(today_file.read_text(encoding="utf-8"))
            if data.get("shipped"):
                parts.append(f"Shipped today: {', '.join(data['shipped'][:5])}")
            if data.get("schedule"):
                parts.append(f"Schedule: {', '.join(data['schedule'][:3])}")
            if data.get("open_loops"):
                parts.append(f"Open loops: {', '.join(data['open_loops'][:3])}")
        except Exception as e:
            log.debug(f"today.json read error: {e}")

    if context_file.exists():
        try:
            data = json.loads(context_file.read_text(encoding="utf-8"))
            if data.get("mood_history"):
                latest = data["mood_history"][-1]
                parts.append(f"Recent mood: {latest.get('mindset', 'unknown')}")
            if data.get("stems_to_try"):
                parts.append(f"Stem to try: {data['stems_to_try'][0]}")
        except Exception as e:
            log.debug(f"context.json read error: {e}")

    # Fallback to STATE.md if briefing JSONs were empty.
    if not parts:
        try:
            state = state_md_path().read_text(encoding="utf-8")
            for line in state.split("\n"):
                if line.startswith("**The Thing:**"):
                    parts.append(f"Today's focus: {line.replace('**The Thing:**', '').strip()}")
                elif line.startswith("**Energy:**"):
                    parts.append(f"Energy: {line.replace('**Energy:**', '').strip()}")
                elif line.startswith("**Mode:**"):
                    parts.append(f"Mode: {line.replace('**Mode:**', '').strip()}")
                elif line.startswith("**Current Shift:**"):
                    parts.append(f"Shift: {line.replace('**Current Shift:**', '').strip()}")
        except FileNotFoundError:
            pass
        except Exception as e:
            log.debug(f"STATE.md fallback error: {e}")

    if parts:
        return "What you know about the Hero:\n" + "\n".join(f"- {p}" for p in parts)
    return ""
