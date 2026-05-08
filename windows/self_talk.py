"""Merlin's internal monologue for self-diagnosis.

Wizard principle (Hero safety first, then power, then self-improvement):
when something in his environment degrades, Merlin reasons through it
*to himself* via a private LLM call, proposes ONE concrete action from a
whitelist, applies it if safe, then watches the next health check to see
if it helped. The Hero sees the result, not the reasoning — Merlin tells
him "I lowered the mic threshold from 0.07 to 0.04 because the noise
floor was high" only when an action was actually taken.

This is NOT general agency. The action whitelist is small, every value
is clamped, and nothing destructive is reachable. Speaking aloud, sending
messages, modifying the filesystem, spending money — none of these are
options the self-talk loop can pick. If the diagnostic suggests an
action outside the whitelist, Merlin records the suggestion in the log
and surfaces it to the Hero as a regular system_message.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

import config
import requests

# Poll cadence and trigger thresholds
_POLL_INTERVAL_S = 30.0          # how often to check audio health
_DEGRADED_RUNS_TO_TRIGGER = 2    # consecutive bad checks before self-talk fires
_SELF_TALK_COOLDOWN_S = 120.0    # don't spam — wait this long between sessions
_MAX_TURNS_PER_SESSION = 3       # cap reasoning so a stuck loop can't run wild

# Action whitelist — every key MUST be reviewed for Hero-safety before adding.
# Each entry is (clamp_low, clamp_high). Values outside the range are clamped,
# never rejected — the LLM doesn't get to refuse.
_ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "set_energy_threshold": (0.02, 0.15),
}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


_SELF_TALK_SYSTEM = """You are Merlin, talking to yourself in private. The Hero cannot hear this.

You have noticed your hearing is degraded. Reason through what might be wrong using ONLY the metrics provided. Be brief — this is your inner voice, not a speech.

After reasoning, output exactly ONE action on the last line, in this format:
  ACTION: <name> <value>
or, if no safe action applies:
  ACTION: report_only

Available actions (the only ones that will work):
  set_energy_threshold <float>   — set VAD onset/silence threshold. Sensible
                                   values run 0.02 (very sensitive) to 0.15
                                   (very strict). The mic's noise floor is in
                                   the metrics — pick a value ~2x the floor.
  report_only                    — no action; surface the diagnosis to the
                                   Hero and stop.

Do not invent other actions. Do not speak to the Hero in this monologue —
that comes later, only if you took an action."""


class SelfTalk:
    """Background loop that watches audio health and runs internal monologues.

    Wired up by merlin.py once subsystems are ready. Lives for the lifetime
    of the Merlin process. Holds soft references to brain/audio/stt and the
    bus — passes them in lazily because subsystems load asynchronously.
    """

    def __init__(self, get_audio: Callable, get_stt: Callable, get_brain: Callable, bus: Any):
        self._get_audio = get_audio
        self._get_stt = get_stt
        self._get_brain = get_brain
        self._bus = bus
        self._running = False
        self._thread: threading.Thread | None = None
        self._consecutive_degraded = 0
        self._last_session_t = 0.0
        # Public state for /api/health surfacing.
        self.last_session: dict[str, Any] | None = None

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="merlin-self-talk")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── core loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        # Initial settle window — the audio-health watcher needs ~30s of
        # samples before its diagnoses are meaningful.
        time.sleep(45)
        while self._running:
            try:
                self._tick()
            except Exception as e:
                print(f"[self-talk] loop error: {e}")
            time.sleep(_POLL_INTERVAL_S)

    def _tick(self) -> None:
        from merlin_health import check_audio
        audio = self._get_audio()
        stt = self._get_stt()
        if not audio or not stt:
            return

        health = check_audio(audio, stt)
        if health.get("ok"):
            self._consecutive_degraded = 0
            return

        self._consecutive_degraded += 1
        if self._consecutive_degraded < _DEGRADED_RUNS_TO_TRIGGER:
            return
        if time.time() - self._last_session_t < _SELF_TALK_COOLDOWN_S:
            return

        self._last_session_t = time.time()
        self._run_session(health)

    # ── session: one LLM monologue + at most one action ─────────

    def _run_session(self, initial_health: dict) -> None:
        print(f"[self-talk] degraded for {self._consecutive_degraded} polls — entering monologue")
        if self._bus:
            self._bus.emit("self_talk_start", reason=initial_health.get("message"))

        transcript: list[dict] = []
        action_taken: dict | None = None

        for turn in range(_MAX_TURNS_PER_SESSION):
            metrics_blob = json.dumps(initial_health.get("metrics", {}), indent=2)
            user = (
                f"Health diagnosis (turn {turn+1}):\n"
                f"  severity: {initial_health.get('severity')}\n"
                f"  message: {initial_health.get('message')}\n"
                f"  suggested action (from heuristic): {initial_health.get('action')}\n"
                f"  metrics:\n{metrics_blob}"
            )
            monologue = self._llm(user, transcript)
            if not monologue:
                print("[self-talk] LLM returned nothing; aborting")
                break
            transcript.append({"role": "user", "content": user})
            transcript.append({"role": "assistant", "content": monologue})
            print(f"[self-talk] T{turn+1}: {monologue}")

            action = self._parse_action(monologue)
            if not action:
                continue

            if action["name"] == "report_only":
                action_taken = action
                break

            ok = self._apply_action(action)
            action_taken = {**action, "ok": ok}
            if not ok:
                # If the action couldn't be applied (out of bounds, unknown
                # name), let the model try once more with the failure noted.
                from merlin_health import check_audio
                initial_health = check_audio(self._get_audio(), self._get_stt())
                continue
            # Action applied — break and let the next poll evaluate.
            break

        self.last_session = {
            "started_at": self._last_session_t,
            "turns": len(transcript) // 2,
            "action": action_taken,
            "monologue": [t["content"] for t in transcript if t["role"] == "assistant"],
        }
        if self._bus:
            self._bus.emit("self_talk_done", action=action_taken,
                          turns=len(transcript) // 2)

        # Surface the result to the Hero only when something actually changed.
        if action_taken and action_taken.get("name") and action_taken["name"] != "report_only":
            msg = self._summarise_action(action_taken, initial_health)
            print(f"[self-talk] → Hero: {msg}")
            if self._bus:
                self._bus.emit("system_message", text=msg, level="info")
        elif action_taken and action_taken.get("name") == "report_only":
            # Pass the diagnostic through without claiming Merlin fixed it.
            if self._bus:
                self._bus.emit("system_message",
                               text=initial_health.get("message", "Hearing degraded."),
                               level="warn")

    # ── llm + parser ─────────────────────────────────────────────

    def _llm(self, user_msg: str, prior_transcript: list[dict]) -> str:
        """Private LLM call. Uses the same LM Studio endpoint the brain uses,
        but with a different system prompt and no exemplars or RBOS context —
        we want raw reasoning, not character voice."""
        try:
            messages = [{"role": "system", "content": _SELF_TALK_SYSTEM}]
            messages.extend(prior_transcript)
            messages.append({"role": "user", "content": user_msg})
            r = requests.post(
                config.LLM_URL,
                json={
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 250,
                    "temperature": 0.3,    # lower temp — we want consistent reasoning
                    "reasoning_effort": "low",
                    "stream": False,
                },
                timeout=120,
            )
            if not r.ok:
                print(f"[self-talk] LLM {r.status_code}: {r.text[:160]}")
                return ""
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"[self-talk] LLM error: {e}")
            return ""

    @staticmethod
    def _parse_action(monologue: str) -> dict | None:
        """Extract the trailing `ACTION: ...` line. Tolerant of whitespace
        and case. Returns None if no parseable action is found."""
        m = re.search(r"ACTION:\s*(\w+)(?:\s+([\-+]?\d*\.?\d+))?", monologue, re.IGNORECASE)
        if not m:
            return None
        name = m.group(1).lower()
        raw_value = m.group(2)
        try:
            value = float(raw_value) if raw_value is not None else None
        except ValueError:
            value = None
        return {"name": name, "value": value, "raw": m.group(0)}

    # ── action application (whitelist enforced here) ────────────

    def _apply_action(self, action: dict) -> bool:
        name = action.get("name")
        value = action.get("value")
        if name not in _ACTION_BOUNDS:
            print(f"[self-talk] ignoring unknown action: {action!r}")
            return False
        if value is None:
            print(f"[self-talk] action {name} missing value")
            return False
        clamped = _clamp(value, _ACTION_BOUNDS[name])
        if clamped != value:
            print(f"[self-talk] clamped {name} from {value} to {clamped}")
            action["clamped_to"] = clamped
        if name == "set_energy_threshold":
            old = config.ENERGY_THRESHOLD
            config.ENERGY_THRESHOLD = clamped
            action["from"] = old
            action["to"] = clamped
            print(f"[self-talk] ENERGY_THRESHOLD: {old} → {clamped}")
            return True
        return False

    @staticmethod
    def _summarise_action(action: dict, health: dict) -> str:
        name = action.get("name")
        if name == "set_energy_threshold":
            return (
                f"Tuned my mic threshold to {action.get('to'):.3f} "
                f"(was {action.get('from'):.3f}) — {health.get('message','noise floor was high.')}"
            )
        return f"Self-talk applied {name}."
