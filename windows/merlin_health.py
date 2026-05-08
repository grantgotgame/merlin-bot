"""Self-checks for Merlin's runtime dependencies.

Wizard principle: Merlin notices when his environment is broken and either
fixes it himself or surfaces a plain-English problem with one specific action.
He does not silently degrade, and he does not ask the Hero to debug LM Studio.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger("merlin.health")

_TIMEOUT = 5.0


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/models"


def _load_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/models/load"


def _list_models(base_url: str) -> list[dict[str, Any]] | None:
    try:
        r = requests.get(_models_endpoint(base_url), timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("models", [])
    except (requests.RequestException, ValueError):
        return None


def _is_loaded(model: dict[str, Any]) -> bool:
    return bool(model.get("loaded_instances"))


def _first_loaded_llm(models: list[dict[str, Any]]) -> str | None:
    for m in models:
        if m.get("type") == "llm" and _is_loaded(m):
            return m.get("key")
    return None


def _post_load(base_url: str, model_key: str) -> bool:
    try:
        # NB: LM Studio v1's load endpoint rejects unknown JSON keys
        # ({"error":"Unrecognized key(s) in object: 'config'"}). Don't add
        # extra fields here unless you've verified the API accepts them.
        r = requests.post(
            _load_endpoint(base_url),
            json={"model": model_key},
            timeout=60.0,
        )
        return r.ok
    except requests.RequestException:
        return False


def _warm_up(base_url: str, model_key: str) -> None:
    """Fire one tiny inference to push the model past cold-start.
    Best-effort; failures are silent."""
    if not model_key:
        return
    try:
        chat_url = base_url.rstrip("/") + "/v1/chat/completions"
        requests.post(
            chat_url,
            json={
                "model": model_key,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": False,
            },
            timeout=120.0,
        )
    except requests.RequestException:
        pass


def check_llm(preferred_model: str, base_url: str) -> dict[str, Any]:
    """Inspect LM Studio without trying to fix anything.

    Returns: {ok, severity, message, action, loaded_model}
      severity: 'ok' | 'warn' | 'error'
      action:   one concrete next step, or None when ok
    """
    models = _list_models(base_url)
    if models is None:
        return {
            "ok": False,
            "severity": "error",
            "message": "LM Studio isn't responding.",
            "action": "Start LM Studio from the Start Menu, then restart Merlin.",
            "loaded_model": None,
        }

    by_key = {m.get("key"): m for m in models}
    target = by_key.get(preferred_model)

    if target is None:
        loaded = _first_loaded_llm(models)
        if loaded:
            return {
                "ok": False,
                "severity": "warn",
                "message": f"Preferred model '{preferred_model}' isn't downloaded. Using '{loaded}'.",
                "action": f"Download {preferred_model} via 'lms get {preferred_model}' or LM Studio's Discover tab.",
                "loaded_model": loaded,
            }
        return {
            "ok": False,
            "severity": "error",
            "message": f"Preferred model '{preferred_model}' isn't downloaded and nothing else is loaded.",
            "action": f"Download {preferred_model} via 'lms get {preferred_model}'.",
            "loaded_model": None,
        }

    if _is_loaded(target):
        return {
            "ok": True,
            "severity": "ok",
            "message": f"LM Studio serving '{preferred_model}'.",
            "action": None,
            "loaded_model": preferred_model,
        }

    return {
        "ok": False,
        "severity": "warn",
        "message": f"'{preferred_model}' is downloaded but not loaded.",
        "action": "Merlin will try to load it.",
        "loaded_model": _first_loaded_llm(models),
    }


def ensure_llm_ready(
    preferred_model: str,
    base_url: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Like check_llm, but actively load preferred_model if it isn't loaded.
    Falls back to whatever LLM is loaded if the load attempt fails.
    Never raises."""
    status = check_llm(preferred_model, base_url)
    if status["ok"]:
        # Already loaded — still warm it up to flush any cold weights.
        _warm_up(base_url, status["loaded_model"])
        return status

    # Can't reach LM Studio at all → nothing to do
    if status["severity"] == "error" and status["loaded_model"] is None and "isn't responding" in status["message"]:
        return status

    # Preferred model not downloaded → keep the fallback we already picked
    if "isn't downloaded" in status["message"]:
        return status

    # Preferred model downloaded but not loaded → try to load it
    log.info(f"Loading {preferred_model} into LM Studio...")
    if not _post_load(base_url, preferred_model):
        # Load request failed outright; keep any fallback we found
        return {
            "ok": False,
            "severity": "error",
            "message": f"LM Studio refused to load '{preferred_model}'.",
            "action": "Restart LM Studio if this persists.",
            "loaded_model": status["loaded_model"],
        }

    # Poll until loaded or timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        recheck = check_llm(preferred_model, base_url)
        if recheck["ok"]:
            _warm_up(base_url, preferred_model)
            return recheck

    return {
        "ok": False,
        "severity": "error",
        "message": f"Timed out waiting for '{preferred_model}' to load.",
        "action": "Check LM Studio's Developer Logs for load errors.",
        "loaded_model": _first_loaded_llm(_list_models(base_url) or []),
    }


def llm_base_url(llm_url: str) -> str:
    """Strip the OpenAI-style suffix off config.LLM_URL to get LM Studio's base."""
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1"):
        if llm_url.endswith(suffix):
            return llm_url[: -len(suffix)]
    return llm_url


# ── Audio + STT self-diagnosis ───────────────────────────────────────
#
# Wizard principle: when Merlin can't hear well he notices and tells the
# Hero what to fix, instead of silently mishearing forever. These thresholds
# are tuned for the EMEET PIXY at 1-2 ft away. Adjust if hardware changes.

_CLIP_RATE_BAD = 0.02        # >2% of frames clipping → input gain too high
_NOISE_FLOOR_HIGH = 0.04     # noise floor near onset threshold (0.05 default) → false triggers
_PEAK_LOW = 0.05             # never exceeded onset threshold → mic too quiet
_LOGPROB_BAD = -1.0          # avg over recent utterances; -1.0 ≈ very unsure
_NO_SPEECH_BAD = 0.6         # Whisper thinks audio wasn't speech
_RECENT_UTTERANCES_FOR_AVG = 5


def _audio_signals(audio_module: Any) -> dict[str, Any]:
    """Snapshot audio.py's runtime metrics into plain numbers."""
    history = list(getattr(audio_module, "_rms_history", []) or [])
    clip_count = int(getattr(audio_module, "_clip_count", 0))
    sample_count = int(getattr(audio_module, "_sample_count", 0)) or 1
    if history:
        peak = max(history)
        # Noise floor: median of the lowest quartile (robust to occasional speech bursts).
        sorted_h = sorted(history)
        q1 = sorted_h[: max(1, len(sorted_h) // 4)]
        noise_floor = sum(q1) / len(q1)
    else:
        peak = 0.0
        noise_floor = 0.0
    return {
        "samples": len(history),
        "peak_rms": float(peak),
        "noise_floor": float(noise_floor),
        "clip_rate": clip_count / sample_count,
    }


def _stt_signals(stt_module: Any) -> dict[str, Any]:
    """Snapshot recent Whisper confidence into a single dict."""
    recent = list(getattr(stt_module, "_recent_quality", []) or [])
    recent = recent[-_RECENT_UTTERANCES_FOR_AVG:]
    if not recent:
        return {"samples": 0, "avg_logprob": 0.0, "max_no_speech": 0.0}
    avg_logprob = sum(r["avg_logprob"] for r in recent) / len(recent)
    max_no_speech = max(r["no_speech_prob"] for r in recent)
    return {
        "samples": len(recent),
        "avg_logprob": float(avg_logprob),
        "max_no_speech": float(max_no_speech),
    }


def check_audio(audio_module: Any, stt_module: Any) -> dict[str, Any]:
    """Diagnose hearing problems. Returns one health dict per concern, or
    ok if everything looks fine. Severity tiers match check_llm:
      ok    — no action needed
      warn  — degraded, the Hero should know
      error — Merlin probably can't hear at all
    """
    a = _audio_signals(audio_module) if audio_module else {"samples": 0}
    s = _stt_signals(stt_module) if stt_module else {"samples": 0}

    # Need a few seconds of data before any check is meaningful.
    if a.get("samples", 0) < 30:
        return {
            "ok": True, "severity": "ok",
            "message": "Calibrating mic…", "action": None,
            "metrics": {"audio": a, "stt": s},
        }

    # 1. Clipping is the worst — Whisper can't recover from clipped input.
    if a["clip_rate"] > _CLIP_RATE_BAD:
        return {
            "ok": False, "severity": "error",
            "message": f"Mic is clipping ({a['clip_rate']:.0%} of frames at digital ceiling).",
            "action": "Lower input volume in Windows Sound Settings → EMEET Piko → Properties → Levels.",
            "metrics": {"audio": a, "stt": s},
        }

    # 2. Mic essentially silent — speech never crossed onset threshold.
    if a["peak_rms"] < _PEAK_LOW:
        return {
            "ok": False, "severity": "error",
            "message": f"Mic input is silent (peak RMS {a['peak_rms']:.3f} over {a['samples']/10:.0f}s).",
            "action": "Check the mic isn't muted; raise Input Volume to 100% in Windows Sound Settings.",
            "metrics": {"audio": a, "stt": s},
        }

    # 3. Noise floor near onset threshold — false triggers will dominate.
    if a["noise_floor"] > _NOISE_FLOOR_HIGH:
        return {
            "ok": False, "severity": "warn",
            "message": f"Background noise is high (floor {a['noise_floor']:.3f}). Whisper may transcribe ambient sounds.",
            "action": "Quiet the room, or raise ENERGY_THRESHOLD in config.py.",
            "metrics": {"audio": a, "stt": s},
        }

    # 4. Whisper is unsure — audio is unclear even though levels look fine.
    if s["samples"] >= 3 and s["avg_logprob"] < _LOGPROB_BAD:
        return {
            "ok": False, "severity": "warn",
            "message": f"Having trouble hearing — Whisper confidence is low (avg_logprob {s['avg_logprob']:.2f}).",
            "action": "Speak closer to the mic, or check for echo / room reverb.",
            "metrics": {"audio": a, "stt": s},
        }

    # 5. Whisper thinks recent audio isn't speech at all (hallucination risk).
    if s["samples"] >= 3 and s["max_no_speech"] > _NO_SPEECH_BAD:
        return {
            "ok": False, "severity": "warn",
            "message": f"Recent audio doesn't sound like speech (no_speech_prob {s['max_no_speech']:.2f}).",
            "action": "If Merlin's transcript is hallucinating, raise ENERGY_THRESHOLD in config.py.",
            "metrics": {"audio": a, "stt": s},
        }

    return {
        "ok": True, "severity": "ok",
        "message": f"Hearing well (peak {a['peak_rms']:.2f}, floor {a['noise_floor']:.3f}).",
        "action": None,
        "metrics": {"audio": a, "stt": s},
    }
