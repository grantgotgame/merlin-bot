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
