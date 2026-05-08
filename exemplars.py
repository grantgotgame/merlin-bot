"""Few-shot exemplar selector for Merlin's voice.

Reads training/lora-pairs-final.jsonl and picks a handful of (input, output)
pairs to inject into each LLM call as user/assistant turns *before* the
actual user message. The base model copies the voice from the exemplars
without us having to fine-tune.

Wizard principle: Merlin picks his own examples per turn instead of being
frozen at training time. Selection tilts toward his core voice (general
neutral pairs) plus one or two category-matched pairs when we have a
hint about the conversation's energy or intent. Seed rotates per call so
he doesn't repeat the same exemplars back-to-back.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

# How many pairs we ask the model to imitate per turn.
DEFAULT_N = 6
# Always anchor to general/neutral — the largest, most voice-defining bucket.
GENERAL_RATIO = 0.6
# stt_error is always in the mix — teaches "Didn't catch that" failure mode.
ALWAYS_INCLUDE_STT_ERROR = True


def _default_pairs_path() -> str:
    """`training/lora-pairs-final.jsonl` relative to repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    # windows/exemplars.py → repo root is one level up
    candidate = os.path.normpath(os.path.join(here, "..", "training", "lora-pairs-final.jsonl"))
    if os.path.exists(candidate):
        return candidate
    # If module is mirrored to repo root, training/ is a sibling.
    candidate = os.path.normpath(os.path.join(here, "training", "lora-pairs-final.jsonl"))
    return candidate


class Exemplars:
    """Loads the curated pair set once; picks a fresh subset per call."""

    def __init__(self, path: str | None = None):
        self.path = path or _default_pairs_path()
        self.pairs: list[dict[str, Any]] = []
        self._by_category: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                self.pairs = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"[exemplars] Training pairs not found at {self.path} — voice will use stock model only.")
            self.pairs = []
            return
        except Exception as e:
            print(f"[exemplars] Failed to load {self.path}: {e}")
            self.pairs = []
            return

        # Index by category for fast filtered sampling.
        self._by_category = {}
        for p in self.pairs:
            cat = p.get("category", "general")
            self._by_category.setdefault(cat, []).append(p)

        counts = ", ".join(f"{c}={len(v)}" for c, v in sorted(self._by_category.items()))
        print(f"[exemplars] Loaded {len(self.pairs)} voice pairs ({counts})")

    def loaded(self) -> bool:
        return bool(self.pairs)

    def _filter_clean(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop pairs whose output looks truncated or contains stale specifics."""
        clean: list[dict[str, Any]] = []
        for p in pairs:
            out = (p.get("output") or "").strip()
            if not out or len(out) > 220:
                continue
            # Skip outputs that end mid-sentence (training-data artefact).
            if out.endswith((",", ":", "—", "-")) or out.endswith("inject "):
                continue
            clean.append(p)
        return clean

    def pick(
        self,
        n: int = DEFAULT_N,
        category: str | None = None,
        energy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pick `n` exemplars, biased toward general voice + optional context.

        Selection (when n=6):
          - 60% (4) from general/neutral — Merlin's core voice
          - 1 from `stt_error` — teaches the "Didn't catch that" graceful fail
          - 1 from the matching category (red_frustrated, green_productive,
            schedule_day) when we have a hint, else random other.

        Seed rotates per minute so back-to-back calls in the same minute
        share examples (cache-friendly) but a long conversation rotates.
        """
        if not self.pairs:
            return []

        rng = random.Random(int(time.time() // 60))

        general = self._filter_clean(self._by_category.get("general", []))
        n_general = max(1, int(round(n * GENERAL_RATIO)))
        chosen: list[dict[str, Any]] = []
        if general:
            chosen.extend(rng.sample(general, min(n_general, len(general))))

        if ALWAYS_INCLUDE_STT_ERROR:
            stt = self._filter_clean(self._by_category.get("stt_error", []))
            if stt and len(chosen) < n:
                chosen.append(rng.choice(stt))

        # Category-matched pair (red_frustrated, green_productive, etc.)
        # If no hint provided, sample from a non-general bucket for variety.
        side_buckets = [c for c in self._by_category if c not in ("general", "stt_error")]
        target_cat = category if category in side_buckets else (rng.choice(side_buckets) if side_buckets else None)
        if target_cat:
            side = self._filter_clean(self._by_category[target_cat])
            while side and len(chosen) < n:
                pick = rng.choice(side)
                if pick not in chosen:
                    chosen.append(pick)
                if len(side) < 3:
                    break  # avoid infinite loop on tiny buckets

        # If we under-shot (small dataset), top up from general.
        while len(chosen) < n and general:
            extra = rng.choice(general)
            if extra not in chosen:
                chosen.append(extra)
            else:
                break

        return chosen[:n]

    def as_few_shot_messages(self, n: int = DEFAULT_N, category: str | None = None) -> list[dict[str, str]]:
        """Format picks as OpenAI-style chat messages: alternating user/assistant
        turns ready to drop in front of the real user message."""
        msgs: list[dict[str, str]] = []
        for p in self.pick(n=n, category=category):
            inp = (p.get("input") or "").strip()
            out = (p.get("output") or "").strip()
            if not inp or not out:
                continue
            msgs.append({"role": "user", "content": inp})
            msgs.append({"role": "assistant", "content": out})
        return msgs
