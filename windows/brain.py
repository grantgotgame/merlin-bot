"""LLM conversation engine via LM Studio.
Handles wake words, conversation window, mute/unmute, and chat history.
"""

import re
import requests
import time
import config


class Brain:
    def __init__(self, bus=None):
        self.bus = bus
        self.history = []
        self.last_response_time = 0
        self.muted = False
        self.greeted_today = False
        self._check_connection()

    def _emit(self, event, **kwargs):
        if self.bus is not None:
            try:
                self.bus.emit(event, **kwargs)
            except Exception:
                pass

    def _check_connection(self):
        """Verify LM Studio is reachable."""
        try:
            url = config.LLM_URL.replace("/chat/completions", "/models")
            r = requests.get(url, timeout=3)
            if r.ok:
                models = r.json().get("data", [])
                if models:
                    print(f"[brain] LM Studio connected. Model: {models[0].get('id', 'unknown')}")
                else:
                    print("[brain] ⚠️  LM STUDIO HAS NO MODEL LOADED!")
                    print("[brain] ⚠️  Open LM Studio → search for a model → download → load it")
                    print("[brain] ⚠️  Merlin CANNOT respond without a model. This is the #1 setup issue.")
            else:
                print(f"[brain] LM Studio responded with {r.status_code}.")
        except requests.ConnectionError:
            print("[brain] WARNING: Cannot reach LM Studio at localhost:1234.")
            print("[brain] Start LM Studio and load a model, then restart Merlin.")
        except Exception as e:
            print(f"[brain] Connection check error: {e}")

    def process(self, text):
        """
        Process transcribed speech. Returns a response string, or None if
        the utterance should be ignored (not addressed to Merlin, muted, etc.).
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # --- Mute controls ---
        if any(w in text_lower for w in config.MUTE_WORDS):
            self.muted = True
            print("[brain] Muted.")
            self._emit("mute_toggled", muted=True)
            return None

        if any(w in text_lower for w in config.UNMUTE_WORDS):
            self.muted = False
            print("[brain] Unmuted.")
            self._emit("mute_toggled", muted=False)
            return "I'm listening."

        # If muted, only wake word breaks through
        if self.muted:
            if any(w in text_lower for w in config.WAKE_WORDS):
                self.muted = False
                print("[brain] Unmuted via wake word.")
                self._emit("mute_toggled", muted=False)
            else:
                return None

        # --- Nevermind ---
        if any(w in text_lower for w in config.NEVERMIND_WORDS):
            self.last_response_time = 0
            self._emit("conversation_dismissed")
            return None

        # --- Wake word or conversation window check ---
        in_window = (time.time() - self.last_response_time) < config.CONVERSATION_WINDOW
        has_wake = any(w in text_lower for w in config.WAKE_WORDS)

        if not has_wake and not in_window:
            self._emit("speech_ignored", text=text)
            return None

        return self._handle(text, text_lower, intent="conversation")

    def process_typed(self, text):
        """
        Path used by typed input from the web UI: skip wake-word gating but
        still respect mute (mute can be toggled in the UI). Always invokes
        the LLM.
        """
        if not text:
            return None
        if self.muted:
            self.muted = False
            self._emit("mute_toggled", muted=False)
        return self._handle(text, text.lower().strip(), intent="typed")

    def _handle(self, text, text_lower, intent):
        # Strip wake word prefix
        message = text
        for w in sorted(config.WAKE_WORDS, key=len, reverse=True):
            if text_lower.startswith(w):
                message = text[len(w):].strip(" ,!?.")
                break
        if not message:
            message = "hello"

        self._emit("thinking_start", message=message, intent=intent)
        t0 = time.time()
        response = self._call_llm(message)
        latency_ms = int((time.time() - t0) * 1000)

        if response:
            self.last_response_time = time.time()
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})
            if len(self.history) > config.MAX_HISTORY * 2:
                self.history = self.history[-(config.MAX_HISTORY * 2):]
            self._emit("thinking_complete", text=response, intent=intent, latency_ms=latency_ms)
        else:
            self._emit("thinking_failed", latency_ms=latency_ms)

        return response

    def _call_llm(self, message):
        """Send message to LM Studio and return the response text."""
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        try:
            r = requests.post(
                config.LLM_URL,
                json={
                    "messages": messages,
                    "max_tokens": config.MAX_TOKENS,
                    "temperature": config.TEMPERATURE,
                    "stream": False,
                },
                timeout=30,
            )
            if r.ok:
                raw = r.json()["choices"][0]["message"]["content"].strip()
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return cleaned if cleaned else raw
            else:
                print(f"[brain] LLM returned {r.status_code}: {r.text[:200]}")
                return None
        except requests.Timeout:
            print("[brain] LLM timed out (30s). Model may be loading.")
            return None
        except requests.ConnectionError:
            print("[brain] Can't reach LM Studio. Is it running?")
            return None
        except Exception as e:
            print(f"[brain] LLM error: {e}")
            return None

    def on_face_arrived(self):
        """Called when face tracker detects someone at the desk."""
        if not self.greeted_today and not self.muted:
            self.greeted_today = True
            hour = time.localtime().tm_hour
            return "Morning." if hour < 12 else "Hey."
        return None
