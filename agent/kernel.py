"""Agent Kernel — ReAct loop that sends messages to Ollama with tools."""

import json
import requests
from .tools.base import BaseTool
from .config import OLLAMA_URL, REQUEST_TIMEOUT, MAX_TOOL_ROUNDS

# ANSI colors for terminal output
DIM = "\033[2m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RED = "\033[31m"
RESET = "\033[0m"


class AgentKernel:
    def __init__(self, model: str, tools: list[BaseTool], system_prompt: str,
                 ollama_url: str = OLLAMA_URL, max_rounds: int = MAX_TOOL_ROUNDS):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.tools_schema = [t.to_ollama_schema() for t in tools]
        self.system_prompt = system_prompt
        self.ollama_url = ollama_url
        self.max_rounds = max_rounds
        self.history: list[dict] = []

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def run(self, user_message: str) -> str:
        """Process a user message through the full ReAct loop. Returns final text."""
        self.history.append({"role": "user", "content": user_message})

        for round_num in range(self.max_rounds):
            # Build messages: system + history
            messages = [{"role": "system", "content": self.system_prompt}] + self.history

            # Call Ollama
            try:
                resp = requests.post(self.ollama_url, json={
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools_schema,
                    "stream": False,
                    "think": False,
                }, timeout=REQUEST_TIMEOUT)
            except requests.ConnectionError:
                msg = f"Cannot reach Ollama at {self.ollama_url}. Is it running?"
                print(f"{RED}[error]{RESET} {msg}")
                return msg
            except requests.Timeout:
                msg = f"Ollama request timed out after {REQUEST_TIMEOUT}s."
                print(f"{RED}[error]{RESET} {msg}")
                return msg

            if resp.status_code != 200:
                msg = f"Ollama returned status {resp.status_code}: {resp.text[:200]}"
                print(f"{RED}[error]{RESET} {msg}")
                return msg

            data = resp.json()
            message = data.get("message", {})

            # Show thinking if present
            thinking = message.get("thinking", "")
            if thinking:
                print(f"{DIM}[thought]{RESET} {DIM}{thinking}{RESET}")

            content = message.get("content", "")
            tool_calls = message.get("tool_calls")

            if tool_calls:
                # Append the full assistant response (with tool_calls) to history
                self.history.append(message)

                # Show any text content alongside tool calls
                if content:
                    print(f"{DIM}[thought]{RESET} {DIM}{content}{RESET}")

                # Execute each tool call
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    arguments = func.get("arguments", {})

                    print(f"{YELLOW}[action]{RESET} {tool_name}({', '.join(f'{k}={json.dumps(v)}' for k, v in arguments.items())})")

                    # Look up and execute tool
                    tool = self.tools.get(tool_name)
                    if tool is None:
                        result = f"Error: unknown tool '{tool_name}'"
                        print(f"{RED}[error]{RESET} {result}")
                    else:
                        try:
                            result = tool.execute(**arguments)
                        except Exception as e:
                            result = f"Error executing {tool_name}: {e}"
                            print(f"{RED}[error]{RESET} {result}")

                    # Show observation (truncated for terminal)
                    display = result[:500] + "..." if len(result) > 500 else result
                    print(f"{CYAN}[observation]{RESET} {display}")

                    # Append tool result to history
                    self.history.append({"role": "tool", "content": result})

                # Loop back for next round
                continue

            else:
                # Final text answer — no tool calls
                self.history.append({"role": "assistant", "content": content})
                print(f"{BOLD}[answer]{RESET} {content}")
                return content

        # Safety limit reached
        msg = f"Tool loop limit reached ({self.max_rounds} rounds)."
        print(f"{RED}[error]{RESET} {msg}")
        return msg
