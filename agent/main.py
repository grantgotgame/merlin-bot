#!/usr/bin/env python3
"""RBOS Agent — CLI entry point. REPL or single-shot.

Run from RBOS root:
    python3 merlin/agent/main.py                         # REPL mode
    python3 merlin/agent/main.py "What's The Thing?"     # single-shot
    python3 merlin/agent/main.py --model gemma4:e4b "..." # different model
    python3 merlin/agent/main.py --no-mcp "..."          # skip MCP servers
"""

import argparse
import atexit
import os
import sys

# Add RBOS root to path so imports work when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from merlin.agent.config import DEFAULT_MODEL, SYSTEM_PROMPT
from merlin.agent.kernel import AgentKernel
from merlin.agent.tools.filesystem import default_tools
from merlin.agent.tools.mcp_bridge import load_mcp_tools

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Track MCP clients for cleanup
_mcp_clients = []


def build_agent(model: str, use_mcp: bool = True) -> AgentKernel:
    tools = default_tools()

    if use_mcp:
        mcp_tools, clients = load_mcp_tools()
        tools.extend(mcp_tools)
        _mcp_clients.extend(clients)

    return AgentKernel(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def cleanup():
    """Shut down MCP servers on exit."""
    for client in _mcp_clients:
        client.stop()


def repl(agent: AgentKernel):
    """Interactive REPL mode."""
    tool_count = len(agent.tools)
    print(f"{BOLD}RBOS Agent{RESET} — model: {agent.model}, {tool_count} tools loaded")
    print(f"{DIM}Type 'quit' to exit, 'reset' to clear history, 'tools' to list tools.{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{BOLD}>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print(f"{DIM}History cleared.{RESET}")
            continue
        if user_input.lower() == "tools":
            for name, tool in sorted(agent.tools.items()):
                print(f"  {name}: {tool.description[:70]}")
            print()
            continue

        agent.run(user_input)
        print()  # blank line between turns


def single_shot(agent: AgentKernel, prompt: str):
    """Run one prompt and exit."""
    result = agent.run(prompt)
    if not result:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RBOS Agent — local AI with tools")
    parser.add_argument("prompt", nargs="?", help="Single-shot prompt (omit for REPL)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-mcp", action="store_true", help="Skip loading MCP servers")
    args = parser.parse_args()

    atexit.register(cleanup)
    agent = build_agent(args.model, use_mcp=not args.no_mcp)

    if args.prompt:
        single_shot(agent, args.prompt)
    else:
        repl(agent)


if __name__ == "__main__":
    main()
