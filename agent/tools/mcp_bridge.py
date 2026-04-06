"""MCP Bridge — discovers tools from MCP servers and wraps them as BaseTools."""

import json
from pathlib import Path
from .base import BaseTool
from ..mcp_client import MCPClient

# Default config path
MCP_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"


class MCPTool(BaseTool):
    """Wraps a single MCP tool as a BaseTool for the agent kernel."""

    def __init__(self, client: MCPClient, tool_def: dict):
        self.client = client
        self.tool_def = tool_def
        # Prefix tool name with server name to avoid collisions
        self.name = f"{client.name}__{tool_def['name']}"
        self.description = tool_def.get("description", "")
        self._params = tool_def.get("inputSchema", {
            "type": "object",
            "properties": {},
            "required": [],
        })

    def parameters(self) -> dict:
        return self._params

    def execute(self, **kwargs) -> str:
        return self.client.call_tool(self.tool_def["name"], kwargs)


def load_mcp_tools(config_path: str = None, verbose: bool = True) -> tuple[list[BaseTool], list[MCPClient]]:
    """Load MCP servers from config and return all discovered tools.

    Returns (tools, clients) — caller should stop() clients on shutdown.
    """
    config_file = Path(config_path) if config_path else MCP_CONFIG_PATH
    if not config_file.exists():
        if verbose:
            print(f"[mcp] No config at {config_file} — skipping MCP tools")
        return [], []

    with open(config_file) as f:
        config = json.load(f)

    servers = config.get("servers", {})
    all_tools = []
    clients = []

    for name, server_config in servers.items():
        if not server_config.get("enabled", True):
            if verbose:
                print(f"[mcp] {name}: disabled, skipping")
            continue

        command = server_config.get("command", "")
        args = server_config.get("args", [])
        env = server_config.get("env")

        if verbose:
            print(f"[mcp] {name}: starting...")

        client = MCPClient(name=name, command=command, args=args, env=env)
        try:
            client.start()
            tools = client.list_tools()
            if verbose:
                print(f"[mcp] {name}: {len(tools)} tools discovered")
                for t in tools:
                    print(f"  - {t['name']}: {t.get('description', '')[:60]}")

            for tool_def in tools:
                all_tools.append(MCPTool(client, tool_def))
            clients.append(client)

        except Exception as e:
            print(f"[mcp] {name}: failed to start — {e}")
            client.stop()

    return all_tools, clients
