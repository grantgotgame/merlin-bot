"""PTZ preset names shared between HTTP handler, MCP server, and brain."""

PTZ_ACTIONS = frozenset(
    {"look_left", "look_right", "look_up", "look_down", "look_center", "look_around"}
)
