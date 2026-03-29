"""MCP server that exposes robot tools from the core_tools registry.

Run standalone (stdio transport)::

    python -m reachy_mini_conversation_app.mcp_servers.robot_server

Pending:
- Decide robot connection ownership: dedicated connection vs shared IPC
  with the main process.
- Wire BackgroundToolManager lifecycle into mcp lifespan.
- Add @mcp.resource() endpoints for robot state (head pose, battery, etc.).
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("robot")


@mcp.tool()
async def list_available_tools() -> list[str]:
    """Return the names of all registered robot tools.

    TODO: replace with live registry once ToolDependencies are available.
    """
    return [
        "dance",
        "stop_dance",
        "play_emotion",
        "stop_emotion",
        "move_head",
        "camera",
        "do_nothing",
        "head_tracking",
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
