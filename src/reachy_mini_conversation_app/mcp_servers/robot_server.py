"""MCP server that exposes robot tools from the core_tools registry.

Run standalone (stdio transport)::

    python -m reachy_mini_conversation_app.mcp_servers.robot_server

TODOs:
- Decide robot connection ownership: dedicated connection vs shared IPC
  with the main process.
- Wire BackgroundToolManager lifecycle into mcp lifespan.
- Add @mcp.resource() endpoints for robot state (head pose, battery, etc.).
- _register_tools() is commented out below pending live ToolDependencies —
  uncomment and supply deps to activate.
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("robot")


# def _register_tools(deps: "ToolDependencies") -> None:  # noqa: ERA001
#     """Iterate the tool registry and register each as an MCP tool.
#
#     Args:
#         deps: Live ToolDependencies connected to a running robot.
#     """
#     from reachy_mini_conversation_app.tools.core_tools import get_tool_specs, ALL_TOOLS  # noqa: ERA001
#
#     get_tool_specs()  # ensures ALL_TOOLS is populated                    # noqa: ERA001
#     for name, tool in ALL_TOOLS.items():                                   # noqa: ERA001
#         # Capture loop variable explicitly                                 # noqa: ERA001
#         def _make_handler(t=tool):                                         # noqa: ERA001
#             async def handler(**kwargs):                                    # noqa: ERA001
#                 return await t.execute(deps, kwargs)                       # noqa: ERA001
#             handler.__name__ = t.name                                      # noqa: ERA001
#             handler.__doc__ = t.description                                # noqa: ERA001
#             return handler                                                  # noqa: ERA001
#         mcp.tool()(_make_handler())                                        # noqa: ERA001


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
