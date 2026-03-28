"""MCP server lifecycle coordinator.

Manages startup and shutdown of the three built-in MCP servers
(memory, vision, robot) as background asyncio tasks.

Usage (from an async context)::

    manager = MCPManager()
    await manager.start()
    # ... application runs ...
    await manager.stop()

TODOs:
- Integrate startup into main.py's run() alongside MovementManager.start().
  Example::

      mcp_manager = MCPManager()
      loop = asyncio.get_event_loop()
      loop.run_until_complete(mcp_manager.start())

- Switch from stdio to SSE transport for remote / multi-client access::

      mcp.run(transport="sse", host="0.0.0.0", port=PORT)

- Home Assistant integration: robot_server could forward state-change
  events to HA via python-homeassistant-api or a webhook.  memory_server
  could sync with HA input_text helpers for cross-device persistence.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MCPManager:
    """Lifecycle coordinator for the built-in MCP servers.

    Each server is started as an independent background asyncio task
    using stdio transport (suitable for local use and testing).
    """

    def __init__(self) -> None:
        """Initialise the manager with references to each server."""
        from reachy_mini_conversation_app.mcp_servers.memory_server import mcp as memory_mcp
        from reachy_mini_conversation_app.mcp_servers.robot_server import mcp as robot_mcp
        from reachy_mini_conversation_app.mcp_servers.vision_server import mcp as vision_mcp

        self._servers = {
            "memory": memory_mcp,
            "vision": vision_mcp,
            "robot": robot_mcp,
        }
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start all MCP servers as background asyncio tasks.

        Each server runs in its own task so a crash in one server does
        not bring down the others.  Errors are logged rather than raised.
        """
        for name, server in self._servers.items():
            logger.info("MCPManager: starting %s server", name)
            task = asyncio.create_task(self._run_server(name, server), name=f"mcp-{name}")
            self._tasks.append(task)

    async def stop(self) -> None:
        """Cancel all MCP server tasks and wait for them to finish."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("MCPManager: all servers stopped")

    @staticmethod
    async def _run_server(name: str, server: object) -> None:
        """Run a single MCP server, logging any unexpected errors.

        Args:
            name: Human-readable server name for log messages.
            server: FastMCP instance.

        TODO: switch to ``server.run_async(transport="sse")`` for remote
        access, or ``server.run_async(transport="stdio")`` for local.
        """
        try:
            logger.info("MCPManager: %s server task started", name)
            # FastMCP v2: run_async() with stdio transport for local use.
            # TODO: switch to SSE for remote clients.
            await server.run_async(transport="stdio")  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            logger.info("MCPManager: %s server task cancelled", name)
        except Exception as exc:
            logger.error("MCPManager: %s server crashed: %s", name, exc, exc_info=True)
