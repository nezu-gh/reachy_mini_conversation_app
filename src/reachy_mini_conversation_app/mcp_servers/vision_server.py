"""MCP server for camera and vision capabilities.

Run standalone (stdio transport)::

    python -m reachy_mini_conversation_app.mcp_servers.vision_server

TODOs:
- Wire VisionProcessor (SmolVLM2) from vision/processors.py into describe_scene().
- Wire CameraWorker to supply frames for both tools.
- Add @mcp.resource() endpoint for live YOLO face-bbox stream.
- Consider exposing a Home Assistant camera entity as an image source via
  python-homeassistant-api so the robot can describe rooms it cannot see
  directly.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("vision")


@mcp.tool()
async def describe_scene(prompt: Optional[str] = None) -> str:
    """Describe what the robot's camera currently sees.

    Args:
        prompt: Optional question or instruction to guide the description.

    Returns:
        A natural-language description of the scene.

    TODO: capture frame from CameraWorker, pass to VisionProcessor.describe().
    """
    logger.warning("vision_server.describe_scene: VisionProcessor not yet wired")
    return "Vision pipeline not yet wired."


@mcp.tool()
async def track_face() -> dict:
    """Return the bounding box of the most prominent face in view.

    Returns:
        Dict with keys x, y, width, height (normalised 0-1), or empty dict
        if no face is detected.

    TODO: wire YOLO head tracker from vision/yolo_head_tracker.py.
    """
    logger.warning("vision_server.track_face: YOLO tracker not yet wired")
    return {}


if __name__ == "__main__":
    mcp.run(transport="stdio")
