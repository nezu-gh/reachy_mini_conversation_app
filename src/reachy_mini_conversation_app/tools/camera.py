import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        frame = None
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()

        # Fall back to subprocess-based capture when camera_worker has no frames
        # (works around GStreamer Python threading issue with unixfdsrc)
        if frame is None:
            try:
                from reachy_mini_conversation_app.camera_capture import capture_frame
                frame = await asyncio.to_thread(capture_frame)
            except Exception as e:
                logger.warning("Subprocess camera capture failed: %s", e)

        if frame is None:
            logger.error("No frame available from any camera source")
            return {"error": "No frame available"}

        if deps.vision_processor is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_processor.process_image, frame, question,
            )
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # No vision processor available — return an error instead of
        # dumping a raw base64 image into the text-only LLM context.
        logger.warning("camera: no vision processor, cannot analyse image")
        return {
            "error": "Vision is not available. I took a picture but I cannot "
                     "analyse it because no vision model is configured. "
                     "Tell the user you cannot see right now."
        }
