import base64
import asyncio
import logging
import os
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

        # No local vision processor — check if LLM is multimodal and can
        # handle images directly (e.g., Qwen3.5 with vision support).
        _multimodal_patterns = ("vlm", "vl-", "vision", "llava", "smolvlm", "qwen3.5")
        model_name = os.environ.get("MODEL_NAME", "").lower()
        is_multimodal = (
            os.environ.get("LLM_MULTIMODAL", "").lower() in ("1", "true", "yes")
            or any(p in model_name for p in _multimodal_patterns)
        )

        if is_multimodal:
            # Encode frame as base64 JPEG for the multimodal LLM
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                logger.info("camera: returning base64 image for multimodal LLM")
                return {"b64_im": b64, "question": question}

        logger.warning("camera: no vision processor and LLM is not multimodal")
        return {
            "error": "Vision is not available. I took a picture but I cannot "
                     "analyse it because no vision model is configured. "
                     "Tell the user you cannot see right now."
        }
