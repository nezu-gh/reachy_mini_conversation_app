"""Tool: play a non-verbal micro-expression sound.

Plays a short sound effect through the robot speaker and synchronises
head movement via HeadWobbler — much faster than generating TTS for
simple reactions like acknowledgments or surprise.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict

import numpy as np

from reachy_mini_conversation_app.audio.sound_library import SoundLibrary
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

_library: SoundLibrary | None = None


def _get_library() -> SoundLibrary:
    global _library
    if _library is None:
        _library = SoundLibrary()
    return _library


class MicroExpression(Tool):
    """Play a short non-verbal sound as a quick emotional reaction."""

    name = "micro_expression"
    description = (
        "Play a brief non-verbal sound (beep, chirp, hum) as a quick "
        "emotional micro-reaction. Use INSTEAD of speech for simple "
        "acknowledgments, or BEFORE speech to express immediate emotion. "
        f"Available expressions: {', '.join(_get_library().list_expressions())}."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "The micro-expression to play. Options: "
                    + ", ".join(_get_library().list_expressions())
                ),
            },
        },
        "required": ["expression"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        expression = kwargs.get("expression", "")
        if not expression:
            return {"error": "expression is required"}

        lib = _get_library()
        pcm = lib.get(expression)
        if pcm is None:
            return {
                "error": f"Unknown expression '{expression}'",
                "available": lib.list_expressions(),
            }

        if deps.output_queue is None:
            return {"error": "Audio output not available (no output_queue)"}

        logger.info("micro_expression: playing '%s' (%d samples)", expression, len(pcm))

        sr = deps.audio_sample_rate

        # Feed HeadWobbler so the robot moves in sync with the sound
        if deps.head_wobbler is not None:
            b64 = base64.b64encode(pcm.tobytes()).decode("utf-8")
            deps.head_wobbler.feed(b64)

        # Push audio to the output queue in chunks (~20ms each) so it
        # interleaves smoothly with the pipeline and can be interrupted.
        chunk_samples = sr // 50  # 20ms chunks
        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            await deps.output_queue.put((sr, chunk.reshape(1, -1)))

        return {"status": "played", "expression": expression}
