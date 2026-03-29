"""Tests for the camera tool."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_returns_error_without_vision_on_text_only_llm() -> None:
    """Text-only LLM without a vision processor should return an error."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    with patch.dict("os.environ", {"MODEL_NAME": "phi-3-mini", "LLM_MULTIMODAL": ""}):
        result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    assert "vision" in result["error"].lower()


@pytest.mark.asyncio
async def test_camera_tool_returns_b64_for_multimodal_llm() -> None:
    """Multimodal LLM should get base64 image when no vision processor."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    with patch.dict("os.environ", {"MODEL_NAME": "Qwen3.5-35B", "LLM_MULTIMODAL": ""}):
        result = await Camera()(deps, question="What color is this?")

    assert "b64_im" in result
    assert "question" in result


@pytest.mark.asyncio
async def test_camera_tool_multimodal_env_override() -> None:
    """LLM_MULTIMODAL=1 should force multimodal mode regardless of model name."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    with patch.dict("os.environ", {"MODEL_NAME": "some-text-only-model", "LLM_MULTIMODAL": "1"}):
        result = await Camera()(deps, question="What do you see?")

    assert "b64_im" in result


@pytest.mark.asyncio
async def test_camera_tool_multimodal_env_disable() -> None:
    """LLM_MULTIMODAL=0 should disable multimodal even for Qwen3.5."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    with patch.dict("os.environ", {"MODEL_NAME": "Qwen3.5-35B", "LLM_MULTIMODAL": "0"}):
        result = await Camera()(deps, question="What do you see?")

    assert "error" in result
    assert "vision" in result["error"].lower()


@pytest.mark.asyncio
async def test_camera_tool_uses_local_vision_processor_when_available() -> None:
    """The camera tool should use on-demand local vision when configured."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = np.zeros((32, 32, 3), dtype=np.uint8)

    vision_processor = MagicMock()
    vision_processor.process_image.return_value = "A red cup on a table."

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_processor=vision_processor,
    )

    result = await Camera()(deps, question="What do you see?")

    assert result == {"image_description": "A red cup on a table."}
    vision_processor.process_image.assert_called_once_with(
        camera_worker.get_latest_frame.return_value,
        "What do you see?",
    )


@pytest.mark.asyncio
async def test_camera_tool_returns_error_without_frame() -> None:
    """No frame from any source should return a clear error."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = None

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
    )

    with patch("reachy_mini_conversation_app.camera_capture.capture_frame", return_value=None):
        result = await Camera()(deps, question="What do you see?")

    assert "error" in result
