"""Pipecat-based local conversation provider stub.

This module wires together a fully-local audio pipeline using pipecat-ai.
All heavy imports are guarded by a try/except so the module can be imported
even when pipecat-ai is not installed; a RuntimeError is raised at
instantiation time instead.

All three backends run as Docker containers on a local VM and expose
OpenAI-compatible ``/v1`` endpoints.  Default URLs are configured via
environment variables (see ``_DEFAULT_*`` constants below).

Local pipeline design (TODOs — not yet wired):
  STT : Qwen-ASR via OpenAI-compat /v1 endpoint (ASR_BASE_URL)
  LLM : Qwen3.5-35B-A3B via ik-llama.cpp /v1 endpoint (LLM_BASE_URL)
  TTS : Qwen3-TTS via OpenAI-compat /v1 endpoint (TTS_BASE_URL)
  VAD : SileroVADAnalyzer (local, no endpoint)
  Tools: dispatch_tool_call_with_manager() via Pipecat FunctionCallProcessor
  Audio: pipecat native 16 kHz → fastrtc contract 24 kHz resampling
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.providers.base import ConversationProvider

if TYPE_CHECKING:
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

logger = logging.getLogger(__name__)

try:
    import pipecat  # noqa: F401

    _PIPECAT_AVAILABLE = True
except ImportError:
    _PIPECAT_AVAILABLE = False

# Default VM IP and endpoints (override via env vars)
_DEFAULT_VM = "192.168.178.155"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", f"http://{_DEFAULT_VM}:3443/v1")
LLM_MODEL = os.environ.get("MODEL_NAME", "qwen3:8b")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", f"http://{_DEFAULT_VM}:7034/v1")
ASR_BASE_URL = os.environ.get("ASR_BASE_URL", f"http://{_DEFAULT_VM}:8015/v1")

LOCAL_VISION_MODEL = os.environ.get("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")


class PipecatProvider(ConversationProvider):
    """Fully-local conversation backend powered by pipecat-ai.

    Raises RuntimeError at instantiation if pipecat-ai is not installed.
    Install the local_pipeline extra to use this provider::

        pip install '.[local_pipeline]'
    """

    def __init__(
        self,
        deps: "ToolDependencies",
        *,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialise the provider and verify pipecat-ai is available."""
        if not _PIPECAT_AVAILABLE:
            raise RuntimeError(
                "pipecat-ai is not installed. "
                "Run: pip install '.[local_pipeline]' to use the local pipeline."
            )
        super().__init__()
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

    # ------------------------------------------------------------------
    # ConversationProvider abstract methods
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Apply personality profile.

        TODO: call _load_profile_tools() first so tool specs are resolved
        before the pipeline starts.
        """
        logger.warning("PipecatProvider.apply_personality: not yet implemented (profile=%r)", profile)
        return f"PipecatProvider: personality '{profile}' noted but not applied (stub)."

    async def get_available_voices(self) -> List[str]:
        """Return voices available from the local TTS engine.

        TODO: query Qwen3-TTS at TTS_BASE_URL for available voice IDs.
        """
        logger.warning("PipecatProvider.get_available_voices: not yet implemented")
        return []

    # ------------------------------------------------------------------
    # AsyncStreamHandler audio contract — no-op stubs
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive an audio frame from fastrtc.

        TODO: feed into pipecat pipeline input transport.
        """
        logger.warning("PipecatProvider.receive: audio pipeline not yet wired")

    def copy(self) -> "PipecatProvider":
        """Return a new handler for an incoming WebRTC connection."""
        return PipecatProvider(self.deps, gradio_mode=self.gradio_mode, instance_path=self.instance_path)

    async def emit(self) -> Any:
        """Emit an audio frame to fastrtc.

        TODO: pull from pipecat pipeline output transport and resample
        16 kHz → 24 kHz to match the fastrtc contract.
        """
        logger.warning("PipecatProvider.emit: audio pipeline not yet wired")
        return None

    async def start_up(self) -> None:
        """Start the pipecat pipeline.

        TODO:
        - Instantiate SileroVADAnalyzer
        - Instantiate STT via OpenAI-compat client → ASR_BASE_URL (Qwen-ASR)
        - Instantiate LLM via OpenAI-compat client → LLM_BASE_URL (Qwen3.5-35B, model=LLM_MODEL)
        - Instantiate TTS via OpenAI-compat client → TTS_BASE_URL (Qwen3-TTS)
        - Wire FunctionCallProcessor with dispatch_tool_call_with_manager()
        - Build and start pipecat Pipeline
        """
        logger.warning("PipecatProvider.start_up: pipeline not yet implemented")

    async def shutdown(self) -> None:
        """Stop the pipecat pipeline gracefully.

        TODO: cancel pipeline task and clean up resources.
        """
        logger.warning("PipecatProvider.shutdown: pipeline not yet implemented")
