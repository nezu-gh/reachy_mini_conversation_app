"""SmallWebRTC transport — browser-based voice conversation via WebRTC.

Mounts /webrtc/* endpoints on the existing FastAPI app so users can
talk to the robot from any browser (phone/laptop) without Gradio.

Env vars:
    WEBRTC_MAX_CLIENTS  — max concurrent WebRTC sessions (default: 2)
    WEBRTC_ICE_SERVERS   — comma-separated STUN/TURN URLs (optional)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

MAX_CLIENTS = int(os.environ.get("WEBRTC_MAX_CLIENTS", "2"))
_ICE_RAW = os.environ.get("WEBRTC_ICE_SERVERS", "stun:stun.l.google.com:19302")
ICE_SERVERS = [s.strip() for s in _ICE_RAW.split(",") if s.strip()] if _ICE_RAW else []

# Track active sessions
_active_sessions: dict[str, Any] = {}
_session_lock = asyncio.Lock()


def mount_webrtc_routes(app: FastAPI, handler: Any) -> None:
    """Register WebRTC signalling endpoints on the FastAPI app.

    Args:
        app: The existing FastAPI application.
        handler: The pipecat provider instance (for building pipelines).
    """
    try:
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.request_handler import (
            ConnectionMode,
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    except ImportError:
        logger.warning("SmallWebRTC transport not available — skipping /webrtc routes")
        return

    webrtc_handler = SmallWebRTCRequestHandler(
        ice_servers=ICE_SERVERS or None,
        connection_mode=ConnectionMode.MULTIPLE,
    )

    @app.get("/webrtc")
    async def webrtc_ui() -> HTMLResponse:
        """Serve the WebRTC voice chat UI."""
        html_path = Path(__file__).parent.parent / "static" / "webrtc.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>WebRTC UI not found</h1>", status_code=404)

    @app.post("/webrtc/offer")
    async def webrtc_offer(request_data: dict) -> JSONResponse:
        """Handle WebRTC SDP offer from browser client."""
        async with _session_lock:
            if len(_active_sessions) >= MAX_CLIENTS:
                return JSONResponse(
                    {"error": f"Max {MAX_CLIENTS} concurrent sessions"},
                    status_code=429,
                )

        request = SmallWebRTCRequest.from_dict(request_data)

        async def on_connection(connection: SmallWebRTCConnection):
            pc_id = connection.pc_id
            logger.info("WebRTC client connected: %s", pc_id)
            async with _session_lock:
                _active_sessions[pc_id] = connection

            # Build and run a pipeline for this client
            asyncio.create_task(_run_webrtc_pipeline(connection, handler, pc_id))

        answer = await webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=on_connection,
        )
        if answer:
            return JSONResponse(answer)
        return JSONResponse({"error": "no answer"}, status_code=500)

    @app.patch("/webrtc/offer")
    async def webrtc_ice(request_data: dict) -> JSONResponse:
        """Handle ICE candidate trickle from browser."""
        patch = SmallWebRTCPatchRequest(
            pc_id=request_data["pc_id"],
            candidates=request_data.get("candidates", []),
        )
        await webrtc_handler.handle_patch_request(patch)
        return JSONResponse({"status": "ok"})

    @app.get("/webrtc/sessions")
    async def webrtc_sessions() -> JSONResponse:
        """Return count of active WebRTC sessions."""
        async with _session_lock:
            return JSONResponse({
                "active": len(_active_sessions),
                "max": MAX_CLIENTS,
            })

    logger.info(
        "WebRTC routes mounted: /webrtc (UI), /webrtc/offer (signalling), max_clients=%d",
        MAX_CLIENTS,
    )


async def _run_webrtc_pipeline(
    connection: Any,
    handler: Any,
    pc_id: str,
) -> None:
    """Build and run a pipecat pipeline for a single WebRTC client session."""
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.openai.tts import OpenAITTSService
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        from pipecat.vad.silero import SileroVADAnalyzer
        from pipecat.vad.vad import VADProcessor

        from reachy_mini_conversation_app.providers.pipecat_provider import (
            ASR_BASE_URL,
            ASR_MODEL,
            LLM_BASE_URL,
            LLM_MODEL,
            PIPELINE_SAMPLE_RATE,
            TTS_BASE_URL,
            TTS_MODEL,
        )
        from reachy_mini_conversation_app.prompts import get_session_instructions

        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                audio_out_enabled=True,
                audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
                video_in_enabled=False,
                video_out_enabled=False,
            ),
        )

        stt = OpenAISTTService(
            api_key="not-needed",
            base_url=ASR_BASE_URL,
            settings=OpenAISTTService.Settings(model=ASR_MODEL),
        )

        llm = OpenAILLMService(
            api_key="not-needed",
            base_url=LLM_BASE_URL,
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL,
                system_instruction=get_session_instructions(),
                extra={
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            ),
        )

        tts = OpenAITTSService(
            api_key="not-needed",
            base_url=TTS_BASE_URL,
            settings=OpenAITTSService.Settings(model=TTS_MODEL),
        )

        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(sample_rate=PIPELINE_SAMPLE_RATE),
        )

        pipeline = Pipeline([
            transport.input(),
            vad,
            stt,
            llm,
            tts,
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
                audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
                enable_metrics=True,
            ),
        )

        runner = PipelineRunner(handle_sigint=False)
        logger.info("WebRTC pipeline started for %s", pc_id)
        await runner.run(task)

    except Exception:
        logger.exception("WebRTC pipeline error for %s", pc_id)
    finally:
        async with _session_lock:
            _active_sessions.pop(pc_id, None)
        logger.info("WebRTC session ended: %s", pc_id)
