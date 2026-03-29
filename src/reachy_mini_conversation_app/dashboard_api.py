"""Dashboard API endpoints for the Reachy Mini web UI.

Provides REST endpoints for robot health, live log streaming (SSE),
runtime config, and movement controls. Mounted on the existing
settings_app FastAPI instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)

_start_time = time.time()


class DashboardLogHandler(logging.Handler):
    """Logging handler that pushes formatted records to an asyncio queue for SSE."""

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxlen)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "name": record.name,
            "msg": self.format(record),
        }
        # Fan-out to all active SSE subscribers
        for q in list(self._subscribers):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass  # subscriber is slow, drop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


_log_handler: DashboardLogHandler | None = None


def mount_dashboard_routes(
    app: FastAPI,
    handler: Any,
) -> None:
    """Register dashboard API endpoints on the FastAPI settings app."""
    from starlette.responses import StreamingResponse

    global _log_handler

    # Install log handler on root logger
    if _log_handler is None:
        _log_handler = DashboardLogHandler()
        _log_handler.setLevel(logging.INFO)
        _log_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logging.getLogger().addHandler(_log_handler)

    def _get_deps() -> Any:
        return getattr(handler, "deps", None)

    # --- GET /api/health ---
    @app.get("/api/health")
    def health() -> JSONResponse:
        deps = _get_deps()
        movement = {}
        if deps and hasattr(deps, "movement_manager") and deps.movement_manager:
            try:
                movement = deps.movement_manager.get_status()
            except Exception:
                pass

        camera_active = False
        if deps and hasattr(deps, "camera_worker") and deps.camera_worker:
            try:
                camera_active = deps.camera_worker.get_latest_frame() is not None
            except Exception:
                pass

        pipeline_health = {}
        if hasattr(handler, "get_pipeline_health") and callable(handler.get_pipeline_health):
            try:
                pipeline_health = handler.get_pipeline_health()
            except Exception:
                pass

        return JSONResponse({
            "uptime_s": round(time.time() - _start_time, 1),
            "movement": movement,
            "camera": {"active": camera_active},
            "provider": os.environ.get("PROVIDER", "unknown"),
            "model": os.environ.get("MODEL_NAME", "unknown"),
            "multimodal": os.environ.get("LLM_MULTIMODAL", ""),
            "pipeline": pipeline_health,
        })

    # --- GET /api/metrics ---
    @app.get("/api/metrics")
    def metrics() -> JSONResponse:
        if hasattr(handler, "get_detailed_metrics") and callable(handler.get_detailed_metrics):
            try:
                return JSONResponse(handler.get_detailed_metrics())
            except Exception:
                pass
        return JSONResponse({"error": "metrics not available"}, status_code=503)

    # --- GET /api/config ---
    @app.get("/api/config")
    def get_config() -> JSONResponse:
        return JSONResponse({
            "provider": os.environ.get("PROVIDER", "unknown"),
            "model_name": os.environ.get("MODEL_NAME", ""),
            "llm_base_url": os.environ.get("LLM_BASE_URL", ""),
            "tts_base_url": os.environ.get("TTS_BASE_URL", ""),
            "tts_model": os.environ.get("TTS_MODEL", ""),
            "asr_base_url": os.environ.get("ASR_BASE_URL", ""),
            "asr_model": os.environ.get("ASR_MODEL", ""),
            "llm_multimodal": os.environ.get("LLM_MULTIMODAL", ""),
            "head_tracker": os.environ.get("HEAD_TRACKER", ""),
            "custom_profile": os.environ.get("REACHY_MINI_CUSTOM_PROFILE", ""),
            "ha_url": os.environ.get("HA_URL", ""),
        })

    # --- GET /api/logs (SSE) ---
    @app.get("/api/logs")
    async def logs_sse() -> StreamingResponse:
        if _log_handler is None:
            return StreamingResponse(
                iter([]),
                media_type="text/event-stream",
            )
        q = _log_handler.subscribe()

        async def event_stream():
            try:
                while True:
                    entry = await q.get()
                    yield f"data: {json.dumps(entry)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                _log_handler.unsubscribe(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- POST /api/move ---
    @app.post("/api/move")
    async def move(data: dict) -> JSONResponse:
        deps = _get_deps()
        if not deps or not deps.movement_manager:
            return JSONResponse({"error": "movement not available"}, status_code=503)

        action = data.get("action", "")
        name = data.get("name", "")
        repeat = max(1, min(int(data.get("repeat", 1)), 10))

        if action == "emotion":
            try:
                from reachy_mini.motion.recorded_move import RecordedMoves
                from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

                recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
                available = recorded_moves.list_moves()
                if name not in available:
                    return JSONResponse({"error": f"unknown emotion: {name}", "available": available}, status_code=400)
                move = EmotionQueueMove(name, recorded_moves)
                deps.movement_manager.queue_move(move)
                return JSONResponse({"status": f"playing emotion: {name}"})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        elif action == "dance":
            try:
                from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
                from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

                if name not in AVAILABLE_MOVES:
                    return JSONResponse({"error": f"unknown dance: {name}", "available": list(AVAILABLE_MOVES.keys())}, status_code=400)
                move = DanceQueueMove(name)
                for _ in range(repeat):
                    deps.movement_manager.queue_move(move)
                return JSONResponse({"status": f"playing dance: {name} x{repeat}"})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        elif action == "head":
            try:
                from reachy_mini.utils import create_head_pose
                from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove

                deltas = {
                    "left": (0, 0, 0, 0, 0, 40),
                    "right": (0, 0, 0, 0, 0, -40),
                    "up": (0, 0, 0, 0, -30, 0),
                    "down": (0, 0, 0, 0, 30, 0),
                    "front": (0, 0, 0, 0, 0, 0),
                }
                if name not in deltas:
                    return JSONResponse({"error": f"unknown direction: {name}"}, status_code=400)
                target = create_head_pose(*deltas[name], degrees=True)
                current_head = deps.reachy_mini.get_current_head_pose()
                _, current_ant = deps.reachy_mini.get_current_joint_positions()
                goto = GotoQueueMove(
                    target_head_pose=target,
                    start_head_pose=current_head,
                    target_antennas=(0, 0),
                    start_antennas=(current_ant[0], current_ant[1]),
                    target_body_yaw=0,
                    start_body_yaw=0,
                    duration=0.5,
                )
                deps.movement_manager.queue_move(goto)
                return JSONResponse({"status": f"looking {name}"})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse({"error": f"unknown action: {action}"}, status_code=400)

    # --- POST /api/listen ---
    @app.post("/api/listen")
    async def listen(data: dict) -> JSONResponse:
        deps = _get_deps()
        if not deps or not deps.movement_manager:
            return JSONResponse({"error": "movement not available"}, status_code=503)

        enable = bool(data.get("enable", False))
        deps.movement_manager.set_listening(enable)
        return JSONResponse({"status": "listening" if enable else "idle"})
