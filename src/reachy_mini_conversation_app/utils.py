from __future__ import annotations
import os
import sys
import logging
import argparse
import warnings
import subprocess
from typing import TYPE_CHECKING, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker

# Import config early so .env is loaded before parse_args reads os.environ
import reachy_mini_conversation_app.config  # noqa: F401


if TYPE_CHECKING:
    from reachy_mini_conversation_app.vision.processors import VisionProcessor


class CameraVisionInitializationError(Exception):
    """Raised when camera or vision setup fails in an expected way."""


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe"],
        default=None,
        help="Head-tracking backend: yolo uses a local face detector, mediapipe uses reachy_mini_toolbox. Disabled by default.",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of gpt-realtime vision",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=os.environ.get("REACHY_ROBOT_NAME"),
        help="[Optional] Robot host/IP to target. Falls back to REACHY_ROBOT_NAME env var.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "pipecat"],
        default=os.environ.get("PROVIDER", "openai"),
        help="Conversation backend. 'pipecat' requires the local_pipeline extra. Env: PROVIDER",
    )
    return parser.parse_known_args()


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
) -> tuple[CameraWorker | None, VisionProcessor | None]:
    """Initialize camera capture, optional head tracking, and optional local vision."""
    camera_worker: Optional[CameraWorker] = None
    head_tracker = None
    vision_processor: Optional[VisionProcessor] = None

    if not args.no_camera:
        if args.head_tracker is not None:
            try:
                if args.head_tracker == "yolo":
                    from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

                    head_tracker = HeadTracker()
                elif args.head_tracker == "mediapipe":
                    from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]

                    head_tracker = HeadTracker()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Head tracker '%s' failed to initialize: %s — continuing without tracking",
                    args.head_tracker,
                    e,
                )

        camera_worker = CameraWorker(current_robot, head_tracker)

        if args.local_vision:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from reachy_mini_conversation_app.vision.processors import VisionProcessor",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode < 0:
                raise CameraVisionInitializationError(
                    "Local vision import crashed on this machine. "
                    "Run without --local-vision or install compatible dependencies.",
                )
            try:
                from reachy_mini_conversation_app.vision.processors import initialize_vision_processor

            except ImportError as e:
                raise CameraVisionInitializationError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e

            vision_processor = initialize_vision_processor()
        else:
            logging.getLogger(__name__).info(
                "Using gpt-realtime for vision (default). Use --local-vision for local processing.",
            )

    return camera_worker, vision_processor


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    fmt = "%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
    )

    # Add a rotating file handler so logs are always readable even when
    # the daemon captures stdout/stderr via a socket.
    # Attach to our app logger (not root) so uvicorn's dictConfig can't remove it.
    try:
        from logging.handlers import RotatingFileHandler

        app_logger = logging.getLogger("reachy_mini_conversation_app")
        # Avoid duplicate handlers on repeated calls
        if not any(isinstance(h, RotatingFileHandler) for h in app_logger.handlers):
            fh = RotatingFileHandler(
                "/tmp/r3mn1.log", maxBytes=2 * 1024 * 1024, backupCount=2,
            )
            fh.setLevel(getattr(logging, log_level, logging.INFO))
            fh.setFormatter(logging.Formatter(fmt))
            app_logger.addHandler(fh)
            # Ensure this logger processes records (don't rely solely on root level)
            app_logger.setLevel(getattr(logging, log_level, logging.INFO))
            app_logger.info("File logging initialized: /tmp/r3mn1.log")
    except Exception:
        pass  # best-effort — don't crash if /tmp is unwritable

    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
