"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import signal
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def _build_handler(provider: str, deps: Any, gradio_mode: bool, instance_path: Any) -> Any:
    """Return the appropriate conversation handler for *provider*."""
    if provider == "pipecat":
        from reachy_mini_conversation_app.providers.pipecat_provider import PipecatProvider

        return PipecatProvider(deps, gradio_mode=gradio_mode, instance_path=instance_path)
    from reachy_mini_conversation_app.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(deps, gradio_mode=gradio_mode, instance_path=instance_path)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["host"] = args.robot_name
                robot_kwargs["connection_mode"] = "network"

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(args, robot)
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_processor=vision_processor,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = _build_handler(getattr(args, "provider", "openai"), deps, args.gradio, instance_path)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )

        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # Wake the robot up so it's ready to interact
    try:
        robot.enable_motors()
        robot.wake_up()
        logger.info("Robot woke up successfully")
    except Exception as e:
        logger.warning("Failed to wake up robot: %s", e)

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    # Handle SIGTERM (from systemd) the same as KeyboardInterrupt
    def _sigterm_handler(signum: int, frame: Any) -> None:
        logger.info("Received SIGTERM, initiating shutdown...")
        if stream_manager is not None:
            try:
                stream_manager.close()
            except Exception:
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        stream_manager.launch()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received, closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()

        # Put robot to sleep before shutting down
        try:
            robot.goto_sleep()
            robot.disable_motors()
        except Exception as e:
            logger.debug(f"Error during robot sleep: {e}")

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def __init__(self, **kwargs: Any) -> None:
        args, _ = parse_args()
        # When --robot-name is set, the robot is remote — force network mode
        # even if something else is listening on localhost:8000.
        if args.robot_name is not None:
            self._force_network = True
        else:
            self._force_network = False
        super().__init__(**kwargs)
        if self._force_network:
            self.daemon_on_localhost = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    # Pass robot host from --robot-name / REACHY_ROBOT_NAME env var
    _args, _ = parse_args()
    _robot_kwargs: Dict[str, Any] = {}
    if _args.robot_name is not None:
        _robot_kwargs["host"] = _args.robot_name
    try:
        app.wrapped_run(**_robot_kwargs)
    except KeyboardInterrupt:
        app.stop()
