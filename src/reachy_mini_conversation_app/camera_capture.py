"""Subprocess-based camera frame capture.

Works around a GLib main context threading issue where ``unixfdsrc``
in Python fails to transition from PAUSED → PLAYING when run alongside
other GStreamer pipelines (audio) in the same process.

Uses ``gst-launch-1.0`` as a subprocess to capture a single JPEG frame
from the daemon's camera IPC socket.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_CAMERA_SOCKET = "/tmp/reachymini_camera_socket"

# GStreamer pipeline that captures one frame as JPEG
_GST_PIPELINE = (
    f"unixfdsrc socket-path={_CAMERA_SOCKET} num-buffers=5 "
    "! queue "
    "! videoconvert "
    "! jpegenc quality=90 "
    "! filesink location={output}"
)


def capture_frame(timeout: float = 5.0) -> Optional[npt.NDArray[np.uint8]]:
    """Capture a single BGR frame from the robot's camera.

    Returns the frame as a numpy array (H, W, 3) or None on failure.
    """
    socket_path = Path(_CAMERA_SOCKET)
    if not socket_path.exists():
        logger.warning("Camera socket not found: %s", _CAMERA_SOCKET)
        return None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        output_path = tmp.name

    pipeline = _GST_PIPELINE.format(output=output_path)

    try:
        result = subprocess.run(
            ["gst-launch-1.0", "-e"] + pipeline.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            logger.warning("gst-launch failed: %s", result.stderr[:200])
            return None

        # Read the JPEG file
        frame = cv2.imread(output_path)
        if frame is None:
            logger.warning("Failed to decode captured frame")
            return None

        return frame

    except subprocess.TimeoutExpired:
        logger.warning("Camera capture timed out after %.1fs", timeout)
        return None
    except FileNotFoundError:
        logger.warning("gst-launch-1.0 not found — install GStreamer tools")
        return None
    except Exception as e:
        logger.warning("Camera capture error: %s", e)
        return None
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass
