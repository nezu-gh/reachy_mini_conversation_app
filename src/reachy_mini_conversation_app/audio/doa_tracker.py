"""Direction of Arrival (DoA) head tracker.

Uses the ReSpeaker mic array's DoA reading to gently orient the robot's
head toward the speaker.  Runs as a background thread, polling the SDK's
``get_DoA()`` at ~5 Hz and updating ``MovementManager.set_speech_offsets``
with a smoothed yaw offset.

Enable via ``ENABLE_DOA_TRACKING=1`` environment variable.

Coordinate mapping (ReSpeaker → robot head yaw):
    ReSpeaker: 0 rad = left, π/2 rad = front, π rad = right
    Robot yaw: positive = turn left, negative = turn right (radians)
    Mapping: yaw_offset = (π/2 - doa_angle) * gain
"""

from __future__ import annotations

import math
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# Smoothing: exponential moving average weight (0 = ignore new, 1 = no smoothing)
_EMA_ALPHA = 0.3
# Max yaw offset in radians (~15 degrees) — prevent extreme head turns
_MAX_YAW_RAD = math.radians(15)
# Gain: how aggressively to track the speaker (< 1 = subtle)
_GAIN = 0.5
# Poll interval in seconds
_POLL_INTERVAL = 0.2


class DoATracker:
    """Background thread that orients the robot's head toward the speaker.

    Parameters
    ----------
    robot : ReachyMini
        Robot instance with media.get_DoA() available.
    set_offsets : callable
        Callback to set secondary head offsets ``(x, y, z, roll, pitch, yaw)``.
        Typically ``MovementManager.set_speech_offsets``.

    """

    def __init__(
        self,
        robot: "ReachyMini",
        set_offsets: Callable[[Tuple[float, float, float, float, float, float]], None],
    ) -> None:
        self._robot = robot
        self._set_offsets = set_offsets
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._enabled = True  # can be toggled by VAD state
        self._smoothed_yaw: float = 0.0

    def start(self) -> None:
        """Start the DoA polling thread."""
        if self._thread is not None:
            return
        # Verify DoA is available
        try:
            result = self._robot.media.get_DoA()
            if result is None:
                logger.info("DoATracker: no ReSpeaker device — disabled")
                return
        except Exception as exc:
            logger.info("DoATracker: get_DoA() failed (%s) — disabled", exc)
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="doa-tracker"
        )
        self._thread.start()
        logger.info("DoATracker: started (poll every %.1fs)", _POLL_INTERVAL)

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Reset head to neutral
        self._set_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        logger.info("DoATracker: stopped")

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable tracking (e.g., disable during TTS playback)."""
        self._enabled = enabled
        if not enabled:
            # Smoothly return to neutral
            self._smoothed_yaw *= 0.5

    def _poll_loop(self) -> None:
        """Background loop: read DoA, smooth, apply yaw offset."""
        while self._running:
            try:
                if not self._enabled:
                    time.sleep(_POLL_INTERVAL)
                    continue

                result = self._robot.media.get_DoA()
                if result is None:
                    time.sleep(_POLL_INTERVAL)
                    continue

                angle_rad, speech_detected = result

                if not speech_detected:
                    # No speech → slowly drift back to center
                    self._smoothed_yaw *= 0.95
                    self._apply_yaw()
                    time.sleep(_POLL_INTERVAL)
                    continue

                # Map ReSpeaker angle to robot yaw offset
                # ReSpeaker: 0=left, π/2=front, π=right
                # Robot: positive yaw = turn left
                raw_yaw = (math.pi / 2 - angle_rad) * _GAIN

                # Clamp
                raw_yaw = max(-_MAX_YAW_RAD, min(_MAX_YAW_RAD, raw_yaw))

                # Exponential moving average
                self._smoothed_yaw = (
                    _EMA_ALPHA * raw_yaw + (1 - _EMA_ALPHA) * self._smoothed_yaw
                )

                self._apply_yaw()

            except Exception as exc:
                logger.debug("DoATracker poll error: %s", exc)

            time.sleep(_POLL_INTERVAL)

    def _apply_yaw(self) -> None:
        """Push the smoothed yaw offset to the movement manager."""
        # Only apply yaw — keep all other offsets at zero
        self._set_offsets((0.0, 0.0, 0.0, 0.0, 0.0, self._smoothed_yaw))
