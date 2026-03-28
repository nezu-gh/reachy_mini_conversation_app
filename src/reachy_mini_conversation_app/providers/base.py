"""Base class for conversation providers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from fastrtc import AsyncStreamHandler


class ConversationProvider(AsyncStreamHandler, ABC):
    """Abstract base for conversation backends.

    Extends fastrtc's AsyncStreamHandler with semantic methods that each
    provider must implement (or accept the no-op defaults).

    No ``__init__`` is defined here intentionally — avoids MRO argument
    conflicts in the diamond with AsyncStreamHandler when subclasses also
    inherit from a concrete handler such as OpenaiRealtimeHandler.
    """

    @abstractmethod
    async def apply_personality(self, profile: Optional[str]) -> str:
        """Apply a personality profile at runtime.

        Args:
            profile: Profile name, or None to reset to default.

        Returns:
            A short status message for UI feedback.

        """

    @abstractmethod
    async def get_available_voices(self) -> List[str]:
        """Return the list of voices supported by this provider."""

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Notify the provider that the robot has been idle.

        Default implementation is a no-op.  Providers that want to react
        to silence (e.g. send a nudge message) can override this.

        Args:
            idle_duration: Seconds since the last interaction.

        """

    def get_tool_specs_override(self) -> Optional[List[dict]]:
        """Return a custom tool-spec list, or None to use the default.

        Returns None by default so existing profile-based tool loading
        is unaffected.
        """
        return None
