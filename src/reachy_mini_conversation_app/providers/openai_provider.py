"""OpenAI Realtime provider — thin bridge to OpenaiRealtimeHandler."""

from reachy_mini_conversation_app.providers.base import ConversationProvider
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler


class OpenAIProvider(OpenaiRealtimeHandler, ConversationProvider):
    """OpenAI Realtime API conversation backend.

    Python's C3 MRO resolves the diamond:
      OpenAIProvider → OpenaiRealtimeHandler → AsyncStreamHandler
                     → ConversationProvider  → AsyncStreamHandler

    OpenaiRealtimeHandler already implements all abstract methods
    (``apply_personality`` at line 142, ``get_available_voices`` at line 788),
    so this class body can remain empty.
    """
