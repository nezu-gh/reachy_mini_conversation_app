"""LLM Router — selects model configuration based on intent classification.

Routes casual/simple messages to a fast/small LLM and complex messages
to the full model.  Falls back to the default model if no fast model
is configured.

Env vars:
    LLM_FAST_BASE_URL  — base URL for fast LLM (optional)
    LLM_FAST_MODEL     — model name for fast LLM (optional)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from reachy_mini_conversation_app.llm.intent_classifier import Intent

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for a single LLM endpoint."""

    base_url: str
    model: str
    max_tokens: int = 4096


class LLMRouter:
    """Routes intent categories to LLM configurations."""

    def __init__(
        self,
        default_base_url: str,
        default_model: str,
        *,
        fast_base_url: Optional[str] = None,
        fast_model: Optional[str] = None,
    ) -> None:
        self.default = LLMConfig(base_url=default_base_url, model=default_model)
        self.fast: Optional[LLMConfig] = None

        _fast_url = fast_base_url or os.environ.get("LLM_FAST_BASE_URL", "")
        _fast_model = fast_model or os.environ.get("LLM_FAST_MODEL", "")

        if _fast_url and _fast_model:
            self.fast = LLMConfig(
                base_url=_fast_url,
                model=_fast_model,
                max_tokens=1024,
            )
            logger.info("LLM Router: fast=%s@%s, default=%s@%s",
                        _fast_model, _fast_url, default_model, default_base_url)
        else:
            logger.info("LLM Router: no fast model configured, using default for all intents")

    def route(self, intent: Intent) -> LLMConfig:
        """Select LLM config based on intent."""
        if self.fast is None:
            return self.default

        if intent == Intent.CASUAL:
            return self.fast
        return self.default
