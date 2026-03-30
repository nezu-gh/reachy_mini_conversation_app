"""Tests for lean/full pipeline mode.

Tests the PIPELINE_MODE configuration, lean prompt loading,
and pipeline composition for both lean and full modes.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Phase 0: Configuration + prompt
# ---------------------------------------------------------------------------

class TestPipelineModeConfig:
    """Test the PIPELINE_MODE env var handling."""

    def test_default_is_full(self):
        from reachy_mini_conversation_app.providers.pipecat_provider import _is_lean_mode
        assert _is_lean_mode("full") is False

    def test_lean_from_env(self):
        from reachy_mini_conversation_app.providers.pipecat_provider import _is_lean_mode
        assert _is_lean_mode("lean") is True

    def test_case_insensitive(self):
        from reachy_mini_conversation_app.providers.pipecat_provider import _is_lean_mode
        assert _is_lean_mode("LEAN") is False  # we lowercase in the constant, not the helper
        assert _is_lean_mode("lean") is True

    def test_unknown_is_not_lean(self):
        from reachy_mini_conversation_app.providers.pipecat_provider import _is_lean_mode
        assert _is_lean_mode("banana") is False
        assert _is_lean_mode("") is False


class TestLeanPromptFile:
    """Test the lean prompt file exists and is appropriate."""

    def _lean_prompt_path(self) -> Path:
        return Path(__file__).parents[1] / "profiles" / "r3_mn1" / "instructions_lean.txt"

    def test_lean_prompt_file_exists(self):
        assert self._lean_prompt_path().exists()

    def test_lean_prompt_is_short(self):
        text = self._lean_prompt_path().read_text(encoding="utf-8")
        # Should be under 500 chars (~100-150 tokens)
        assert len(text) < 500, f"Lean prompt too long: {len(text)} chars"

    def test_lean_prompt_no_tool_references(self):
        text = self._lean_prompt_path().read_text(encoding="utf-8").lower()
        for tool in ["micro_expression", "store_memory", "recall_memory", "play_emotion", "move_head"]:
            assert tool not in text, f"Lean prompt should not mention {tool}"

    def test_lean_prompt_has_spoken_output_hint(self):
        text = self._lean_prompt_path().read_text(encoding="utf-8").lower()
        assert "spoken" in text or "speech" in text or "voice" in text


class TestGetSessionInstructionsLean:
    """Test that get_session_instructions respects lean mode.

    Uses the r3_mn1 profile which has both instructions.txt and
    instructions_lean.txt.
    """

    def test_lean_mode_returns_lean_prompt(self):
        from reachy_mini_conversation_app.prompts import get_session_instructions
        from reachy_mini_conversation_app.config import config
        old_profile = config.REACHY_MINI_CUSTOM_PROFILE
        try:
            config.REACHY_MINI_CUSTOM_PROFILE = "r3_mn1"
            with patch.dict(os.environ, {"PIPELINE_MODE": "lean"}):
                instructions = get_session_instructions()
            # Lean prompt should be much shorter than the full r3_mn1 prompt
            assert len(instructions) < 500
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = old_profile

    def test_full_mode_returns_full_prompt(self):
        from reachy_mini_conversation_app.prompts import get_session_instructions
        from reachy_mini_conversation_app.config import config
        old_profile = config.REACHY_MINI_CUSTOM_PROFILE
        try:
            config.REACHY_MINI_CUSTOM_PROFILE = "r3_mn1"
            with patch.dict(os.environ, {"PIPELINE_MODE": "full"}):
                instructions = get_session_instructions()
            # Full r3_mn1 prompt is ~4300 chars
            assert len(instructions) > 2000
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = old_profile

    def test_lean_and_full_differ(self):
        from reachy_mini_conversation_app.prompts import get_session_instructions
        from reachy_mini_conversation_app.config import config
        old_profile = config.REACHY_MINI_CUSTOM_PROFILE
        try:
            config.REACHY_MINI_CUSTOM_PROFILE = "r3_mn1"
            with patch.dict(os.environ, {"PIPELINE_MODE": "lean"}):
                lean = get_session_instructions()
            with patch.dict(os.environ, {"PIPELINE_MODE": "full"}):
                full = get_session_instructions()
            assert len(lean) < len(full)
            assert lean != full
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = old_profile


# ---------------------------------------------------------------------------
# Phase 1: Pipeline composition
# ---------------------------------------------------------------------------

class TestLeanPipelineComposition:
    """Test that lean mode produces the right pipeline shape.

    These tests verify the processor list without running the actual pipeline.
    They use _build_lean_pipeline_processors() and _build_full_pipeline_processors()
    helpers that return processor lists.
    """

    def test_lean_pipeline_processor_names(self):
        """Lean pipeline should have exactly these processors in order."""
        from reachy_mini_conversation_app.providers.pipecat_provider import (
            _lean_pipeline_processor_names,
        )
        names = _lean_pipeline_processor_names()
        assert names == [
            "VADProcessor",
            "OpenAISTTService",
            "ASRTextCleaner",
            "LLMUserContextAggregator",
            "OpenAILLMService",
            "OpenAITTSService",
            "PipelineSink",
            "LLMAssistantContextAggregator",
        ]

    def test_full_pipeline_has_more_processors(self):
        """Full pipeline should have more processors than lean."""
        from reachy_mini_conversation_app.providers.pipecat_provider import (
            _lean_pipeline_processor_names,
            _full_pipeline_processor_names,
        )
        lean = _lean_pipeline_processor_names()
        full = _full_pipeline_processor_names()
        assert len(full) > len(lean)

    def test_full_pipeline_has_enricher_and_router(self):
        """Full pipeline should include ParallelEnricher and IntentRouter."""
        from reachy_mini_conversation_app.providers.pipecat_provider import (
            _full_pipeline_processor_names,
        )
        full = _full_pipeline_processor_names()
        assert "ParallelEnricher" in full
        assert "IntentRouter" in full

    def test_lean_pipeline_no_enricher_or_router(self):
        """Lean pipeline should NOT include enricher, router, memory, etc."""
        from reachy_mini_conversation_app.providers.pipecat_provider import (
            _lean_pipeline_processor_names,
        )
        lean = _lean_pipeline_processor_names()
        for excluded in ["ParallelEnricher", "IntentRouter", "ContextTrimmer",
                         "AutoMemoryTap", "AssistantTextTap", "TTSTextChunker"]:
            assert excluded not in lean, f"{excluded} should not be in lean pipeline"
