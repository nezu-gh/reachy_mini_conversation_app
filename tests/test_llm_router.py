"""Tests for intent classifier and LLM router."""

import pytest

from reachy_mini_conversation_app.llm.intent_classifier import Intent, classify
from reachy_mini_conversation_app.llm.llm_router import LLMRouter


class TestIntentClassifier:
    """Parametrized tests for intent classification."""

    @pytest.mark.parametrize("text,expected", [
        ("hi", Intent.CASUAL),
        ("hello", Intent.CASUAL),
        ("hey!", Intent.CASUAL),
        ("thanks", Intent.CASUAL),
        ("yeah", Intent.CASUAL),
        ("ok", Intent.CASUAL),
        ("bye", Intent.CASUAL),
        ("good morning", Intent.CASUAL),
        ("how are you?", Intent.CASUAL),
    ])
    def test_casual_intents(self, text: str, expected: Intent):
        result = classify(text)
        assert result.intent == expected, f"'{text}' classified as {result.intent}, expected {expected}"

    @pytest.mark.parametrize("text,expected", [
        ("dance for me", Intent.COMMAND),
        ("move your head left", Intent.COMMAND),
        ("take a photo", Intent.COMMAND),
        ("show me an emotion", Intent.COMMAND),
        ("remember my name is Alex", Intent.COMMAND),
    ])
    def test_command_intents(self, text: str, expected: Intent):
        result = classify(text)
        assert result.intent == expected, f"'{text}' classified as {result.intent}, expected {expected}"

    @pytest.mark.parametrize("text,expected", [
        ("explain quantum computing", Intent.REASONING),
        ("why is the sky blue?", Intent.REASONING),
        ("what is the difference between Python and JavaScript", Intent.REASONING),
    ])
    def test_reasoning_intents(self, text: str, expected: Intent):
        result = classify(text)
        assert result.intent == expected, f"'{text}' classified as {result.intent}, expected {expected}"

    @pytest.mark.parametrize("text,expected", [
        ("tell me a joke", Intent.CREATIVE),
        ("write me a poem about robots", Intent.CREATIVE),
        ("make up a story", Intent.CREATIVE),
    ])
    def test_creative_intents(self, text: str, expected: Intent):
        result = classify(text)
        assert result.intent == expected, f"'{text}' classified as {result.intent}, expected {expected}"

    def test_empty_text_is_casual(self):
        result = classify("")
        assert result.intent == Intent.CASUAL

    def test_confidence_is_in_range(self):
        for text in ["hi", "explain quantum physics", "dance", "tell a joke", ""]:
            result = classify(text)
            assert 0.0 <= result.confidence <= 1.0


class TestLLMRouter:
    def test_default_only_returns_default_for_all(self):
        router = LLMRouter("http://default:3443/v1", "big-model")
        for intent in Intent:
            config = router.route(intent)
            assert config.base_url == "http://default:3443/v1"
            assert config.model == "big-model"

    def test_fast_model_routes_casual(self):
        router = LLMRouter(
            "http://default:3443/v1", "big-model",
            fast_base_url="http://fast:3443/v1",
            fast_model="small-model",
        )
        assert router.route(Intent.CASUAL).model == "small-model"
        assert router.route(Intent.REASONING).model == "big-model"
        assert router.route(Intent.COMMAND).model == "big-model"
        assert router.route(Intent.CREATIVE).model == "big-model"

    def test_fallback_when_no_fast_configured(self):
        router = LLMRouter("http://default:3443/v1", "big-model")
        assert router.fast is None
        config = router.route(Intent.CASUAL)
        assert config.model == "big-model"
