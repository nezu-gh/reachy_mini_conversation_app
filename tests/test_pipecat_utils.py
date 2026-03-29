"""Tests for pure functions extracted from pipecat_provider.

These functions have no pipecat dependency and can be tested without
installing pipecat-ai.
"""

import pytest

from reachy_mini_conversation_app.providers.pipecat_provider import (
    _extract_inline_tool_calls,
    _is_noise,
    _strip_inline_tool_calls,
    _trim_context,
)


# ---------------------------------------------------------------------------
# _is_noise
# ---------------------------------------------------------------------------


class TestIsNoise:
    def test_empty_string(self):
        assert _is_noise("") is True

    def test_single_char(self):
        assert _is_noise("a") is True

    def test_whitespace_only(self):
        assert _is_noise("   ") is True

    def test_cjk_dominant(self):
        # Pure CJK — typical Qwen ASR hallucination on ambient noise
        assert _is_noise("你好世界") is True

    def test_cjk_minority(self):
        # Mostly Latin with one CJK char — valid speech
        assert _is_noise("Hello 你 world") is False

    def test_punctuation_only(self):
        assert _is_noise("...") is True
        assert _is_noise("!?!") is True

    def test_valid_short_speech(self):
        assert _is_noise("ok") is False

    def test_valid_sentence(self):
        assert _is_noise("Hello, how are you?") is False

    def test_mixed_valid(self):
        assert _is_noise("I said hello!") is False

    def test_japanese_kana(self):
        # Hiragana/Katakana are also filtered as noise
        assert _is_noise("あいう") is True
        assert _is_noise("カタカナ") is True

    def test_korean(self):
        assert _is_noise("한국어") is True

    def test_cjk_below_threshold(self):
        # One CJK char in a long string — below 30% threshold
        text = "This is a normal English sentence with one 你 character"
        assert _is_noise(text) is False


# ---------------------------------------------------------------------------
# _trim_context
# ---------------------------------------------------------------------------


class TestTrimContext:
    def test_empty_returns_none(self):
        assert _trim_context([]) is None

    def test_under_limit_returns_none(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        assert _trim_context(msgs) is None

    def test_at_limit_returns_none(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(40)]
        assert _trim_context(msgs) is None

    def test_over_limit_trims(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        result = _trim_context(msgs)
        assert result is not None
        assert len(result) == 40
        # Should keep the last 40
        assert result[0]["content"] == "msg 10"
        assert result[-1]["content"] == "msg 49"

    def test_preserves_system_messages(self):
        system = [{"role": "system", "content": "You are a robot."}]
        user_msgs = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        result = _trim_context(system + user_msgs)
        assert result is not None
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a robot."
        # system + 40 non-system = 41 total
        assert len(result) == 41

    def test_preserves_multiple_system_messages(self):
        system = [
            {"role": "system", "content": "instruction 1"},
            {"role": "system", "content": "instruction 2"},
        ]
        user_msgs = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        result = _trim_context(system + user_msgs)
        assert result is not None
        assert len(result) == 42  # 2 system + 40 non-system
        assert result[0]["content"] == "instruction 1"
        assert result[1]["content"] == "instruction 2"

    def test_custom_max_turns(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = _trim_context(msgs, max_turns=10)
        assert result is not None
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _extract_inline_tool_calls
# ---------------------------------------------------------------------------


class TestExtractInlineToolCalls:
    def test_function_style_micro_expression(self):
        calls = _extract_inline_tool_calls("micro_expression(happy)")
        assert calls == [("micro_expression", "happy")]

    def test_function_style_with_quotes(self):
        calls = _extract_inline_tool_calls("micro_expression('laugh')")
        assert calls == [("micro_expression", "laugh")]

    def test_function_style_play_emotion(self):
        calls = _extract_inline_tool_calls("play_emotion(joy)")
        assert calls == [("play_emotion", "joy")]

    def test_function_style_dance(self):
        calls = _extract_inline_tool_calls("dance(groovy)")
        assert calls == [("dance", "groovy")]

    def test_markdown_style(self):
        calls = _extract_inline_tool_calls("*Micro-expression: laugh*")
        assert calls == [("micro_expression", "laugh")]

    def test_markdown_style_underscore(self):
        calls = _extract_inline_tool_calls("*micro_expression: happy*")
        assert calls == [("micro_expression", "happy")]

    def test_no_match(self):
        calls = _extract_inline_tool_calls("Hello, how are you?")
        assert calls == []

    def test_mixed_text_and_calls(self):
        text = "Sure! micro_expression(happy) I'm glad to help."
        calls = _extract_inline_tool_calls(text)
        assert len(calls) == 1
        assert calls[0] == ("micro_expression", "happy")

    def test_multiple_calls(self):
        text = "micro_expression(happy) dance(groovy)"
        calls = _extract_inline_tool_calls(text)
        assert len(calls) == 2

    def test_both_patterns(self):
        text = "micro_expression(happy) *Micro-expression: laugh*"
        calls = _extract_inline_tool_calls(text)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# _strip_inline_tool_calls
# ---------------------------------------------------------------------------


class TestStripInlineToolCalls:
    def test_strips_function_style(self):
        result = _strip_inline_tool_calls("Hello micro_expression(happy) world")
        assert "micro_expression" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_markdown_style(self):
        result = _strip_inline_tool_calls("Hello *Micro-expression: laugh* world")
        assert "Micro-expression" not in result

    def test_no_match_unchanged(self):
        text = "Hello, how are you?"
        assert _strip_inline_tool_calls(text) == text

    def test_empty_after_strip(self):
        result = _strip_inline_tool_calls("micro_expression(happy)")
        assert result == ""
