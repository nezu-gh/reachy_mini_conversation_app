"""Tests for tool result truncation."""

import json

from reachy_mini_conversation_app.tools.core_tools import (
    _MAX_TOOL_RESULT_BYTES,
    _truncate_result,
)


class TestTruncateResult:
    def test_small_result_unchanged(self):
        result = {"status": "ok", "message": "done"}
        assert _truncate_result(result) == result

    def test_empty_result_unchanged(self):
        result = {}
        assert _truncate_result(result) == result

    def test_large_result_truncated(self):
        # Create a result larger than 4KB
        result = {"data": "x" * 5000}
        truncated = _truncate_result(result)
        assert "truncated" in json.dumps(truncated)
        assert truncated != result

    def test_truncated_output_under_limit(self):
        result = {"data": "x" * 10000}
        truncated = _truncate_result(result)
        assert len(json.dumps(truncated)) <= _MAX_TOOL_RESULT_BYTES

    def test_exact_boundary(self):
        # Result that is exactly at the limit should pass through
        # Build a result that's just under the limit
        padding = "x" * (_MAX_TOOL_RESULT_BYTES - 20)
        result = {"d": padding}
        serialized = json.dumps(result)
        if len(serialized) <= _MAX_TOOL_RESULT_BYTES:
            assert _truncate_result(result) == result
