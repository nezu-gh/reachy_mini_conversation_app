"""Tests for pipecat_provider module-level utilities.

Tests health-check probes and service validation. These mock urllib
so they don't require actual network access or running services.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.providers.pipecat_provider import (
    _check_services,
    _probe_service,
)


class TestProbeService:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is True

    def test_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False

    def test_connection_refused(self):
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False

    def test_url_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("unreachable"),
        ):
            assert _probe_service("TestSvc", "http://localhost:1234/v1") is False


class TestCheckServices:
    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    def test_all_healthy_no_retry(self, mock_probe):
        mock_probe.return_value = True
        _check_services()
        # Should be called exactly 3 times (one per service)
        assert mock_probe.call_count == 3

    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    @patch("time.sleep")  # prevent actual sleeping in tests
    def test_retries_on_failure(self, mock_sleep, mock_probe, caplog):
        # Fail twice, then succeed
        call_count = {"n": 0}

        def side_effect(name, url, timeout=5.0):
            call_count["n"] += 1
            if call_count["n"] <= 6:  # first 2 rounds × 3 services = 6 calls
                return False
            return True

        mock_probe.side_effect = side_effect
        with caplog.at_level(logging.WARNING):
            _check_services()
        # 3 rounds × 3 services = 9 total probe calls
        assert mock_probe.call_count == 9

    @patch("reachy_mini_conversation_app.providers.pipecat_provider._probe_service")
    @patch("time.sleep")
    def test_logs_error_after_all_retries(self, mock_sleep, mock_probe, caplog):
        mock_probe.return_value = False
        with caplog.at_level(logging.ERROR):
            _check_services()
        assert "still unreachable after 3 attempts" in caplog.text
