#!/usr/bin/env bash
# launcher.sh — Start the conversation app with crash recovery.
#
# This script is designed to be called by systemd (via the .service file)
# or directly.  It activates the virtualenv, sets environment variables,
# and exec's into Python so that systemd's Restart=always can handle
# automatic restarts on crash.
#
# Usage:
#   ./scripts/launcher.sh              # uses defaults
#   PROVIDER=pipecat ./scripts/launcher.sh  # override provider

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtualenv
VENV="${REACHY_VENV:-/venvs/apps_venv}"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# Ensure GStreamer plugins are available (needed for camera/audio)
export GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-}:/opt/gst-plugins-rs/lib/aarch64-linux-gnu/"

# Load .env from the app directory
cd "$APP_DIR"

# Unbuffered Python output so logs appear immediately in journalctl
exec python -u -m reachy_mini_conversation_app.main "$@"
