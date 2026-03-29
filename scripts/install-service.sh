#!/usr/bin/env bash
# install-service.sh — Install the conversation app as a systemd service.
#
# Run on the robot (as pollen, with sudo access):
#   cd /home/pollen/r3-mn1 && ./scripts/install-service.sh
#
# After installation:
#   sudo systemctl start reachy-mini-conversation   # start now
#   sudo systemctl status reachy-mini-conversation   # check status
#   journalctl -u reachy-mini-conversation -f        # follow logs
#
# The service will auto-restart on crash (Restart=always, 5s delay).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/reachy-mini-conversation.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: Service file not found: $SERVICE_FILE"
    exit 1
fi

echo "Installing reachy-mini-conversation.service..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/reachy-mini-conversation.service
sudo systemctl daemon-reload
sudo systemctl enable reachy-mini-conversation.service

echo ""
echo "Service installed and enabled (will start on boot)."
echo ""
echo "Commands:"
echo "  sudo systemctl start reachy-mini-conversation    # start now"
echo "  sudo systemctl stop reachy-mini-conversation     # stop"
echo "  sudo systemctl restart reachy-mini-conversation  # restart"
echo "  sudo systemctl status reachy-mini-conversation   # status"
echo "  journalctl -u reachy-mini-conversation -f        # live logs"
