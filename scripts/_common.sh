#!/usr/bin/env bash
# Shared robot connection helpers for all scripts.
# Source this file: source "$(dirname "$0")/_common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load robot connection from .env if present
if [[ -f "$PROJECT_DIR/.env.r3mn1" ]]; then
    ROBOT_HOST="${ROBOT_HOST:-$(grep -oP '(?<=REACHY_ROBOT_NAME=).*' "$PROJECT_DIR/.env.r3mn1" 2>/dev/null || echo "192.168.178.127")}"
    LOCAL_VM_IP="${LOCAL_VM_IP:-$(grep -oP '(?<=LOCAL_VM_IP=).*' "$PROJECT_DIR/.env.r3mn1" 2>/dev/null || echo "192.168.178.155")}"
fi
ROBOT_HOST="${ROBOT_HOST:-192.168.178.127}"
ROBOT_USER="${ROBOT_USER:-pollen}"
ROBOT_PASS="${ROBOT_PASS:-root}"
ROBOT_APP_DIR="/home/pollen/r3-mn1"
LOCAL_VM_IP="${LOCAL_VM_IP:-192.168.178.155}"

SSH_OPTS="-o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o ConnectTimeout=5"

_ssh() {
    sshpass -p "$ROBOT_PASS" ssh $SSH_OPTS "$ROBOT_USER@$ROBOT_HOST" "$@"
}

_rsync() {
    sshpass -p "$ROBOT_PASS" rsync -az --delete -e "ssh $SSH_OPTS" "$@"
}

_daemon_api() {
    _ssh "curl -s --max-time 30 http://localhost:8000$1 2>/dev/null" || echo '{"error": "request failed"}'
}
