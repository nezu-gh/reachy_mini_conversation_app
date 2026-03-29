#!/usr/bin/env bash
# Open an SSH shell on the robot, or run a remote command.
#
# Usage:
#   ./scripts/robot-shell.sh                # interactive shell
#   ./scripts/robot-shell.sh ps aux         # run a command
#   ./scripts/robot-shell.sh htop           # interactive tools work too

source "$(dirname "$0")/_common.sh"

if [[ $# -eq 0 ]]; then
    echo "Connecting to $ROBOT_USER@$ROBOT_HOST ..."
    sshpass -p "$ROBOT_PASS" ssh $SSH_OPTS -t "$ROBOT_USER@$ROBOT_HOST"
else
    _ssh "$@"
fi
