#!/usr/bin/env bash
set -o errexit -o pipefail -o nounset

session_name="$1"
tmux new -s $session_name -d
tmux rename-window -t $session_name queue
tmux send-keys -t $session_name "guild-env" ENTER
tmux new-window -n run -t $session_name
tmux send-keys -t $session_name "guild-env" ENTER
tmux new-window -n compare -t $session_name
tmux send-keys -t $session_name "guild-env" ENTER
tmux new-window -t $session_name
tmux send-keys -t $session_name "guild-env" ENTER
tmux setw -t $session_name -g mouse on
tmux a -t $session_name
