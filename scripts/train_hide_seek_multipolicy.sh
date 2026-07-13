#!/usr/bin/env bash
# Orchestrates MULTI-POLICY self-play training (two distinct policies: seeker + hider):
#   1. start the Python multi-policy trainer (opens server on 11008, waits)
#   2. launch the headless Godot scene (it self-declares multi_policy on its Sync node, so the
#      agents emit distinct policy_names — no --multi-policy cmdline gate; see #73)
#   3. wait for the trainer, then ensure Godot is gone
# SCENE override selects single vs parallel; defaults to the parallel (fast) scene.
# SNAPSHOT_EVERY=N (env-steps) enables #189 simultaneous self-play: BOTH live learners are
# cross-frozen into the opponent pool (POOL_DIR, #29 layout: ncnn + pool.json ELO ledger)
# mid-run — the pool grows during the run, no phase restarts.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-800000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"
POOL_DIR="${POOL_DIR:-models/selfplay_pool}"

echo "Starting multi-policy trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_hide_seek_multipolicy.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
	--snapshot_every "$SNAPSHOT_EVERY" --pool_dir "$POOL_DIR" &
TRAINER_PID=$!

sleep 5

echo "Launching headless Godot scene ($SCENE)..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
