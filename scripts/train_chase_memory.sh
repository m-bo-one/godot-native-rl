#!/usr/bin/env bash
# Memory chase (#378): sb3-contrib RecurrentPPO over the godot-rl bridge.
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene once the port is actually listening
#   3. wait for the trainer (it exports the LSTM actor to ncnn + sidecar, then Godot quits)
# Requires the opt-in add-on: .venv-train/bin/pip install -r requirements-recurrent.txt
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-500000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
N_STEPS="${N_STEPS:-256}"
BASE_PORT="${BASE_PORT:-11008}"
SCENE="${SCENE:-res://examples/chase_the_target/chase_memory_train_parallel.tscn}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-models/chase_memory.zip}"
OUTDIR="${OUTDIR:-models}"

echo "Starting RecurrentPPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_chase_memory.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
	--action_repeat "$ACTION_REPEAT" --n_steps "$N_STEPS" \
	--save_model_path "$SAVE_MODEL_PATH" --outdir "$OUTDIR" &
TRAINER_PID=$!

# Launch Godot only once the trainer's server port is bound (listen-poll, #12/#286 startup-race
# class: a cold torch import can exceed Godot's 10s connect window). Never probe-connect — that
# would consume the trainer's accept().
port_listening() {
	if command -v lsof >/dev/null 2>&1; then
		lsof -iTCP:"$BASE_PORT" -sTCP:LISTEN >/dev/null 2>&1
	else
		ss -ltn 2>/dev/null | grep -q ":$BASE_PORT "
	fi
}
for _ in $(seq 1 180); do
	port_listening && break
	kill -0 "$TRAINER_PID" 2>/dev/null || { echo "Trainer died before binding $BASE_PORT"; exit 1; }
	sleep 1
done

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
