#!/usr/bin/env bash
# Orchestrates SKRL PPO training over the godot_rl wire protocol, then converts the exported
# actor to ncnn (#25):
#   1. start the SKRL trainer in .venv-train (opens server on BASE_PORT, blocks for Godot);
#      it traces the trained policy trunk to TorchScript + shape sidecar itself
#   2. launch the headless Godot chase training scene (connects on BASE_PORT)
#   3. wait for the trainer; kill Godot
#   4. convert .pt -> ncnn + parity check (export_to_ncnn.py -> .venv/bin/pnnx)
# Fifth backend alongside SB3, CleanRL, SampleFactory and RLlib. Ecosystem interop: see #25.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-200000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
ROLLOUTS="${ROLLOUTS:-512}"
OUTDIR="${OUTDIR:-models}"
SCENE="${SCENE:-res://examples/chase_the_target/chase_the_target_train.tscn}"
ATOL="${ATOL:-5e-2}"

PT_PATH="$OUTDIR/chase_skrl_policy.pt"

cleanup() {
	set +e
	[ -n "${GODOT_PID:-}" ] && kill "$GODOT_PID" 2>/dev/null
	[ -n "${TRAINER_PID:-}" ] && kill "$TRAINER_PID" 2>/dev/null
	return 0
}
trap cleanup EXIT

echo "Starting SKRL trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
"$PY_TRAIN" scripts/train_skrl.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --rollouts "$ROLLOUTS" \
	--out "$PT_PATH" &
TRAINER_PID=$!

# torch + skrl import take a few seconds before the env server opens.
sleep "${STARTUP_DELAY:-10}"

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" \
	"speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" "port=$BASE_PORT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
TRAINER_PID=""
kill "$GODOT_PID" 2>/dev/null
GODOT_PID=""
set -e
echo "Trainer exited with code $TRAINER_RC"
[ "$TRAINER_RC" -ne 0 ] && exit "$TRAINER_RC"

echo "Converting TorchScript -> ncnn (+ parity check)..."
"$PY_TRAIN" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR" --atol "$ATOL"

echo "Done: $OUTDIR/chase_skrl_policy.ncnn.{param,bin}"
