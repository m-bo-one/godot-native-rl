#!/usr/bin/env bash
# Orchestrates CleanRL (single-file PPO) training of the Sorter agent with the attention encoder
# (#46/#258) over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot Sorter training scene (connects as client)
#   3. wait for the trainer to finish (it exports the ncnn deploy artifact, then Godot quits)
# Mirrors scripts/train_cleanrl.sh; the trainer exports via the DIRECT attention exporter, not ONNX.
set -euo pipefail
cd "$(dirname "$0")/.."

# Unbuffered stdout so per-update progress streams live even when redirected to a file/pipe.
export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/sorter/sorter_train_parallel.tscn}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-models/sorter_attention.pt}"
# Directory + stem for the exported <stem>.ncnn.{param,bin} (the ncnn deploy artifact).
OUTDIR="${OUTDIR:-models}"
STEM="${STEM:-sorter_attention}"

echo "Starting CleanRL Sorter trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_sorter_cleanrl.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
	--action_repeat "$ACTION_REPEAT" --save_model_path "$SAVE_MODEL_PATH" \
	--outdir "$OUTDIR" --stem "$STEM" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot Sorter training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

# Wait for the trainer to finish; then make sure Godot is gone.
set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
