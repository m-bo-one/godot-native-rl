#!/usr/bin/env bash
# Orchestrates SB3 PPO training of the Sorter agent with the attention features extractor
# (#46/#259 M3) over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot Sorter training scene (connects as client)
#   3. wait for the trainer to finish (it exports the ncnn deploy artifact, then Godot quits)
# Mirrors scripts/train_chase.sh; the SB3 policy uses AttentionFeaturesExtractor + net_arch=[] and
# exports via the DIRECT attention exporter (same ncnn artifact shape as the CleanRL M2 trainer).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_STEPS="${NUM_STEPS:-256}"   # rollout length; lower it for a fast smoke under the 8-world scene
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/sorter/sorter_train_parallel.tscn}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-models/sorter_attention_sb3.zip}"
OUTDIR="${OUTDIR:-models}"
STEM="${STEM:-sorter_attention_sb3}"

echo "Starting SB3 Sorter trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_sorter_sb3.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
	--action_repeat "$ACTION_REPEAT" --n_steps "$NUM_STEPS" --save_model_path "$SAVE_MODEL_PATH" \
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
