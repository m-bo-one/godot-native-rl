#!/usr/bin/env bash
# Train chase on the pure-NumPy twin (no Godot, no socket) and convert the actor to ncnn (#37).
#   1. SB3 PPO over N in-process ChaseTwinEnv copies -> models/chase_twin.pt (+ shape sidecar)
#   2. export_to_ncnn.py converts the .pt -> models/chase_twin.ncnn.{param,bin}
# The twin reproduces the Godot chase dynamics, so the net deploys straight into the chase scenes.
#
# Env: TIMESTEPS=300000 N_ENVS=8 OUT=models/chase_twin.pt OUTDIR=models PY=.venv-train/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-300000}"
N_ENVS="${N_ENVS:-8}"
OUT="${OUT:-models/chase_twin.pt}"
OUTDIR="${OUTDIR:-models}"

echo "== Training chase on the NumPy twin (timesteps=$TIMESTEPS, n_envs=$N_ENVS, no Godot) =="
"$PY" scripts/train_chase_twin.py --timesteps "$TIMESTEPS" --n_envs "$N_ENVS" \
	--save_model_path "${OUT%.pt}.zip" --out "$OUT"

echo "== Converting $OUT -> ncnn =="
"$PY" scripts/export_to_ncnn.py "$OUT" --outdir "$OUTDIR"

echo "== Done: $OUTDIR/$(basename "${OUT%.pt}").ncnn.{param,bin} =="
