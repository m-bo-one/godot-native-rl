#!/usr/bin/env bash
# Train chase_rays on the pure-NumPy twin (no Godot, no socket) and convert the actor to ncnn (#364).
#   1. SB3 PPO over N in-process ChaseRaysTwinEnv copies -> models/chase_rays_twin.pt (+ shape sidecar)
#   2. export_to_ncnn.py converts the .pt -> models/chase_rays_twin.ncnn.{param,bin}
# Rays are analytic in the twin and REAL physics casts at deploy — proven equal (test_chase_rays.gd).
#
# Env: TIMESTEPS=400000 N_ENVS=8 OUT=models/chase_rays_twin.pt OUTDIR=models PY=.venv-train/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-400000}"
N_ENVS="${N_ENVS:-8}"
OUT="${OUT:-models/chase_rays_twin.pt}"
OUTDIR="${OUTDIR:-models}"

echo "== Training chase_rays on the NumPy twin (timesteps=$TIMESTEPS, n_envs=$N_ENVS, no Godot) =="
"$PY" scripts/train_chase_rays_twin.py --timesteps "$TIMESTEPS" --n_envs "$N_ENVS" \
	--save_model_path "${OUT%.pt}.zip" --out "$OUT"

echo "== Converting $OUT -> ncnn =="
"$PY" scripts/export_to_ncnn.py "$OUT" --outdir "$OUTDIR"

echo "== Done: $OUTDIR/$(basename "${OUT%.pt}").ncnn.{param,bin} =="
