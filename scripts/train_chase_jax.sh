#!/usr/bin/env bash
# Train chase on the JAX batch twin (whole-batch jit rollouts, no Godot, no socket) and convert
# the actor to ncnn (#361):
#   1. PureJaxRL-style PPO over chase_twin_jax.batched_step -> models/chase_jax_twin.pt (+ sidecar)
#      (measured ~100k env-steps/s on CPU at batch 128 — ~47x the NumPy twin, ~3 orders of
#      magnitude past the wire bridge; the same code runs unchanged on GPU)
#   2. export_to_ncnn.py converts the .pt -> models/chase_jax_twin.ncnn.{param,bin}
# Deps: the OPTIONAL requirements-jax.txt add-on — install once:
#   .venv-train/bin/pip install -r requirements-jax.txt
#
# Env: TIMESTEPS=1000000 NUM_ENVS=128 NUM_STEPS=128 OUT=models/chase_jax_twin.pt OUTDIR=models
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-128}"
OUT="${OUT:-models/chase_jax_twin.pt}"
OUTDIR="${OUTDIR:-models}"

echo "== Training chase on the JAX batch twin (timesteps=$TIMESTEPS, num_envs=$NUM_ENVS, no Godot) =="
"$PY" scripts/train_chase_jax.py --timesteps "$TIMESTEPS" --num_envs "$NUM_ENVS" \
	--num_steps "$NUM_STEPS" --out "$OUT"

echo "== Converting $OUT -> ncnn =="
"$PY" scripts/export_to_ncnn.py "$OUT" --outdir "$OUTDIR"

echo "== Done: $OUTDIR/$(basename "${OUT%.pt}").ncnn.{param,bin} =="
