#!/usr/bin/env bash
# Orchestrates Ray/RLlib MULTI-POLICY PPO over the PettingZoo GodotParallelEnv adapter (#123),
# then exports EVERY policy's actor to ncnn:
#   1. start the RLlib trainer in .venv-train (opens server on BASE_PORT, blocks for Godot)
#   2. launch the headless Godot multi-policy hide & seek scene (its Sync sets multi_policy=true,
#      #73, so agents emit distinct policy_names on the wire)
#   3. wait for the trainer; kill Godot (trap cleans up stray ray workers)
#   4. per policy in env_meta.json: RLModule --module_id -> TorchScript .pt + sidecar
#   5. convert each .pt -> ncnn + parity check (export_to_ncnn.py -> .venv/bin/pnnx)
# The canonical-upstream multi-agent combination on top of #110 (RLlib single-policy) + #111
# (PettingZoo adapter). See docs/superpowers/specs/2026-07-12-rllib-multipolicy-pettingzoo-design.md.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_RLLIB="${PY_RLLIB:-.venv-train/bin/python}"
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-200000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-hide_seek_rllib}"
TRAIN_DIR="${TRAIN_DIR:-logs/rllib}"
OUTDIR="${OUTDIR:-models}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn}"
# Declared module set (#374): forward to the trainer so a non-default SCENE emitting other policy
# names (e.g. "predator prey") can declare them here instead of failing loud in validate_policies.
DECLARE_POLICIES="${DECLARE_POLICIES:-seeker hider}"
# Fully-trained runs reach logit magnitudes where benign fp32 torch-vs-ncnn drift slightly
# exceeds the default 1e-2 atol (argmax stays exact and is enforced regardless).
ATOL="${ATOL:-5e-2}"

cleanup() {
	set +e
	[ -n "${GODOT_PID:-}" ] && kill "$GODOT_PID" 2>/dev/null
	[ -n "${TRAINER_PID:-}" ] && kill "$TRAINER_PID" 2>/dev/null
	pkill -f "ray::" 2>/dev/null
	return 0
}
trap cleanup EXIT

echo "Starting RLlib multi-policy trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
# shellcheck disable=SC2086  # DECLARE_POLICIES is a space-separated name list by construction
"$PY_RLLIB" scripts/train_rllib_pettingzoo.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
	--experiment "$EXPERIMENT" --train_dir "$TRAIN_DIR" --policies $DECLARE_POLICIES &
TRAINER_PID=$!

# Race-free startup (the #286 gotcha class): wait until the trainer has actually BOUND the env
# port before launching Godot — a blind sleep loses whenever ray/torch cold-start exceeds it,
# and Godot's connect window (10s) then falls back to HUMAN mode while accept() starves.
# Listen-check WITHOUT connecting (a probe connect would consume the trainer's accept()):
# lsof works on macOS + Linux; fall back to ss where lsof is absent.
port_listening() {
	if command -v lsof >/dev/null 2>&1; then
		lsof -iTCP:"$BASE_PORT" -sTCP:LISTEN >/dev/null 2>&1
	else
		ss -ltn 2>/dev/null | grep -q ":$BASE_PORT "
	fi
}
for _ in $(seq 1 180); do
	port_listening && break
	sleep 1
done

echo "Launching headless Godot multi-policy scene ($SCENE)..."
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

META="$TRAIN_DIR/$EXPERIMENT/env_meta.json"
test -f "$META" || { echo "FAIL: $META not written by the trainer" >&2; exit 1; }
# Pass $META via argv, not string-interpolated into the source (#374): a TRAIN_DIR path with a
# quote or space would otherwise break the inline Python (or worse, inject).
OBS_DIM="$("$PY_TRAIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['obs_dim'])" "$META")"
NVEC="$("$PY_TRAIN" -c "import json,sys;print(' '.join(str(n) for n in json.load(open(sys.argv[1]))['nvec']))" "$META")"
POLICIES="$("$PY_TRAIN" -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['policies']))" "$META")"

for POLICY in $POLICIES; do
	PT_PATH="$OUTDIR/hide_seek_rllib_${POLICY}.pt"
	echo "Exporting policy '$POLICY' -> TorchScript ($PT_PATH)..."
	# shellcheck disable=SC2086  # NVEC is a space-separated int list by construction
	"$PY_RLLIB" scripts/export_rllib_to_torchscript.py --train_dir "$TRAIN_DIR" \
		--experiment "$EXPERIMENT" --module_id "$POLICY" \
		--obs_dim "$OBS_DIM" --nvec $NVEC --out "$PT_PATH"
	echo "Converting '$POLICY' TorchScript -> ncnn (+ parity check)..."
	"$PY_TRAIN" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR" --atol "$ATOL"
done

echo "Done: $OUTDIR/hide_seek_rllib_{$(echo "$POLICIES" | tr ' ' ',')}.ncnn.{param,bin}"
