# Curriculum trainer-driven promotion smoke (#198)

**Status:** approved design
**Issue:** [#198](https://github.com/minigraphx/godot-native-rl/issues/198) — follow-up to #28/#186
**Date:** 2026-07-06

## Problem

The curriculum feature (#28) is covered at every layer *except* the one that actually broke:
the link between a **real trainer** and the curriculum scene. Unit tests (promotion logic,
controller), the wire-override protocol test, and the simulated-episode promotion smoke all
passed while the documented training command was silently broken — `train_chase.sh` ignored
`SCENE=` (fixed in #186, `bc537f4`), so a real run trained the plain scene and **never promoted**.
Only a manual evidence run caught it.

We need a guarded CI smoke that runs an actual SB3 trainer against the curriculum scene and
asserts a promotion fires, so this class of trainer↔scene regression is caught automatically.

## Goal & non-goals

- **Goal:** prove the trainer → curriculum scene → `record_episode` → promotion **plumbing**
  fires end-to-end under a real trainer, in CI, quickly.
- **Non-goal:** assert the policy *learns*. The smoke deliberately makes promotion independent of
  policy quality (see the guaranteed-met threshold below) so it is deterministic and non-flaky.
- **Non-goal:** any change to the shipped curriculum scene, `chase_curriculum.json`, or the
  existing promotion behavior.

## Design

Five additive pieces. Nothing changes for existing training/inference paths.

### 1. `CurriculumController` cmdline stages override

`addons/godot_native_rl/training/curriculum_controller.gd` gains a launch-time override for its
stages source, so the **existing** curriculum scene can be pointed at a different stages JSON
without editing the `.tscn`.

- New **pure static helper** `parse_stages_arg(args: PackedStringArray) -> String` — scans for a
  `curriculum_stages=<path>` token and returns the path, or `""` when absent/malformed. Pure and
  headlessly unit-testable, mirroring the existing `RunSpeed` cmdline-parse pattern
  (`OS.get_cmdline_args()` + `split("=")`).
- In `_ready()`, **before** the `stages_json_path`/`set_stages()` resolution, call
  `parse_stages_arg(OS.get_cmdline_args())`; if it returns a non-empty path, it overrides
  `stages_json_path`. Precedence: an explicit `set_stages()` (external/programmatic) still wins, as
  today; the cmdline override only supersedes the exported `stages_json_path`.

Interface: what it does — resolves which stages file to load; how you use it — pass
`curriculum_stages=res://…json` on the Godot command line; depends on — only `OS.get_cmdline_args()`.

### 2. Guaranteed-met smoke stages fixture

`test/integration/chase_curriculum_smoke.json` — a 2-stage curriculum whose promote block is met
by any episode:

```json
{
  "stages": [
    {"name": "smoke_a", "params": {"touch_radius": 120.0, "arena_size_x": 500.0, "arena_size_y": 300.0},
     "promote": {"metric": "mean_reward", "threshold": -1000000000.0, "window": 3, "min_episodes": 3}},
    {"name": "smoke_b", "params": {"touch_radius": 80.0, "arena_size_x": 500.0, "arena_size_y": 300.0}}
  ]
}
```

`threshold: -1e9` is below any achievable `mean_reward`, so promotion fires as soon as
`min_episodes` (3) episodes are recorded — within seconds, regardless of the (early, untrained)
policy. Lives under `test/` so the existing export skip-root excludes it from game exports.

### 3. `scripts/train_chase.sh` env passthroughs (non-destructive + arg forwarding)

Currently `train_chase.sh` hardcodes the trainer's output paths and forwards no extra Godot args.
Add env passthroughs (all optional, defaults unchanged), bringing it in line with
`train_cleanrl.sh`/`train_sf.sh`:

- `SAVE_MODEL_PATH` → `--save_model_path` (default `models/chase_policy.zip`)
- `ONNX_EXPORT_PATH` → `--onnx_export_path` (default `models/chase_policy.onnx`)
- `CHECKPOINT_DIR` → `--checkpoint_dir` (default `models/chase_checkpoints`)
- `GODOT_EXTRA_ARGS` → appended verbatim to the headless Godot launch (carries
  `curriculum_stages=…`).

The first three let the smoke write to a temp dir and **never clobber `models/`**; the fourth
delivers the cmdline override to Godot.

### 4. `test/run_tests.sh` guarded smoke block

Placed in the `godot_rl`-gated training-smoke cluster (same gate as the CleanRL/MA-POCA smokes;
auto-skips in a bare checkout). Pseudocode:

```bash
echo "== Curriculum trainer-driven promotion smoke (skipped if godot_rl absent in .venv-train) =="
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
    CURRIC_TMP="$(mktemp -d)"
    SCENE=res://examples/chase_the_target/chase_the_target_train_curriculum.tscn \
    TIMESTEPS="${CURRICULUM_SMOKE_TIMESTEPS:-3000}" \
    SAVE_MODEL_PATH="$CURRIC_TMP/m.zip" ONNX_EXPORT_PATH="$CURRIC_TMP/m.onnx" CHECKPOINT_DIR="$CURRIC_TMP/ckpt" \
    GODOT_EXTRA_ARGS="curriculum_stages=res://test/integration/chase_curriculum_smoke.json" \
        ./scripts/train_chase.sh > "$CURRIC_TMP/train.log" 2>&1 \
        || { echo "FAIL: curriculum smoke trainer errored" >&2; tail -40 "$CURRIC_TMP/train.log" >&2; rm -rf "$CURRIC_TMP"; exit 1; }
    grep -q "Curriculum: promoted to stage" "$CURRIC_TMP/train.log" \
        || { echo "FAIL: no curriculum promotion in trainer-driven run" >&2; tail -40 "$CURRIC_TMP/train.log" >&2; rm -rf "$CURRIC_TMP"; exit 1; }
    rm -rf "$CURRIC_TMP"
    echo "Curriculum trainer-driven promotion smoke OK."
else
    echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the curriculum promotion smoke)."
fi
```

`CURRIC_TMP` is added to the existing EXIT-trap cleanup list so a mid-run crash under `set -e`
doesn't leak the temp dir.

### 5. Unit test for the parser

`test/unit/test_curriculum_cmdline.gd` (or an added case in `test_curriculum_controller.gd`) covers
`parse_stages_arg`: token present → path; absent → `""`; malformed (`curriculum_stages` with no
`=value`) → `""`; unrelated args ignored.

## Data flow

```
run_tests.sh
  └─ train_chase.sh (SCENE=curriculum scene, GODOT_EXTRA_ARGS=curriculum_stages=…, temp outputs)
       ├─ train_chase.py  (SB3 PPO, ~3000 steps, temp .zip/.onnx)
       └─ godot --headless <curriculum scene> speedup=.. action_repeat=.. curriculum_stages=<smoke.json>
            └─ CurriculumController._ready()
                 └─ parse_stages_arg(cmdline) → smoke.json → set_stages(low/guaranteed threshold)
            └─ ChaseAgent episode end → controller.record_episode(reward, success)
                 └─ should_promote() (min_episodes=3, threshold=-1e9) → advance()
                      └─ print "Curriculum: promoted to stage 1 \"smoke_b\""   ← asserted by grep
```

## Testing

- **TDD:** write `parse_stages_arg` unit test first (fails), implement the helper + `_ready()` wiring
  to green.
- **End-to-end:** the new `run_tests.sh` block runs in this environment (has `.venv-train` +
  `godot_rl`), so full `./test/run_tests.sh` actually exercises it. Success marker unchanged:
  `All tests passed.`
- **Negative control (manual, during verification):** temporarily revert the `SCENE=` forwarding in
  `train_chase.sh` and confirm the smoke fails with "no curriculum promotion" — proving it would
  have caught the original #186 bug.

## Files touched

| File | Change |
|---|---|
| `addons/godot_native_rl/training/curriculum_controller.gd` | `parse_stages_arg` static helper + `_ready()` override |
| `test/integration/chase_curriculum_smoke.json` | **new** guaranteed-met stages fixture |
| `scripts/train_chase.sh` | `SAVE_MODEL_PATH`/`ONNX_EXPORT_PATH`/`CHECKPOINT_DIR`/`GODOT_EXTRA_ARGS` passthroughs |
| `test/run_tests.sh` | guarded promotion smoke block + trap cleanup entry |
| `test/unit/test_curriculum_cmdline.gd` | **new** parser unit test |
| `CLAUDE.md` | `train_chase.sh` line: output overrides + promotion smoke note |

Closes #198. (#198 is a GitHub-only item, not in the `docs/BACKLOG.md` map, so no checkbox to flip.)

## Risks / mitigations

- **Smoke runtime** — a ~3000-step SB3 chase run is ~1–2 min (matches the RLlib/CleanRL smokes).
  Tunable via `CURRICULUM_SMOKE_TIMESTEPS`.
- **Clobbering committed artifacts** — mitigated by the temp-dir output overrides (piece 3);
  the smoke touches nothing under `models/` or `examples/…/models/`.
- **Flakiness** — eliminated by the guaranteed-met threshold; promotion depends only on episode
  count, not reward magnitude.
