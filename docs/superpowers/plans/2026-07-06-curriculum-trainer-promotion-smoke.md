# Curriculum Trainer-Driven Promotion Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded CI smoke that runs a real SB3 trainer against the curriculum scene and asserts a promotion fires, catching trainer↔scene regressions no existing test covers (#198).

**Architecture:** Teach `CurriculumController` to accept a `curriculum_stages=<res://…>` cmdline override (pure, unit-tested helper) so the shipped scene can be pointed at a guaranteed-promote fixture; make `train_chase.sh` forward that arg and redirect outputs to a temp dir; add a `godot_rl`-gated `run_tests.sh` block that greps the trainer log for the promotion line.

**Tech Stack:** GDScript (Godot 4.5, TAB indentation), Bash, SB3 via `.venv-train`, the dependency-free `test/harness.gd` test harness.

## Global Constraints

- GDScript uses **TAB** indentation.
- Prefer **path-based `extends`/`preload`** over bare `class_name` (headless class-cache gotcha).
- Pure helpers + thin node wrappers; small focused files.
- Godot 4.6 `:=` can't infer from an untyped value — annotate `var x: Array = ...` explicitly.
- Tests: GDScript unit tests `extends "res://test/harness.gd"`, run via `godot --headless --path . --script res://test/unit/<file>.gd`.
- Do not modify the shipped `chase_curriculum.json` or `chase_the_target_train_curriculum.tscn`.
- Every push: full `./test/run_tests.sh` green (`All tests passed.`) before merge.

---

### Task 1: `parse_stages_arg` cmdline override on CurriculumController

**Files:**
- Modify: `addons/godot_native_rl/training/curriculum_controller.gd`
- Test: `test/unit/test_curriculum_cmdline.gd` (create)

**Interfaces:**
- Produces: `static func parse_stages_arg(args: PackedStringArray) -> String` on
  `curriculum_controller.gd` — returns the value of a `curriculum_stages=<path>` token, or `""`
  when absent/malformed. Consumed by `_ready()` (this task) to override `stages_json_path`.

- [ ] **Step 1: Write the failing unit test**

Create `test/unit/test_curriculum_cmdline.gd`:

```gdscript
extends "res://test/harness.gd"

const Controller = preload("res://addons/godot_native_rl/training/curriculum_controller.gd")

func run() -> void:
	# token present -> returns the path
	assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["speedup=8", "curriculum_stages=res://a/b.json", "action_repeat=8"])),
		"res://a/b.json", "extracts curriculum_stages value")
	# absent -> empty
	assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["speedup=8", "action_repeat=8"])),
		"", "absent -> empty string")
	# malformed (no =value) -> empty
	assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["curriculum_stages"])),
		"", "bare key without =value -> empty string")
	# empty args -> empty
	assert_eq(Controller.parse_stages_arg(PackedStringArray([])),
		"", "no args -> empty string")
	# a res:// path containing '=' in a query-like tail keeps everything after the first '='
	assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["curriculum_stages=res://x.json"])),
		"res://x.json", "single token parses")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_curriculum_cmdline.gd`
Expected: FAIL — `parse_stages_arg` not found on the controller script.

- [ ] **Step 3: Add the static helper**

In `addons/godot_native_rl/training/curriculum_controller.gd`, after the `@export var stages_json_path` line, add:

```gdscript
## Launch-time override for the stages source (#198): a `curriculum_stages=<res://…>` cmdline
## token supersedes the exported `stages_json_path`, so the shipped scene can be pointed at a
## different stages JSON without editing the .tscn. Returns "" when absent/malformed.
## Pure (args injected) so it is headlessly unit-testable — mirrors RunSpeed's parse convention.
static func parse_stages_arg(args: PackedStringArray) -> String:
	for argument in args:
		if argument.begins_with("curriculum_stages="):
			return argument.substr("curriculum_stages=".length())
	return ""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_curriculum_cmdline.gd`
Expected: PASS (all assertions).

- [ ] **Step 5: Wire the override into `_ready()`**

In `_ready()`, replace the stages-resolution block (currently lines ~26-29):

```gdscript
	if not _stages_set and stages_json_path != "":
		var loaded := _load_stages_json(stages_json_path)
		if not loaded.is_empty():
			set_stages(loaded)
```

with (cmdline override takes precedence over the exported path; `set_stages()` still wins over both):

```gdscript
	var cmdline_override := parse_stages_arg(OS.get_cmdline_args())
	var effective_stages_path := cmdline_override if cmdline_override != "" else stages_json_path
	if not _stages_set and effective_stages_path != "":
		var loaded := _load_stages_json(effective_stages_path)
		if not loaded.is_empty():
			set_stages(loaded)
```

- [ ] **Step 6: Re-run the unit test + the existing controller test to confirm no regression**

Run: `godot --headless --path . --script res://test/unit/test_curriculum_cmdline.gd`
Run: `godot --headless --path . --script res://test/unit/test_curriculum_controller.gd`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/training/curriculum_controller.gd test/unit/test_curriculum_cmdline.gd
git commit -m "feat(#198): curriculum_stages cmdline override on CurriculumController"
```

---

### Task 2: Guaranteed-met smoke stages fixture

**Files:**
- Create: `test/integration/chase_curriculum_smoke.json`

**Interfaces:**
- Produces: a 2-stage curriculum JSON whose stage-0 `promote` block is met by any episode
  (`threshold -1e9`, `window 3`, `min_episodes 3`). Consumed by the `run_tests.sh` block (Task 4)
  via the cmdline override from Task 1.

- [ ] **Step 1: Create the fixture**

Create `test/integration/chase_curriculum_smoke.json`:

```json
{
	"stages": [
		{"name": "smoke_a", "params": {"touch_radius": 120.0, "arena_size_x": 500.0, "arena_size_y": 300.0},
		 "promote": {"metric": "mean_reward", "threshold": -1000000000.0, "window": 3, "min_episodes": 3}},
		{"name": "smoke_b", "params": {"touch_radius": 80.0, "arena_size_x": 500.0, "arena_size_y": 300.0}}
	]
}
```

- [ ] **Step 2: Validate it parses + is accepted by the pure Curriculum**

Run:
```bash
godot --headless --path . --script res://test/unit/test_curriculum.gd
```
(Confirms the existing curriculum unit tests still pass — the fixture uses the same schema they cover.) Then sanity-check the JSON is well-formed:
```bash
.venv-train/bin/python -c "import json; d=json.load(open('test/integration/chase_curriculum_smoke.json')); assert d['stages'][0]['promote']['threshold']==-1e9; print('fixture OK', len(d['stages']),'stages')"
```
Expected: `fixture OK 2 stages`.

- [ ] **Step 3: Commit**

```bash
git add test/integration/chase_curriculum_smoke.json
git commit -m "test(#198): guaranteed-promote curriculum smoke fixture"
```

---

### Task 3: `train_chase.sh` env passthroughs (non-destructive + arg forwarding)

**Files:**
- Modify: `scripts/train_chase.sh`

**Interfaces:**
- Produces: four optional env vars on `train_chase.sh` (defaults unchanged):
  `SAVE_MODEL_PATH`, `ONNX_EXPORT_PATH`, `CHECKPOINT_DIR` (→ `train_chase.py` flags),
  `GODOT_EXTRA_ARGS` (→ appended to the headless Godot launch). Consumed by Task 4.

- [ ] **Step 1: Add the output-path + extra-args env vars**

In `scripts/train_chase.sh`, after the `SCENE="${SCENE:-…}"` line, add:

```bash
# Output overrides (#198) so a smoke run writes to a temp dir instead of models/ (mirrors
# train_cleanrl.sh / train_sf.sh). Defaults preserve the historical behavior.
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-models/chase_policy.zip}"
ONNX_EXPORT_PATH="${ONNX_EXPORT_PATH:-models/chase_policy.onnx}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-models/chase_checkpoints}"
# Extra user args appended to the Godot launch (e.g. curriculum_stages=res://…). Empty by default.
GODOT_EXTRA_ARGS="${GODOT_EXTRA_ARGS:-}"
```

- [ ] **Step 2: Forward the output paths to the trainer**

Change the trainer launch line from:

```bash
"$PY" scripts/train_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" $BEST_FLAG &
```

to:

```bash
"$PY" scripts/train_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
	--save_model_path "$SAVE_MODEL_PATH" --onnx_export_path "$ONNX_EXPORT_PATH" --checkpoint_dir "$CHECKPOINT_DIR" $BEST_FLAG &
```

- [ ] **Step 3: Forward the extra args to Godot**

Change the Godot launch line from:

```bash
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
```

to (`$GODOT_EXTRA_ARGS` intentionally unquoted so an empty value adds no arg and a value splits into tokens):

```bash
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" $GODOT_EXTRA_ARGS &
```

- [ ] **Step 4: Smoke-check the script still parses + defaults are intact**

Run:
```bash
bash -n scripts/train_chase.sh && echo "syntax OK"
grep -q 'SAVE_MODEL_PATH:-models/chase_policy.zip' scripts/train_chase.sh && echo "defaults preserved"
```
Expected: `syntax OK` and `defaults preserved`.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_chase.sh
git commit -m "feat(#198): SAVE_MODEL_PATH/ONNX_EXPORT_PATH/CHECKPOINT_DIR/GODOT_EXTRA_ARGS on train_chase.sh"
```

---

### Task 4: `run_tests.sh` guarded promotion smoke

**Files:**
- Modify: `test/run_tests.sh`

**Interfaces:**
- Consumes: the cmdline override (Task 1), the smoke fixture (Task 2), and the
  `train_chase.sh` env vars (Task 3).

- [ ] **Step 1: Add `CURRIC_TMP` to the EXIT-trap cleanup**

In `test/run_tests.sh`, change the trap line (currently line ~179):

```bash
trap 'rm -rf "${INT8_TMP:-}" "${SF_TMP:-}" "${RLLIB_TMP:-}" "${CLEANRL_TMP:-}" "${CLEANRL_ICM_TMP:-}" "${CLEANRL_GAIL_TMP:-}" "${MAPOCA_TMP:-}" 2>/dev/null || true' EXIT
```

to (add `${CURRIC_TMP:-}`):

```bash
trap 'rm -rf "${INT8_TMP:-}" "${SF_TMP:-}" "${RLLIB_TMP:-}" "${CLEANRL_TMP:-}" "${CLEANRL_ICM_TMP:-}" "${CLEANRL_GAIL_TMP:-}" "${MAPOCA_TMP:-}" "${CURRIC_TMP:-}" 2>/dev/null || true' EXIT
```

- [ ] **Step 2: Add the smoke block before the final `echo "All tests passed."`**

Immediately before the final `echo "All tests passed."` line, insert:

```bash
echo "== Curriculum trainer-driven promotion smoke (skipped if godot_rl absent in .venv-train) =="
# Runs a real SB3 trainer against the curriculum scene with a guaranteed-promote stages fixture
# (#198): asserts a "Curriculum: promoted to stage" line, catching trainer<->scene regressions
# (e.g. the #186 SCENE= bug) that the unit/wire/simulated-episode tests all missed. Outputs go to
# a temp dir so models/ is never touched.
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

- [ ] **Step 3: Verify the block runs and passes in isolation**

Run (this environment has `.venv-train` + `godot_rl`, so the block executes rather than skips):
```bash
bash -n test/run_tests.sh && echo "syntax OK"
CURRIC_TMP="$(mktemp -d)"; \
SCENE=res://examples/chase_the_target/chase_the_target_train_curriculum.tscn TIMESTEPS=3000 \
SAVE_MODEL_PATH="$CURRIC_TMP/m.zip" ONNX_EXPORT_PATH="$CURRIC_TMP/m.onnx" CHECKPOINT_DIR="$CURRIC_TMP/ckpt" \
GODOT_EXTRA_ARGS="curriculum_stages=res://test/integration/chase_curriculum_smoke.json" \
  ./scripts/train_chase.sh > "$CURRIC_TMP/train.log" 2>&1; \
grep -q "Curriculum: promoted to stage" "$CURRIC_TMP/train.log" && echo "PROMOTION SEEN" || { echo "NO PROMOTION"; tail -40 "$CURRIC_TMP/train.log"; }; \
rm -rf "$CURRIC_TMP"
```
Expected: `syntax OK` then `PROMOTION SEEN`.

- [ ] **Step 4: Negative control (proves the smoke would catch the #186 bug)**

Temporarily point the run at the *non-curriculum* scene (simulating the SCENE= bug) and confirm the assertion fails:
```bash
CURRIC_TMP="$(mktemp -d)"; \
SCENE=res://examples/chase_the_target/chase_the_target_train.tscn TIMESTEPS=3000 \
SAVE_MODEL_PATH="$CURRIC_TMP/m.zip" ONNX_EXPORT_PATH="$CURRIC_TMP/m.onnx" CHECKPOINT_DIR="$CURRIC_TMP/ckpt" \
GODOT_EXTRA_ARGS="curriculum_stages=res://test/integration/chase_curriculum_smoke.json" \
  ./scripts/train_chase.sh > "$CURRIC_TMP/train.log" 2>&1; \
grep -q "Curriculum: promoted to stage" "$CURRIC_TMP/train.log" && echo "UNEXPECTED PASS (bad)" || echo "correctly no promotion on plain scene"; \
rm -rf "$CURRIC_TMP"
```
Expected: `correctly no promotion on plain scene`. (No files committed in this step.)

- [ ] **Step 5: Commit**

```bash
git add test/run_tests.sh
git commit -m "test(#198): guarded curriculum trainer-driven promotion smoke"
```

---

### Task 5: Docs + full-suite verification + close

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none (documentation + verification).

- [ ] **Step 1: Update the `train_chase.sh` note in `CLAUDE.md`**

Find the `**Train (chase):**` bullet (the `TIMESTEPS=120000 ./scripts/train_chase.sh` line) and append a sentence noting the new overrides and the smoke:

```
`SAVE_MODEL_PATH`/`ONNX_EXPORT_PATH`/`CHECKPOINT_DIR` redirect the trainer's outputs (default
`models/…`) and `GODOT_EXTRA_ARGS` forwards extra Godot args (e.g. `curriculum_stages=…`, the
launch-time stages override on `CurriculumController`); the `run_tests.sh` curriculum
trainer-driven promotion smoke (#198) uses these to run a real trainer against the curriculum
scene with a guaranteed-promote fixture and assert a promotion fires.
```

- [ ] **Step 2: Run the FULL suite green**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.`, and the output includes `Curriculum trainer-driven promotion smoke OK.` (not the SKIP line).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(#198): note train_chase.sh output overrides + curriculum promotion smoke"
```

- [ ] **Step 4: Push + open draft PR (via finishing-a-development-branch)**

Push the branch and open a **draft** PR whose body summarizes the change and states `Closes #198`.

---

## Self-Review

**Spec coverage:**
- Cmdline override (spec §Design.1) → Task 1 ✅
- Guaranteed-met fixture (spec §Design.2) → Task 2 ✅
- `train_chase.sh` passthroughs (spec §Design.3) → Task 3 ✅
- `run_tests.sh` guarded block + trap (spec §Design.4) → Task 4 ✅
- Parser unit test (spec §Design.5) → Task 1 (folded in — the deliverable needing it) ✅
- Negative control (spec §Testing) → Task 4 Step 4 ✅
- Docs + Closes #198 (spec §Files) → Task 5 ✅

**Placeholder scan:** none — every code/command step shows exact content.

**Type consistency:** `parse_stages_arg(PackedStringArray) -> String` defined in Task 1 and consumed
by the same task's `_ready()` wiring; env var names (`SAVE_MODEL_PATH`, `ONNX_EXPORT_PATH`,
`CHECKPOINT_DIR`, `GODOT_EXTRA_ARGS`, `CURRIC_TMP`, `CURRICULUM_SMOKE_TIMESTEPS`) are identical
across Tasks 3–5. Promotion grep string `"Curriculum: promoted to stage"` matches
`curriculum_controller.gd`'s `print()` verbatim.
