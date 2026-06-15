# Attention M1 — Godot Variable-Length Entity Observation Block — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Godot side of #46 (sub-issue #257): a pure `EntityObsMath` helper plus
`EntitySensor2D/3D` nodes that encode up to N nearest entities into a fixed-width flat observation
`[N·F entity features (zero-padded)][N presence flags]`, so a later attention encoder can derive its
mask from the flags.

**Architecture:** Pure-helper + thin-node, matching the repo's sensor pattern. `EntityObsMath`
(stateless, `RefCounted`) does sort-by-distance / nearest-N cap / zero-pad / presence flags.
`EntitySensor2D/3D` extend the `ISensor` base **by path**, reuse `RelativePositionMath` for the
egocentric relative-position features, append optional custom per-entity scalars from a duck-typed
`get_entity_features()`, and call `EntityObsMath.build_obs`. Variable entity count rides in the flags,
never in the vector length — so the policy input width is stable.

**Tech Stack:** GDScript (Godot 4.5+), **TAB indentation**, path-based `extends`, dependency-free
headless test harness (`test/harness.gd`, tests are `extends SceneTree`, auto-discovered by
`test/run_tests.sh` via the `test/unit/test_*.gd` glob).

**Conventions to respect (from CLAUDE.md / project memory):**
- Set the Godot binary per machine: `export GODOT=/opt/homebrew/bin/godot` (any Godot 4.5+ build).
  The binary path is machine-specific — probe with `which godot` if that path is absent.
- **Never** assign an untyped array literal to a typed `Array[T]` property — it errors/hangs headless.
  Build a typed local first (`var objs: Array[Node2D] = []`) then assign.
- Godot 4.6 `:=` can't infer from an untyped value — annotate the local explicitly (`var x: Vector2 = ...`).
- A passing suite prints `All tests passed.`; a passing unit file prints `Results: N passed, 0 failed`.

---

## File Structure

| File | Responsibility | New/Modify |
| --- | --- | --- |
| `addons/godot_native_rl/sensors/entity_obs_math.gd` | Pure: `obs_size`, `build_obs` (sort/cap/pad/flags) | New |
| `addons/godot_native_rl/sensors/entity_sensor_2d.gd` | Thin 2D sensor node | New |
| `addons/godot_native_rl/sensors/entity_sensor_3d.gd` | Thin 3D sensor node | New |
| `test/unit/test_entity_obs_math.gd` | Pure-helper unit tests | New |
| `test/unit/stubs/entity_feature_stub_2d.gd` | Test entity exposing `get_entity_features()` (2D) | New |
| `test/unit/stubs/entity_feature_stub_3d.gd` | Same, 3D | New |
| `test/unit/test_entity_sensor_2d.gd` | 2D sensor node unit tests | New |
| `test/unit/test_entity_sensor_3d.gd` | 3D sensor node unit tests | New |
| `CLAUDE.md` | Add `EntitySensor2D/3D` to the sensors list | Modify |

---

## Task 1: `EntityObsMath` pure helper

**Files:**
- Create: `addons/godot_native_rl/sensors/entity_obs_math.gd`
- Test: `test/unit/test_entity_obs_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_entity_obs_math.gd`:

```gdscript
extends SceneTree
# Unit tests for the pure variable-length entity-observation helper (#46 M1).

const Harness = preload("res://test/harness.gd")
const M = preload("res://addons/godot_native_rl/sensors/entity_obs_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	# obs_size = N*F + N (entity block + presence flags).
	h.assert_eq(M.obs_size(3, 2), 9, "obs_size 3x2 = 9")
	h.assert_eq(M.obs_size(4, 0), 4, "obs_size flags-only = N")

	# Fewer than N: real row then zero-padding, flags mark presence.
	var one: Array = M.build_obs([{"dist": 1.0, "feat": [0.5, -0.5]}], 3, 2)
	h.assert_eq(one, [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "one entity padded + flags")

	# More than N: keep the nearest N, sorted ascending by distance.
	var capped: Array = M.build_obs([
		{"dist": 5.0, "feat": [5.0]},
		{"dist": 1.0, "feat": [1.0]},
		{"dist": 3.0, "feat": [3.0]}], 2, 1)
	h.assert_eq(capped, [1.0, 3.0, 1.0, 1.0], "nearest-2 sorted + flags")

	# Ties keep original input order (stable).
	var tie: Array = M.build_obs([
		{"dist": 2.0, "feat": [10.0]},
		{"dist": 2.0, "feat": [20.0]}], 2, 1)
	h.assert_eq(tie, [10.0, 20.0, 1.0, 1.0], "ties keep input order")

	# No entities: all zeros, all flags 0.
	var none: Array = M.build_obs([], 2, 2)
	h.assert_eq(none, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "no entities -> zeros + zero flags")

	# Defensive width fit: a wrong-width feat row is truncated/padded to F.
	var fit: Array = M.build_obs([{"dist": 0.0, "feat": [1.0, 2.0, 3.0]}], 1, 2)
	h.assert_eq(fit, [1.0, 2.0, 1.0], "row fit to feat width")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
export GODOT="${GODOT:-/opt/homebrew/bin/godot}"
"$GODOT" --headless --path . --script res://test/unit/test_entity_obs_math.gd
```
Expected: FAIL — parse error `Static function "obs_size()" not found` (the helper file doesn't exist yet).

- [ ] **Step 3: Write the minimal implementation**

Create `addons/godot_native_rl/sensors/entity_obs_math.gd`:

```gdscript
class_name EntityObsMath
extends RefCounted

# Pure helpers for the variable-length entity observation block (#46). No scene/physics deps, fully
# headless-unit-testable. An EntitySensor encodes up to N NEAREST entities, each F floats, into a
# FIXED-width flat vector: [N*F entity features (zero-padded)] followed by [N presence flags]
# (1.0 real / 0.0 pad). Variable entity count rides in the flags, NOT in the vector length, so the
# policy input width is stable and the attention encoder derives its mask from the flags.

# Total floats: N entity rows of F features + N presence flags.
static func obs_size(n_max: int, feat: int) -> int:
	return n_max * feat + n_max

# Build the flat block. `entities` is an Array of Dictionaries {"dist": float, "feat": Array}.
# Sorted ascending by dist (stable on ties via original index), capped to the nearest n_max,
# zero-padded to n_max rows, with an appended parallel presence-flag tail. Returns n_max*(feat+1)
# floats. A feat row of the wrong length is defensively pad/truncated so the width is always exact.
static func build_obs(entities: Array, n_max: int, feat: int) -> Array:
	var indexed: Array = []
	for i in range(entities.size()):
		indexed.append({"i": i, "e": entities[i]})
	indexed.sort_custom(func(a, b):
		var da: float = float(a["e"].get("dist", 0.0))
		var db: float = float(b["e"].get("dist", 0.0))
		if da == db:
			return int(a["i"]) < int(b["i"])
		return da < db)
	var rows: Array = []
	var flags: Array = []
	var kept := mini(indexed.size(), n_max)
	for k in range(n_max):
		if k < kept:
			var row: Array = indexed[k]["e"].get("feat", [])
			rows.append_array(_fit(row, feat))
			flags.append(1.0)
		else:
			rows.append_array(_zeros(feat))
			flags.append(0.0)
	rows.append_array(flags)
	return rows

static func _zeros(n: int) -> Array:
	var out: Array = []
	out.resize(n)
	out.fill(0.0)
	return out

# Resize a feature row to exactly `feat` floats (pad with 0.0 / truncate).
static func _fit(row: Array, feat: int) -> Array:
	var out: Array = []
	for i in range(feat):
		out.append(float(row[i]) if i < row.size() else 0.0)
	return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_entity_obs_math.gd
```
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/entity_obs_math.gd test/unit/test_entity_obs_math.gd
git commit -m "feat(sensors): EntityObsMath — variable-length entity obs block (#46 M1)"
```

---

## Task 2: `EntitySensor2D` node

**Files:**
- Create: `addons/godot_native_rl/sensors/entity_sensor_2d.gd`
- Create: `test/unit/stubs/entity_feature_stub_2d.gd`
- Test: `test/unit/test_entity_sensor_2d.gd`

- [ ] **Step 1: Write the failing test + stub**

Create `test/unit/stubs/entity_feature_stub_2d.gd`:

```gdscript
extends Node2D
# Test stub: a 2D entity exposing custom per-entity features for EntitySensor2D.
var features: Array = []

func get_entity_features() -> Array:
	return features
```

Create `test/unit/test_entity_sensor_2d.gd`:

```gdscript
extends SceneTree
# Unit tests for EntitySensor2D (#46 M1). Sensor and entities are detached nodes (local-transform
# fallback) so the test is fully headless.

const Harness = preload("res://test/harness.gd")
const Sensor = preload("res://addons/godot_native_rl/sensors/entity_sensor_2d.gd")
const Stub = preload("res://test/unit/stubs/entity_feature_stub_2d.gd")

func _make(pos: Vector2, feats: Array) -> Node2D:
	var s: Node2D = Stub.new()
	s.position = pos
	s.features = feats
	return s

func _initialize() -> void:
	var h = Harness.new()

	# F = relative offset (x,y) only; no extras. obs_size = N*(2) + N.
	var sensor = Sensor.new()
	sensor.max_entities = 3
	sensor.max_distance = 10.0
	sensor.use_separate_direction = false
	sensor.extra_feature_count = 0
	h.assert_eq(sensor.feature_width(), 2, "feature_width = 2 (x,y)")
	h.assert_eq(sensor.obs_size(), 9, "obs_size = 3*2 + 3")

	# One entity at (10,0): scaled offset = (1,0); presence flags [1,0,0].
	var objs: Array[Node2D] = []
	objs.append(_make(Vector2(10, 0), []))
	sensor.objects_to_observe = objs
	var obs: Array = sensor.get_observation()
	h.assert_eq(obs.size(), 9, "obs length matches obs_size")
	h.assert_eq(obs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "one entity offset + flags")

	# Nearest-N cap + ordering: near at (1,0), far at (9,0), N=1 -> keep nearest only.
	var s2 = Sensor.new()
	s2.max_entities = 1
	s2.max_distance = 100.0
	var objs2: Array[Node2D] = []
	objs2.append(_make(Vector2(9, 0), []))   # far
	objs2.append(_make(Vector2(1, 0), []))   # near
	s2.objects_to_observe = objs2
	var obs2: Array = s2.get_observation()
	# nearest is (1,0) -> scaled (0.01, 0); flag [1].
	h.assert_eq(obs2.size(), 3, "N=1 obs length = 2+1")
	h.assert_true(absf(obs2[0] - 0.01) < 1e-5 and absf(obs2[1]) < 1e-5 and obs2[2] == 1.0, "kept the nearest entity")

	# Extra features appended after the relative-position block.
	var s3 = Sensor.new()
	s3.max_entities = 2
	s3.max_distance = 10.0
	s3.extra_feature_count = 2
	h.assert_eq(s3.feature_width(), 4, "feature_width = 2 pos + 2 extra")
	var objs3: Array[Node2D] = []
	objs3.append(_make(Vector2(10, 0), [0.7, 0.3]))
	s3.objects_to_observe = objs3
	var obs3: Array = s3.get_observation()
	# row0 = [offx=1, offy=0, extra 0.7, extra 0.3]; row1 padded; flags [1,0].
	h.assert_eq(obs3, [1.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "extras appended + padded + flags")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_entity_sensor_2d.gd
```
Expected: FAIL — `Could not load ... entity_sensor_2d.gd` / parse error (the sensor doesn't exist yet).

- [ ] **Step 3: Write the minimal implementation**

Create `addons/godot_native_rl/sensors/entity_sensor_2d.gd`:

```gdscript
class_name EntitySensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# Variable-length entity observation (#46): encodes up to `max_entities` NEAREST entities into a
# fixed-width flat block [N*F entity features (zero-padded)] + [N presence flags], for the attention
# encoder. Entities are the union of `objects_to_observe` and any nodes in `group_name`. Each entity
# contributes egocentric relative-position features (RelativePositionMath, like RelativePositionSensor)
# optionally followed by `extra_feature_count` custom scalars from a duck-typed get_entity_features()
# on the entity node. Stable fixed width: variable count rides in the flags, not the vector length.
#
# Extend the ISensor base BY PATH (class-name cache is unreliable headless — see CLAUDE.md).

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")
const EntityObsMath = preload("res://addons/godot_native_rl/sensors/entity_obs_math.gd")

@export var max_entities: int = 8
@export var objects_to_observe: Array[Node2D] = []
@export var group_name: StringName = &""
@export_range(0.01, 20000.0) var max_distance: float = 1.0
@export var include_x: bool = true
@export var include_y: bool = true
@export var use_separate_direction: bool = false
## Number of extra scalar features each entity provides via get_entity_features() (0 = none).
@export var extra_feature_count: int = 0

# Floats per entity = relative-position features + extra scalars.
func feature_width() -> int:
	return RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, false) + extra_feature_count

func obs_size() -> int:
	return EntityObsMath.obs_size(max_entities, feature_width())

func get_observation() -> Array:
	var sensor_xform := global_transform if is_inside_tree() else transform
	var sensor_pos := sensor_xform.origin
	var sensor_rotation := sensor_xform.get_rotation()
	var feat := feature_width()
	var entities: Array = []
	for obj in _candidates():
		if not is_instance_valid(obj):
			continue
		var target_pos: Vector2 = obj.global_position if obj.is_inside_tree() else obj.position
		var offset := target_pos - sensor_pos
		var row: Array = RelativePositionMath.encode_2d(offset, sensor_rotation, max_distance, use_separate_direction, include_x, include_y)
		row.append_array(_extra_features(obj))
		entities.append({"dist": offset.length(), "feat": row})
	return EntityObsMath.build_obs(entities, max_entities, feat)

# Union of explicit targets and group members, de-duplicated (explicit first, then group).
func _candidates() -> Array:
	var out: Array = []
	for o in objects_to_observe:
		if o != null and not out.has(o):
			out.append(o)
	if String(group_name) != "" and is_inside_tree():
		for o in get_tree().get_nodes_in_group(group_name):
			if o != null and not out.has(o):
				out.append(o)
	return out

# Extra per-entity scalars from a duck-typed get_entity_features(); zero-filled when absent or the
# wrong length, so the block width stays exact.
func _extra_features(obj) -> Array:
	if extra_feature_count <= 0:
		return []
	var vals: Array = []
	if obj.has_method("get_entity_features"):
		var got = obj.get_entity_features()
		if got is Array:
			vals = got
	var out: Array = []
	for i in range(extra_feature_count):
		out.append(float(vals[i]) if i < vals.size() else 0.0)
	return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_entity_sensor_2d.gd
```
Expected: PASS — `Results: 8 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/entity_sensor_2d.gd test/unit/stubs/entity_feature_stub_2d.gd test/unit/test_entity_sensor_2d.gd
git commit -m "feat(sensors): EntitySensor2D — nearest-N padded entity obs (#46 M1)"
```

---

## Task 3: `EntitySensor3D` node

**Files:**
- Create: `addons/godot_native_rl/sensors/entity_sensor_3d.gd`
- Create: `test/unit/stubs/entity_feature_stub_3d.gd`
- Test: `test/unit/test_entity_sensor_3d.gd`

- [ ] **Step 1: Write the failing test + stub**

Create `test/unit/stubs/entity_feature_stub_3d.gd`:

```gdscript
extends Node3D
# Test stub: a 3D entity exposing custom per-entity features for EntitySensor3D.
var features: Array = []

func get_entity_features() -> Array:
	return features
```

Create `test/unit/test_entity_sensor_3d.gd`:

```gdscript
extends SceneTree
# Unit tests for EntitySensor3D (#46 M1). Detached nodes -> local-transform fallback, headless.

const Harness = preload("res://test/harness.gd")
const Sensor = preload("res://addons/godot_native_rl/sensors/entity_sensor_3d.gd")
const Stub = preload("res://test/unit/stubs/entity_feature_stub_3d.gd")

func _make(pos: Vector3, feats: Array) -> Node3D:
	var s: Node3D = Stub.new()
	s.position = pos
	s.features = feats
	return s

func _initialize() -> void:
	var h = Harness.new()

	# F = (x,y,z); no extras. obs_size = N*3 + N.
	var sensor = Sensor.new()
	sensor.max_entities = 2
	sensor.max_distance = 10.0
	sensor.include_z = true
	sensor.extra_feature_count = 0
	h.assert_eq(sensor.feature_width(), 3, "feature_width = 3 (x,y,z)")
	h.assert_eq(sensor.obs_size(), 8, "obs_size = 2*3 + 2")

	# One entity at (10,0,0): scaled (1,0,0); flags [1,0].
	var objs: Array[Node3D] = []
	objs.append(_make(Vector3(10, 0, 0), []))
	sensor.objects_to_observe = objs
	var obs: Array = sensor.get_observation()
	h.assert_eq(obs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "one entity offset + flags")

	# Extra features appended after the position block.
	var s2 = Sensor.new()
	s2.max_entities = 1
	s2.max_distance = 10.0
	s2.include_z = true
	s2.extra_feature_count = 1
	h.assert_eq(s2.feature_width(), 4, "feature_width = 3 pos + 1 extra")
	var objs2: Array[Node3D] = []
	objs2.append(_make(Vector3(0, 10, 0), [0.5]))
	s2.objects_to_observe = objs2
	var obs2: Array = s2.get_observation()
	# offset (0,10,0) -> scaled (0,1,0); extra 0.5; flag [1].
	h.assert_eq(obs2, [0.0, 1.0, 0.0, 0.5, 1.0], "3D offset + extra + flag")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_entity_sensor_3d.gd
```
Expected: FAIL — sensor file missing / parse error.

- [ ] **Step 3: Write the minimal implementation**

Create `addons/godot_native_rl/sensors/entity_sensor_3d.gd`:

```gdscript
class_name EntitySensor3D
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"

# 3D variable-length entity observation (#46) — the Node3D analog of EntitySensor2D. Encodes up to
# `max_entities` NEAREST entities into [N*F (zero-padded)] + [N presence flags], reusing
# RelativePositionMath.encode_3d for the egocentric features + optional duck-typed get_entity_features().
# Extend the ISensor base BY PATH (class-name cache is unreliable headless — see CLAUDE.md).

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")
const EntityObsMath = preload("res://addons/godot_native_rl/sensors/entity_obs_math.gd")

@export var max_entities: int = 8
@export var objects_to_observe: Array[Node3D] = []
@export var group_name: StringName = &""
@export_range(0.01, 20000.0) var max_distance: float = 1.0
@export var include_x: bool = true
@export var include_y: bool = true
@export var include_z: bool = true
@export var use_separate_direction: bool = false
## Number of extra scalar features each entity provides via get_entity_features() (0 = none).
@export var extra_feature_count: int = 0

func feature_width() -> int:
	return RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, include_z) + extra_feature_count

func obs_size() -> int:
	return EntityObsMath.obs_size(max_entities, feature_width())

func get_observation() -> Array:
	var sensor_xform := global_transform if is_inside_tree() else transform
	var sensor_pos := sensor_xform.origin
	var sensor_basis := sensor_xform.basis
	var feat := feature_width()
	var entities: Array = []
	for obj in _candidates():
		if not is_instance_valid(obj):
			continue
		var target_pos: Vector3 = obj.global_position if obj.is_inside_tree() else obj.position
		var offset := target_pos - sensor_pos
		var row: Array = RelativePositionMath.encode_3d(offset, sensor_basis, max_distance, use_separate_direction, include_x, include_y, include_z)
		row.append_array(_extra_features(obj))
		entities.append({"dist": offset.length(), "feat": row})
	return EntityObsMath.build_obs(entities, max_entities, feat)

func _candidates() -> Array:
	var out: Array = []
	for o in objects_to_observe:
		if o != null and not out.has(o):
			out.append(o)
	if String(group_name) != "" and is_inside_tree():
		for o in get_tree().get_nodes_in_group(group_name):
			if o != null and not out.has(o):
				out.append(o)
	return out

func _extra_features(obj) -> Array:
	if extra_feature_count <= 0:
		return []
	var vals: Array = []
	if obj.has_method("get_entity_features"):
		var got = obj.get_entity_features()
		if got is Array:
			vals = got
	var out: Array = []
	for i in range(extra_feature_count):
		out.append(float(vals[i]) if i < vals.size() else 0.0)
	return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_entity_sensor_3d.gd
```
Expected: PASS — `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/entity_sensor_3d.gd test/unit/stubs/entity_feature_stub_3d.gd test/unit/test_entity_sensor_3d.gd
git commit -m "feat(sensors): EntitySensor3D — nearest-N padded entity obs (#46 M1)"
```

---

## Task 4: Docs + full-suite gate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the sensors to CLAUDE.md**

In `CLAUDE.md`, find the reusable-library `sensors/` mention in the "Current state" paragraph and add
`EntitySensor2D/3D` to the sensor enumeration. Insert this sentence at the end of the sensors
description (keep it terse — CLAUDE.md is always-loaded):

```
EntitySensor2D/3D (#46 M1) emit a variable-length entity block — up to N nearest entities as
[N*F zero-padded features][N presence flags], pure EntityObsMath — for the attention encoder.
```

- [ ] **Step 2: Run the affected unit tests together**

Run:
```bash
for t in test_entity_obs_math test_entity_sensor_2d test_entity_sensor_3d; do
  echo "== $t =="; "$GODOT" --headless --path . --script res://test/unit/$t.gd 2>&1 | grep -E "Results|FAIL"
done
```
Expected: each prints `Results: N passed, 0 failed`, no `FAIL` lines.

- [ ] **Step 3: Run the full headless suite**

Run:
```bash
./test/run_tests.sh
```
Expected: ends with `All tests passed.` (the three new `test/unit/test_entity_*.gd` files are
auto-discovered by the `test/unit/test_*.gd` glob; the unit-test-count guard still passes).

> If `run_tests.sh` can't find a Godot binary, pass it: `GODOT="$GODOT" ./test/run_tests.sh`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note EntitySensor2D/3D in the sensors list (#46 M1)"
```

---

## Self-Review notes (already applied)

- **Spec coverage:** §1 obs contract (`[N·F][N flags]`, presence flags, nearest-N cap, fixed width)
  → Task 1 (`build_obs`/`obs_size`). §2 Godot obs side (`EntitySensor2D/3D`, reuse
  `RelativePositionMath`, duck-typed extra-feature hook) → Tasks 2–3. Out-of-scope items (Python
  encoder, Sorter env, ncnn) are correctly absent.
- **Type consistency:** `obs_size(n_max, feat)`, `build_obs(entities, n_max, feat)`,
  `feature_width()`, `get_observation()`, `obs_size()`, `get_entity_features()` names are identical
  across every task and match the existing `ISensor` / `RelativePositionMath` signatures.
- **Gotchas baked in:** path-based `extends`; typed-array locals before assignment; explicit local
  types for `:=`-unsafe values; per-machine `GODOT`; pass/fail signal strings.
