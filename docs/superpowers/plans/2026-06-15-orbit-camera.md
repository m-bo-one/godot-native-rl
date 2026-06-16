# Orbit Camera Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable user-controlled orbit camera to the 3D demos so a viewer can rotate/zoom to inspect the trained behavior from any angle.

**Architecture:** One reusable `OrbitCamera` (`extends Camera3D`) in the addon, with two modes — *follow* (fixed behind-and-above, the current behavior) and *orbit* (drag to rotate, scroll to zoom). The pivot tracks the creature each frame via a duck-typed `get_camera_pivot()`. It supersedes `quadruped_camera.gd` on the quadruped/hexapod/rover demos; fly_by keeps its heading-cam and adds the orbit camera as a toggle-able second camera (via a `fallback_camera_path`). Cosmetic + inert headless.

**Tech Stack:** Godot 4.5 GDScript (TAB indentation), the repo's headless `test/harness.gd` SceneTree test pattern. Spec: `docs/superpowers/specs/2026-06-15-orbit-camera-design.md`.

---

## File Structure

- **Create** `addons/godot_native_rl/camera/orbit_camera.gd` — the reusable `OrbitCamera` node (pure spherical helpers + input-apply methods + node lifecycle).
- **Create** `test/unit/test_orbit_camera.gd` — headless unit test for the pure helpers + input-apply state.
- **Create** `test/unit/test_orbit_camera_in_scenes.gd` — scene-structure regression (each target demo carries the OrbitCamera).
- **Modify** `examples/quadruped_walk/quadruped_game.gd` — add `get_camera_pivot()`.
- **Modify** `examples/rover_3d/rover_game.gd` — add `get_camera_pivot()`.
- **Modify** `examples/fly_by/fly_by_game.gd` — add `get_camera_pivot()`.
- **Modify** the 4 quadruped scenes (`quadruped_walk_track.tscn`, `quadruped_hurdles_track.tscn`, `quadruped_race.tscn`, `hexapod_walk_track.tscn`) — swap the Camera3D script to `orbit_camera.gd`.
- **Modify** `examples/rover_3d/rover_3d.tscn` — set the Camera3D script to `orbit_camera.gd`.
- **Modify** `examples/fly_by/fly_by.tscn` — add a second Camera3D with `orbit_camera.gd` (non-current), wired to fall back to the heading cam.

Note: `quadruped_camera.gd` and `fly_by_camera.gd` are left in place (fly_by still uses its heading cam; quadruped_camera.gd becomes unused but is harmless to keep — do NOT delete in this plan).

---

### Task 1: OrbitCamera pure spherical helpers

**Files:**
- Create: `addons/godot_native_rl/camera/orbit_camera.gd`
- Test: `test/unit/test_orbit_camera.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_orbit_camera.gd`:

```gdscript
extends SceneTree
# Unit tests for OrbitCamera's pure helpers + input-apply state (#265). Node lifecycle
# (_process/_unhandled_input) needs the tree/input and is covered by the scene-structure test.

const Harness = preload("res://test/harness.gd")
const OrbitCamera = preload("res://addons/godot_native_rl/camera/orbit_camera.gd")

func _approx(h, got: float, want: float, msg: String) -> void:
	h.assert_true(absf(got - want) < 1e-4, "%s (got %f want %f)" % [msg, got, want])

func _initialize() -> void:
	var h := Harness.new()

	# orbit_position: az=0, el=0 -> straight +Z at `distance` from pivot.
	var p := OrbitCamera.orbit_position(Vector3(1, 2, 3), 0.0, 0.0, 10.0)
	h.assert_true(p.is_equal_approx(Vector3(1, 2, 13)), "orbit_position az0 el0 -> +Z")

	# offset_to_spherical is the inverse of orbit_position (round-trip an arbitrary offset).
	var off := Vector3(4.0, 3.5, -9.0)
	var s := OrbitCamera.offset_to_spherical(off)
	var back := OrbitCamera.orbit_position(Vector3.ZERO, s["azimuth"], s["elevation"], s["distance"])
	h.assert_true(back.is_equal_approx(off), "offset_to_spherical round-trips to the offset")
	_approx(h, s["distance"], off.length(), "spherical distance = offset length")

	# Degenerate offset -> zeroes, no NaN.
	var z := OrbitCamera.offset_to_spherical(Vector3.ZERO)
	_approx(h, z["distance"], 0.0, "zero offset -> zero distance")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: FAIL — `orbit_camera.gd` does not exist yet (load error / parse error).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/camera/orbit_camera.gd` with just the helpers for now:

```gdscript
extends Camera3D
# Reusable orbit camera (#265). FOLLOW mode holds a fixed behind-and-above view (the old
# quadruped_camera behavior); ORBIT mode lets the user drag to rotate + scroll to zoom. The pivot
# tracks the creature each frame in both modes. Cosmetic + inert headless (no input -> stays in
# follow). Reads a duck-typed get_camera_pivot() -> Vector3 from the game node.

# --- Pure helpers (unit-testable; no tree/input needed) ---

# World position `distance` from `pivot` at spherical angles. azimuth=elevation=0 -> straight +Z.
static func orbit_position(pivot: Vector3, azimuth: float, elevation: float, distance: float) -> Vector3:
	return pivot + distance * Vector3(
		cos(elevation) * sin(azimuth),
		sin(elevation),
		cos(elevation) * cos(azimuth))

# Inverse of orbit_position: the spherical angles + distance for a camera-relative offset.
static func offset_to_spherical(offset: Vector3) -> Dictionary:
	var d := offset.length()
	if d < 0.0001:
		return {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0}
	return {
		"azimuth": atan2(offset.x, offset.z),
		"elevation": asin(clampf(offset.y / d, -1.0, 1.0)),
		"distance": d}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/camera/orbit_camera.gd test/unit/test_orbit_camera.gd
git commit -m "feat(camera): OrbitCamera pure spherical helpers (#265)"
```

---

### Task 2: OrbitCamera state + input-apply methods

**Files:**
- Modify: `addons/godot_native_rl/camera/orbit_camera.gd`
- Test: `test/unit/test_orbit_camera.gd`

- [ ] **Step 1: Write the failing test**

Append inside `_initialize()` in `test/unit/test_orbit_camera.gd`, before `h.finish(self)`:

```gdscript
	# --- input-apply state (mutates az/el/dist with clamps; no tree needed) ---
	var cam = OrbitCamera.new()
	cam.default_offset = Vector3(4.0, 3.5, -9.0)
	cam.min_distance = 3.0
	cam.max_distance = 40.0
	cam.init_from_offset()  # derive az/el/dist from default_offset (what _ready does)
	var d0: float = cam.get_distance()
	_approx(h, d0, Vector3(4.0, 3.5, -9.0).length(), "init distance = offset length")

	# zoom in clamps at min_distance; zoom out clamps at max_distance.
	cam.zoom_step = 100.0
	cam.apply_zoom(-1)
	_approx(h, cam.get_distance(), 3.0, "zoom in clamps to min_distance")
	cam.apply_zoom(1)
	_approx(h, cam.get_distance(), 40.0, "zoom out clamps to max_distance")

	# elevation clamps to ~±80deg (1.4 rad).
	cam.apply_orbit_drag(Vector2(0.0, 100000.0))
	h.assert_true(cam.get_elevation() <= 1.4001, "elevation clamps high")
	cam.apply_orbit_drag(Vector2(0.0, -200000.0))
	h.assert_true(cam.get_elevation() >= -1.4001, "elevation clamps low")

	# set_orbit(false) resets the view to the default offset (a 'reset camera').
	cam.apply_orbit_drag(Vector2(500.0, 0.0))
	cam.set_orbit(false)
	_approx(h, cam.get_distance(), d0, "set_orbit(false) resets distance to default")
	cam.free()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: FAIL — `init_from_offset` / `apply_zoom` / `apply_orbit_drag` / `set_orbit` / `get_distance` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `addons/godot_native_rl/camera/orbit_camera.gd` (after the static helpers): the exports, state, and input-apply methods:

```gdscript
# --- Configuration ---
@export var default_offset := Vector3(4.0, 3.5, -9.0)  ## follow-mode view (= the old follow cam)
@export var smooth := 4.0                              ## position easing speed
@export var orbit_sensitivity := 0.008                 ## radians per pixel of drag
@export var zoom_step := 1.5                            ## metres per wheel notch
@export var min_distance := 3.0
@export var max_distance := 40.0
@export var toggle_key := KEY_C                         ## follow <-> orbit (or switch camera, with a fallback)
@export var game_path: NodePath                         ## node with get_camera_pivot(); defaults to the parent
@export var fallback_camera_path: NodePath             ## set on fly_by: the camera to restore when toggling off

const _ELEV_CLAMP := 1.4  # ~80 degrees

var _game
var _orbit := false
var _azimuth := 0.0
var _elevation := 0.0
var _distance := 1.0
var _dragging := false
var _snapped := false

# Derive the spherical state from default_offset so FOLLOW mode reproduces the old fixed offset
# exactly (orbit_position(pivot, az, el, dist) == pivot + default_offset by construction).
func init_from_offset() -> void:
	var s := offset_to_spherical(default_offset)
	_azimuth = s["azimuth"]
	_elevation = s["elevation"]
	_distance = s["distance"]

func get_distance() -> float:
	return _distance

func get_elevation() -> float:
	return _elevation

func apply_orbit_drag(delta: Vector2) -> void:
	_azimuth -= delta.x * orbit_sensitivity
	_elevation = clampf(_elevation + delta.y * orbit_sensitivity, -_ELEV_CLAMP, _ELEV_CLAMP)

func apply_zoom(notches: int) -> void:
	_distance = clampf(_distance + float(notches) * zoom_step, min_distance, max_distance)

# Switch mode. Returning to follow resets the view to default_offset (a 'reset camera').
func set_orbit(on: bool) -> void:
	_orbit = on
	if not on:
		init_from_offset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/camera/orbit_camera.gd test/unit/test_orbit_camera.gd
git commit -m "feat(camera): OrbitCamera orbit state + clamped input-apply (#265)"
```

---

### Task 3: OrbitCamera node lifecycle (resolve pivot, input, position each frame)

**Files:**
- Modify: `addons/godot_native_rl/camera/orbit_camera.gd`

(No new unit test — `_process`/`_unhandled_input` need the tree/input; correctness of the math is covered by Task 1–2 and the node loading by Task 8. This step wires the tested pieces together.)

- [ ] **Step 1: Add the lifecycle methods**

Append to `addons/godot_native_rl/camera/orbit_camera.gd`:

```gdscript
func _ready() -> void:
	# Resolve the game node only — do NOT read the pivot here: a child camera's _ready() runs before
	# the game-root builds its rig, so the pivot may not exist yet. The first _process snaps in.
	_game = get_node_or_null(game_path) if not game_path.is_empty() else get_parent()
	init_from_offset()

func _pivot() -> Vector3:
	if _game != null and _game.has_method("get_camera_pivot"):
		return _game.get_camera_pivot()
	return Vector3.ZERO

func _toggle() -> void:
	if fallback_camera_path.is_empty():
		# Single-camera demos (quadruped/rover): flip follow <-> orbit in place.
		set_orbit(not _orbit)
	else:
		# fly_by: switch the active camera between this orbit cam and the heading cam.
		var fb := get_node_or_null(fallback_camera_path) as Camera3D
		if current:
			if fb != null:
				fb.current = true  # back to the heading cam (deactivates this one)
		else:
			current = true        # activate the orbit cam
			set_orbit(true)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == toggle_key:
		_toggle()
		return
	# Only respond to mouse when we're the active camera.
	if not current:
		return
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = event.pressed
			if event.pressed:
				_orbit = true  # auto-enter orbit on drag
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			_orbit = true
			apply_zoom(-1)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
			_orbit = true
			apply_zoom(1)
	elif event is InputEventMouseMotion and _dragging and _orbit:
		apply_orbit_drag(event.relative)

func _process(delta: float) -> void:
	if _game == null or not _game.has_method("get_camera_pivot"):
		return
	var pivot := _pivot()
	var goal := orbit_position(pivot, _azimuth, _elevation, _distance)
	if _snapped:
		global_position = global_position.lerp(goal, clampf(smooth * delta, 0.0, 1.0))
	else:
		global_position = goal
		_snapped = true
	look_at(pivot, Vector3.UP)
```

- [ ] **Step 2: Verify it parses**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: PASS (still green — the new methods don't break the unit test; this confirms the file parses).

- [ ] **Step 3: Commit**

```bash
git add addons/godot_native_rl/camera/orbit_camera.gd
git commit -m "feat(camera): OrbitCamera node lifecycle — pivot, input, per-frame positioning (#265)"
```

---

### Task 4: `get_camera_pivot()` on the three games

**Files:**
- Modify: `examples/quadruped_walk/quadruped_game.gd`
- Modify: `examples/rover_3d/rover_game.gd`
- Modify: `examples/fly_by/fly_by_game.gd`

- [ ] **Step 1: Add to QuadrupedGame**

In `examples/quadruped_walk/quadruped_game.gd`, add after `torso_pos()`:

```gdscript
# Orbit-camera pivot (#265): the point the demo camera looks at / orbits around.
func get_camera_pivot() -> Vector3:
	return torso_pos()
```

- [ ] **Step 2: Add to RoverGame**

In `examples/rover_3d/rover_game.gd`, add after `get_agent_pos()`:

```gdscript
# Orbit-camera pivot (#265): the rover position the demo camera looks at / orbits around.
func get_camera_pivot() -> Vector3:
	return get_agent_pos()
```

- [ ] **Step 3: Add to FlyByGame**

In `examples/fly_by/fly_by_game.gd`, add after `get_plane_xform()`:

```gdscript
# Orbit-camera pivot (#265): the plane position the demo camera orbits around.
func get_camera_pivot() -> Vector3:
	return get_plane_xform().origin
```

- [ ] **Step 4: Verify they parse**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera.gd`
Expected: PASS (sanity — confirms nothing broke; the games are loaded by later tasks).

- [ ] **Step 5: Commit**

```bash
git add examples/quadruped_walk/quadruped_game.gd examples/rover_3d/rover_game.gd examples/fly_by/fly_by_game.gd
git commit -m "feat(camera): get_camera_pivot() on quadruped/rover/fly_by games (#265)"
```

---

### Task 5: Swap the quadruped/hexapod track scenes to OrbitCamera

**Files:**
- Modify: `examples/quadruped_walk/quadruped_walk_track.tscn`
- Modify: `examples/quadruped_walk/quadruped_hurdles_track.tscn`
- Modify: `examples/quadruped_walk/quadruped_race.tscn`
- Modify: `examples/quadruped_walk/hexapod_walk_track.tscn`

For EACH of the four scenes, change the `ext_resource` that points at `quadruped_camera.gd` to point at the OrbitCamera, keeping the same `id`. The Camera3D node already references that id, so the node now uses OrbitCamera and (with no exports overridden) defaults to `default_offset = Vector3(4, 3.5, -9)` — the same framing the old follow cam used.

- [ ] **Step 1: Walk track**

In `examples/quadruped_walk/quadruped_walk_track.tscn`, replace the line:

```
[ext_resource type="Script" path="res://examples/quadruped_walk/quadruped_camera.gd" id="6"]
```

with:

```
[ext_resource type="Script" path="res://addons/godot_native_rl/camera/orbit_camera.gd" id="6"]
```

- [ ] **Step 2: Hurdles, race, hexapod tracks**

Do the identical swap in the other three scenes. The exact `id` may differ per scene — find the `ext_resource ... quadruped_camera.gd id="N"` line and replace its `path` with `res://addons/godot_native_rl/camera/orbit_camera.gd`, leaving `id="N"` unchanged:

```bash
for f in quadruped_hurdles_track quadruped_race hexapod_walk_track; do
  grep -n "quadruped_camera.gd" "examples/quadruped_walk/$f.tscn"
done
```

Edit each found line, replacing only the `path=...quadruped_camera.gd` with `path="res://addons/godot_native_rl/camera/orbit_camera.gd"`.

- [ ] **Step 3: Verify all four load + instantiate**

Run:

```bash
/opt/homebrew/bin/godot-mono --headless --path . --script res://test/_camload.gd
```

(First create a throwaway `test/_camload.gd`:)

```gdscript
extends SceneTree
func _initialize() -> void:
	for p in ["quadruped_walk_track", "quadruped_hurdles_track", "quadruped_race", "hexapod_walk_track"]:
		var sc := load("res://examples/quadruped_walk/%s.tscn" % p) as PackedScene
		var ok := sc != null and sc.instantiate() != null
		print("%s: %s" % [p, "OK" if ok else "FAIL"])
	quit(0)
```

Expected: all four print `OK`. Then `rm -f test/_camload.gd test/_camload.gd.uid`.

- [ ] **Step 4: Commit**

```bash
git add examples/quadruped_walk/quadruped_walk_track.tscn examples/quadruped_walk/quadruped_hurdles_track.tscn examples/quadruped_walk/quadruped_race.tscn examples/quadruped_walk/hexapod_walk_track.tscn
git commit -m "feat(camera): use OrbitCamera on the quadruped/hexapod track demos (#265)"
```

---

### Task 6: Swap the rover scene to OrbitCamera

**Files:**
- Modify: `examples/rover_3d/rover_3d.tscn`

The rover Camera3D may currently be a plain `Camera3D` (no script) or have a follow script. Attach the OrbitCamera script to it, with a `default_offset` suited to the rover (it's smaller/closer than the quadruped).

- [ ] **Step 1: Inspect the rover camera node**

Run:

```bash
grep -nE 'ext_resource|node name="Camera3D"|script|transform' examples/rover_3d/rover_3d.tscn | head
```

- [ ] **Step 2: Add the ext_resource + script + offset**

Add an `ext_resource` for the OrbitCamera at the top of `examples/rover_3d/rover_3d.tscn` (use the next free `id` integer — call it `idN`):

```
[ext_resource type="Script" path="res://addons/godot_native_rl/camera/orbit_camera.gd" id="idN"]
```

On the `[node name="Camera3D" type="Camera3D" parent="."]` block, add (and remove any existing camera `script =` line):

```
script = ExtResource("idN")
default_offset = Vector3(3, 2.5, -6)
min_distance = 2.0
max_distance = 25.0
```

Increment `load_steps` in the scene's `[gd_scene load_steps=K format=3]` header by 1 (for the new ext_resource).

- [ ] **Step 3: Verify it loads + the rover behavioral regression still passes**

Run:

```bash
/opt/homebrew/bin/godot-mono --headless --path . res://test/integration/rover_3d_smoke_scene.tscn 2>&1 | grep -iE "ROVER|PASS|FAIL"
```

Expected: `ROVER SMOKE PASSED` (the camera is cosmetic + inert headless, so the smoke is unaffected). If the smoke scene path differs, find it: `ls test/integration | grep -i rover`.

- [ ] **Step 4: Commit**

```bash
git add examples/rover_3d/rover_3d.tscn
git commit -m "feat(camera): use OrbitCamera on the rover_3d demo (#265)"
```

---

### Task 7: fly_by — add OrbitCamera as a toggle-able second camera

**Files:**
- Modify: `examples/fly_by/fly_by.tscn`

fly_by keeps `fly_by_camera.gd` (the heading-trail cam) as the default `current` camera and gains a second Camera3D running OrbitCamera, `current = false`, with `fallback_camera_path` pointing at the heading cam. Pressing **C** activates the orbit cam; pressing **C** again restores the heading cam.

- [ ] **Step 1: Inspect the fly_by camera setup**

Run:

```bash
grep -nE 'ext_resource|node name=.*Camera|fly_by_camera|current|type="Camera3D"' examples/fly_by/fly_by.tscn | head
```

Note the heading camera's node name (e.g. `Camera3D`) and confirm it is `current = true` (or is the only camera, hence active).

- [ ] **Step 2: Add the OrbitCamera node**

Add the ext_resource at the top of `examples/fly_by/fly_by.tscn` (next free id `idN`):

```
[ext_resource type="Script" path="res://addons/godot_native_rl/camera/orbit_camera.gd" id="idN"]
```

Add a second camera node as a sibling of the existing camera (replace `<HeadingCamName>` with the actual node name found in Step 1):

```
[node name="OrbitCamera" type="Camera3D" parent="."]
current = false
script = ExtResource("idN")
default_offset = Vector3(0, 6, -16)
min_distance = 5.0
max_distance = 60.0
fallback_camera_path = NodePath("../<HeadingCamName>")
```

Ensure the heading camera explicitly has `current = true` (add the line to its node block if absent, so exactly one camera starts active). Increment `load_steps` by 1.

- [ ] **Step 3: Verify the scene loads + fly_by smoke still passes**

Run:

```bash
/opt/homebrew/bin/godot-mono --headless --path . --script res://test/_flyload.gd
```

with throwaway `test/_flyload.gd`:

```gdscript
extends SceneTree
func _initialize() -> void:
	var sc := load("res://examples/fly_by/fly_by.tscn") as PackedScene
	var r := sc.instantiate()
	var n := 0
	for c in r.get_children():
		if c is Camera3D:
			n += 1
	print("fly_by cameras: %d (expect 2)" % n)
	quit(0)
```

Expected: `fly_by cameras: 2`. Then `rm -f test/_flyload.gd test/_flyload.gd.uid`.

- [ ] **Step 4: Commit**

```bash
git add examples/fly_by/fly_by.tscn
git commit -m "feat(camera): fly_by gains a toggle-able OrbitCamera alongside its heading cam (#265)"
```

---

### Task 8: Scene-structure regression test

**Files:**
- Create: `test/unit/test_orbit_camera_in_scenes.gd`
- Modify: `test/run_tests.sh` (add the new unit test to the run, if it's not auto-discovered)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_orbit_camera_in_scenes.gd`:

```gdscript
extends SceneTree
# Structure regression (#265): every 3D demo that should have the orbit camera carries exactly one
# node running orbit_camera.gd. Scenes are instantiated WITHOUT entering the tree (no _ready / no
# ncnn / no inference) — we only assert the node is wired in.

const Harness = preload("res://test/harness.gd")
const ORBIT := "res://addons/godot_native_rl/camera/orbit_camera.gd"

const SCENES: Array[String] = [
	"res://examples/quadruped_walk/quadruped_walk_track.tscn",
	"res://examples/quadruped_walk/quadruped_hurdles_track.tscn",
	"res://examples/quadruped_walk/quadruped_race.tscn",
	"res://examples/quadruped_walk/hexapod_walk_track.tscn",
	"res://examples/rover_3d/rover_3d.tscn",
	"res://examples/fly_by/fly_by.tscn",
]

func _count(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == ORBIT:
		n += 1
	for c in node.get_children():
		n += _count(c)
	return n

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var root := packed.instantiate()
		h.assert_eq(_count(root), 1, "%s has exactly one OrbitCamera" % path)
		root.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_orbit_camera_in_scenes.gd`
Expected: PASS — `Results: 12 passed, 0 failed` (6 loads + 6 counts). If any scene reports 0, its script swap (Task 5–7) was missed; fix that scene.

- [ ] **Step 3: Ensure both new tests run in the suite**

Check how `test/run_tests.sh` discovers unit tests:

```bash
grep -nE "test/unit|for .* in|\.gd" test/run_tests.sh | head
```

If it globs `test/unit/*.gd` (auto-discovery), nothing to do. If it lists tests explicitly, add `test/unit/test_orbit_camera.gd` and `test/unit/test_orbit_camera_in_scenes.gd` to that list.

- [ ] **Step 4: Run the full suite locally**

Run: `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh 2>&1 | tail -20`
Expected: the suite ends green (all unit tests pass, all integration smokes pass — the cameras are inert headless, so the quadruped/rover/fly_by behavioral + golden regressions are unaffected).

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_orbit_camera_in_scenes.gd test/run_tests.sh
git commit -m "test(camera): scene-structure regression for the OrbitCamera demos (#265)"
```

---

### Task 9: Docs + PR

**Files:**
- Modify: `README.md` (the demos / controls section, if present) and/or `CLAUDE.md` (the quadruped/rover/fly_by example lines)

- [ ] **Step 1: Add a one-line controls note**

In `README.md` near where the 3D demos are described (or `CLAUDE.md`'s example list), add a short note:

> The 3D demos (quadruped walk/hurdles/race, hexapod, rover_3d, fly_by) carry a drop-in `OrbitCamera` (`addons/godot_native_rl/camera/orbit_camera.gd`): press **C** to toggle a free orbit camera, right-drag to rotate, scroll to zoom. Cosmetic + inert headless.

- [ ] **Step 2: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(camera): note the OrbitCamera controls on the 3D demos (#265)"
```

- [ ] **Step 3: Push + open the PR**

```bash
git push -u origin feature/265-orbit-camera
```

Open a PR titled `feat(camera): reusable orbit camera for the 3D demos (#265)`, body summarizing the node + per-demo wiring + the C/RMB/scroll controls + "cosmetic, inert headless", `Closes #265`. (GraphQL may be rate-limited — fall back to `gh api --method POST repos/minigraphx/godot-native-rl/pulls`.)

---

## Self-Review

**Spec coverage:** node in `addons/` ✓ (Task 1–3); follow+orbit modes + C toggle + RMB/scroll + auto-enter ✓ (Task 2–3); pure `orbit_position` helper + test ✓ (Task 1, 8); duck-typed `get_camera_pivot()` ✓ (Task 4); quadruped walk/hurdles/race/hexapod + rover supersede ✓ (Task 5–6); fly_by heading-cam + orbit toggle via `fallback_camera_path` ✓ (Task 7); headless inertness ✓ (no input -> follow, asserted by the unaffected integration smokes in Task 8); scene-structure regression ✓ (Task 8); out-of-scope items (showcase, free-fly, 2D) correctly excluded.

**Placeholder scan:** every code step shows full code; commands have expected output; no TBD/TODO. The only per-scene "find the id/name" steps (Task 5 Step 2, 6–7) are necessary because the exact ext_resource ids differ per scene — the find command + exact replacement are given.

**Type consistency:** `orbit_position(pivot, azimuth, elevation, distance)`, `offset_to_spherical(offset) -> {azimuth, elevation, distance}`, `init_from_offset()`, `apply_orbit_drag(Vector2)`, `apply_zoom(int)`, `set_orbit(bool)`, `get_distance()`, `get_elevation()`, `_toggle()`, `get_camera_pivot() -> Vector3`, `fallback_camera_path` — names are consistent across Task 1–8 and the test references.
