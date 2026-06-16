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

	h.finish(self)
