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
