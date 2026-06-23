class_name HurdleTrack
extends Node3D
# Code-built hurdles for #60 M2: StaticBody3D boxes spanning the track on collision layer 2,
# so a closeness RaycastSensor3D with collision_mask = 2 reads pure hurdle proximity.
# Layout is a pure function (testable); the node rebuilds on apply_curriculum() — wired to the
# CurriculumController, which applies at episode boundaries only.

@export var hurdle_count := 0       ## stage 0 (flat) by default
@export var hurdle_height := 0.15   ## FIRST hurdle's height
@export var hurdle_height_max := 0.0 ## LAST hurdle's height when > hurdle_height: heights ramp up so
                                     ## later hurdles need a real jump (0 = uniform hurdle_height). (#252)
@export var hurdle_spacing := 8.0
@export var start_z := 8.0          ## first hurdle's Z (creature starts at 0, runs +Z)
@export var hurdle_width := 6.0     ## track-spanning X extent
@export var hurdle_depth := 0.3

var _next_index := 0  # first hurdle z not yet passed (episode-monotonic)

# Pure: z positions for `count` hurdles spaced `spacing` apart from `from_z`.
static func hurdle_layout(count: int, spacing: float, from_z: float) -> Array:
	var out: Array = []
	for i in range(maxi(count, 0)):
		out.append(from_z + i * spacing)
	return out

# Pure: height of hurdle `i` of `count`. Ramps linearly from `base` (first) to `max_h` (last) when
# max_h > base, so later hurdles are taller (a real jump); otherwise uniform `base`. (#252)
static func hurdle_height_at(i: int, count: int, base: float, max_h: float) -> float:
	if max_h <= base or count <= 1:
		return base
	return base + (max_h - base) * (float(i) / float(count - 1))

func _ready() -> void:
	rebuild()

func zs() -> Array:
	return hurdle_layout(hurdle_count, hurdle_spacing, start_z)

func rebuild() -> void:
	for c in get_children():
		remove_child(c)
		c.queue_free()
	var layout := zs()
	for i in range(layout.size()):
		var h: float = hurdle_height_at(i, layout.size(), hurdle_height, hurdle_height_max)
		add_child(_make_hurdle(float(layout[i]), h))
	_next_index = 0

func _make_hurdle(z: float, height: float) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.collision_layer = 2
	body.collision_mask = 0
	body.position = Vector3(0.0, height / 2.0, z)
	# The collision box drives the closeness RaycastSensor3D (the net's hurdle perception). It is kept
	# at the TRAINED height and left MESH-LESS (invisible) so the deployed policy behaves byte-identically
	# (#277 is cosmetic-only — no retrain). By design the hurdles never physically collide with the
	# creature (collision_mask = 0), so a full-height visible box looked like the creature running THROUGH
	# a wall. The visible mesh below is decoupled from this box.
	var col := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = Vector3(hurdle_width, height, hurdle_depth)
	col.shape = shape
	body.add_child(col)
	# Visible hurdle (#277): a low, bright athletic-hurdle gate (two standards + a crossbar) the trotting
	# gait visibly clears, with a crossbar height that clearly ASCENDS along the track. Decoupled from the
	# invisible sensor box so the demo reads as "clearing ascending hurdles" without changing what the net
	# senses or does.
	body.add_child(_make_hurdle_visual(height))
	return body

# Pure-ish visual builder: a hurdle gate sitting on the ground (it undoes the body's y offset), whose
# crossbar height maps the trained collision height onto a low, clearable, clearly-ascending range.
func _make_hurdle_visual(height: float) -> Node3D:
	var gate := Node3D.new()
	gate.position = Vector3(0.0, -height / 2.0, 0.0)  # build from ground up (cancel the body's y offset)
	# Map the trained height [hurdle_height, hurdle_height_max] -> a low visible range so the bar stays
	# below the torso (the gait clears it) yet the ascent is obvious. Uniform-height tracks -> all low.
	var span: float = maxf(0.001, hurdle_height_max - hurdle_height)
	var t: float = clampf((height - hurdle_height) / span, 0.0, 1.0) if hurdle_height_max > hurdle_height else 0.0
	var bar_h: float = lerpf(0.10, 0.24, t)
	var half_w := 0.9
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.95, 0.45, 0.1)  # bright orange — reads as a hurdle
	for sx in [-half_w, half_w]:
		var post := MeshInstance3D.new()
		var pm := BoxMesh.new()
		pm.size = Vector3(0.06, bar_h, 0.06)
		post.mesh = pm
		post.material_override = mat
		post.position = Vector3(sx, bar_h / 2.0, 0.0)
		gate.add_child(post)
	var bar := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(2.0 * half_w + 0.06, 0.06, maxf(hurdle_depth, 0.08))
	bar.mesh = bm
	bar.material_override = mat
	bar.position = Vector3(0.0, bar_h, 0.0)
	gate.add_child(bar)
	return gate

# How many hurdles `torso_z` has newly moved past since the last call. Monotonic within an
# episode; pays each hurdle once. Falling ends the episode via the agent's fall terminal, so a
# passed hurdle was passed upright.
func count_newly_passed(torso_z: float) -> int:
	var layout := zs()
	var n := 0
	while _next_index < layout.size() and torso_z > float(layout[_next_index]) + hurdle_depth / 2.0:
		_next_index += 1
		n += 1
	return n

func reset_progress() -> void:
	_next_index = 0

# CurriculumController target: stage params rebuild the track at the episode boundary.
func apply_curriculum(params: Dictionary) -> void:
	hurdle_count = int(params.get("hurdle_count", hurdle_count))
	hurdle_height = float(params.get("hurdle_height", hurdle_height))
	hurdle_height_max = float(params.get("hurdle_height_max", hurdle_height_max))
	hurdle_spacing = float(params.get("hurdle_spacing", hurdle_spacing))
	rebuild()
