# Path-based extends, no bare class_name (headless script-cache gotcha — see CLAUDE.md).
extends Node2D
# Cosmetic demo FX (#229): a short fading motion trail behind the RL agent + a brief "catch pop"
# ring when the agent reaches the target. Reusable across the 2D chase-family PLAY scenes — reads
# the agent position via game.get_agent_pos() and hooks the game's `target_caught` signal, both of
# which ChaseGame and BallChaseGame expose. Purely additive: it NEVER touches obs/actions/rewards/
# physics, and it is inert under --headless (no _draw/redraw happens without a render context), so
# it is invisible to the training scenes and the headless regression checks.

@export var game_path: NodePath = NodePath("..")  ## the game node (ChaseGame / BallChaseGame)
@export var trail_length := 20                     ## number of trailing points kept
@export var trail_width := 12.0
@export var trail_color := Color(0.30, 0.80, 1.0)  ## matches the cyan agent dot
@export var pop_color := Color(1.0, 0.9, 0.4)

const POP_DURATION := 0.45
const POP_START_RADIUS := 12.0
const POP_MAX_RADIUS := 64.0

var _game: Node = null
var _trail: Line2D = null
var _points: Array[Vector2] = []
var _pops: Array = []  # each: {"pos": Vector2, "age": float}


func _ready() -> void:
	_game = get_node_or_null(game_path)
	_trail = Line2D.new()
	_trail.name = "AgentTrail"
	_trail.width = trail_width
	_trail.joint_mode = Line2D.LINE_JOINT_ROUND
	_trail.begin_cap_mode = Line2D.LINE_CAP_ROUND
	_trail.end_cap_mode = Line2D.LINE_CAP_ROUND
	# Draw above the game canvas item (the game paints a SOLID arena rect in its own _draw, so a
	# negative z_index would hide the trail behind it). A child at z_index >= parent draws after it.
	_trail.z_index = 1
	# Fade alpha from the oldest point (offset 0, transparent) to the newest (offset 1, visible).
	var grad := Gradient.new()
	grad.set_color(0, Color(trail_color.r, trail_color.g, trail_color.b, 0.0))
	grad.set_color(1, Color(trail_color.r, trail_color.g, trail_color.b, 0.55))
	_trail.gradient = grad
	# Taper the width from a thin tail to a full-width head.
	var wc := Curve.new()
	wc.add_point(Vector2(0.0, 0.2))
	wc.add_point(Vector2(1.0, 1.0))
	_trail.width_curve = wc
	add_child(_trail)
	if _game != null and _game.has_signal("target_caught"):
		_game.target_caught.connect(_on_caught)


func _on_caught() -> void:
	if _game != null and _game.has_method("get_agent_pos"):
		_pops.append({"pos": _game.get_agent_pos(), "age": 0.0})


func _process(delta: float) -> void:
	if _game == null or not _game.has_method("get_agent_pos"):
		return
	_points.append(_game.get_agent_pos())
	while _points.size() > trail_length:
		_points.remove_at(0)
	_trail.points = PackedVector2Array(_points)
	var i := _pops.size() - 1
	while i >= 0:
		_pops[i]["age"] += delta
		if _pops[i]["age"] >= POP_DURATION:
			_pops.remove_at(i)
		i -= 1
	queue_redraw()


func _draw() -> void:
	for pop in _pops:
		var t: float = clampf(float(pop["age"]) / POP_DURATION, 0.0, 1.0)
		var r: float = lerpf(POP_START_RADIUS, POP_MAX_RADIUS, t)
		var a: float = (1.0 - t) * 0.85
		draw_arc(pop["pos"], r, 0.0, TAU, 40, Color(pop_color.r, pop_color.g, pop_color.b, a), 3.0, true)
