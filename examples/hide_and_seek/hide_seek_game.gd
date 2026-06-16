class_name HideSeekGame
extends Node2D

# Owns the 2D hide & seek world and the single shared episode. ALL world mutation happens here in
# one prioritized _physics_process (runs before the agents via process_physics_priority), so the two
# agents never race on shared state: each agent only SETS its velocity and READS the cached
# has_los/caught/terminal state. Positions reset lazily on the frame after a terminal so both agents
# observe a consistent world and the same terminal flag. Geometry is game-local (tile-offset-safe).

const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0
@export var catch_radius := 40.0
@export var max_steps := 300            ## episode timeout (frames)
@export var opp_max_dist := 1200.0      ## normalizer for the opponent-distance obs
@export var walls: Array[Rect2] = []    ## occluders; empty -> default_walls()
@export var min_separation := 200.0   ## min seeker<->hider spawn distance (avoids instant catch)
@export var seeker_body_path: NodePath
@export var hider_body_path: NodePath

var _seeker_body: Node2D
var _hider_body: Node2D
var _seeker_vel := Vector2.ZERO
var _hider_vel := Vector2.ZERO
var _step := 0
var _has_los := false
var _caught := false
var _terminal := false
var _pending_reset := false
var _approach_norm := 0.0   ## per-frame normalized seeker->hider closing rate (#274 dense reward)
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	# Run before the agents so cached state reflects this frame's integration.
	process_physics_priority = -10
	if walls.is_empty():
		walls = default_walls()
	_seeker_body = get_node_or_null(seeker_body_path) as Node2D
	_hider_body = get_node_or_null(hider_body_path) as Node2D
	reset_positions()

# --- Pure-ish helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func default_walls() -> Array[Rect2]:
	# Two SMALL staggered blocks (#274): one upper, one lower, each ~30% of arena height with open
	# lanes around them. The original two full-height (360px) blocks walled off the arena — the hider
	# trivially broke line-of-sight and the LOS-gated seeker, blind to an occluded target, wandered
	# into walls. Smaller staggered occluders keep "hide & seek" flavour (brief breaks behind a block)
	# while letting the seeker re-acquire and PURSUE, so the dense distance-closing reward produces a
	# legible chase instead of a blind stall.
	return [Rect2(330, 70, 60, 180), Rect2(610, 350, 60, 180)]

# --- Velocity setters (called by the agents; applied next physics frame) ---
func set_seeker_velocity(v: Vector2) -> void:
	_seeker_vel = v

func set_hider_velocity(v: Vector2) -> void:
	_hider_vel = v

# --- Cached-state getters (read by the agents) ---
func has_los() -> bool:
	return _has_los

func was_caught() -> bool:
	return _caught

func is_terminal() -> bool:
	return _terminal

func seeker_pos() -> Vector2:
	return _seeker_body.position if _seeker_body != null else Vector2.ZERO

func hider_pos() -> Vector2:
	return _hider_body.position if _hider_body != null else Vector2.ZERO

func distance() -> float:
	return seeker_pos().distance_to(hider_pos())

# Per-frame normalized closing rate for the dense reward (#274): >0 = seeker got closer this frame,
# <0 = hider opened the gap, 0 = no net change (e.g. both frozen). Range ~[-1, 1].
func approach_norm() -> float:
	return _approach_norm

# --- Episode lifecycle ---
func seed_rng(s: int) -> void:
	_rng.seed = s

func _random_free_position() -> Vector2:
	for _i in range(64):
		var p := Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))
		if not HideSeekMath.point_in_walls(p, walls):
			return p
	return Vector2(arena_size.x * 0.5, arena_size.y * 0.5)

func reset_positions() -> void:
	if _seeker_body != null:
		_seeker_body.position = _random_free_position()
	if _hider_body != null:
		_hider_body.position = _random_free_position()
		# Avoid spawning the hider on top of the seeker (instant catch). Resample a few times.
		if _seeker_body != null:
			for _i in range(32):
				if _seeker_body.position.distance_to(_hider_body.position) >= min_separation:
					break
				_hider_body.position = _random_free_position()
	# Drop carried-over velocities so a fresh episode doesn't nudge the new spawns with the
	# previous episode's last action before the agents set new velocities.
	_seeker_vel = Vector2.ZERO
	_hider_vel = Vector2.ZERO
	_step = 0
	_has_los = false
	_caught = false
	_terminal = false
	_pending_reset = false

# A bridge "reset" (or an agent) requests a world reset; applied at the next frame start.
func request_reset() -> void:
	_pending_reset = true

func _move_body(body: Node2D, vel: Vector2, delta: float) -> void:
	if body == null:
		return
	var target := clamp_to_bounds(body.position + vel * delta)
	# Walls block movement (not just sight); reject a step that would enter a wall.
	if not HideSeekMath.point_in_walls(target, walls):
		body.position = target

func _physics_process(delta: float) -> void:
	if _pending_reset:
		reset_positions()
	# Distance BEFORE this frame's integration (post-reset, so a respawn teleport never spikes the
	# delta). The dense reward shaping (#274) reads the normalized change after the bodies move.
	var dist_before := distance()
	_move_body(_seeker_body, _seeker_vel, delta)
	_move_body(_hider_body, _hider_vel, delta)
	# Normalize the closing rate by the max both agents can close in one frame (2 * move_speed * dt),
	# so `approach` lands in ~[-1, 1]: +1 = seeker closed at full combined speed, -1 = opened fully.
	var max_close := 2.0 * move_speed * delta
	_approach_norm = clampf((dist_before - distance()) / max_close, -1.0, 1.0) if max_close > 0.0 else 0.0
	_step += 1
	_has_los = not HideSeekMath.segment_blocked(seeker_pos(), hider_pos(), walls)
	_caught = _has_los and distance() < catch_radius
	_terminal = _caught or _step >= max_steps
	if _terminal:
		_pending_reset = true

# Lightweight visualizer shared by the random and trained standalone scenes.
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.10, 0.14), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.35, 0.35, 0.45), false, 2.0)
	for wall in walls:
		draw_rect(wall, Color(0.32, 0.34, 0.42), true)
	var seeker := seeker_pos()
	var hider := hider_pos()
	draw_line(seeker, hider,
		Color(0.95, 0.30, 0.25, 0.75) if _has_los else Color(0.35, 0.35, 0.42, 0.45),
		2.0)
	draw_circle(seeker, 18.0, Color(0.95, 0.30, 0.25))
	draw_circle(hider, 18.0, Color(0.25, 0.70, 1.0))
