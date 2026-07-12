extends "res://examples/chase_the_target/chase_game.gd"

# chase_rays (#364): chase + four SOLID AABB obstacles the agent cannot cross. Blocking is
# CODE-side (pure minimal-penetration resolution, mirrored exactly by the NumPy twin in
# scripts/chase_rays_twin_env.py) — like chase's kinematics, NOT engine physics, so the twin
# stays exact. The scene adds matching StaticBody2D boxes purely for the RaycastSensor2D to
# observe. Spawn/relocation positions are rejection-sampled outside the boxes (an in-box
# target could be uncatchable).

# (min_x, min_y) + 100x100 — keep in sync with chase_rays_twin_env.OBSTACLES and the golden
# fixture (test/fixtures/chase_rays_golden_obs.json); the unit tests pin this on both sides.
const OBSTACLES: Array[Rect2] = [
	Rect2(250, 150, 100, 100),
	Rect2(650, 150, 100, 100),
	Rect2(250, 350, 100, 100),
	Rect2(650, 350, 100, 100),
]


## Push `pos` out of `rect` along the minimal-penetration axis (no-op outside). Identical rule
## to the twin's resolve_aabb — tie-break order: left, right, up (min_y), down (max_y).
static func resolve_aabb(pos: Vector2, rect: Rect2) -> Vector2:
	var min_x := rect.position.x
	var min_y := rect.position.y
	var max_x := rect.end.x
	var max_y := rect.end.y
	if not (pos.x > min_x and pos.x < max_x and pos.y > min_y and pos.y < max_y):
		return pos
	var push_left := pos.x - min_x
	var push_right := max_x - pos.x
	var push_up := pos.y - min_y
	var push_down := max_y - pos.y
	var m: float = min(push_left, push_right, push_up, push_down)
	if m == push_left:
		return Vector2(min_x, pos.y)
	if m == push_right:
		return Vector2(max_x, pos.y)
	if m == push_up:
		return Vector2(pos.x, min_y)
	return Vector2(pos.x, max_y)


static func resolve_obstacles(pos: Vector2) -> Vector2:
	var p := pos
	for rect in OBSTACLES:
		p = resolve_aabb(p, rect)
	return p


static func inside_any_obstacle(pos: Vector2) -> bool:
	for rect in OBSTACLES:
		if pos.x > rect.position.x and pos.x < rect.end.x \
				and pos.y > rect.position.y and pos.y < rect.end.y:
			return true
	return false


func move_agent(velocity: Vector2, delta: float) -> void:
	if _agent_body != null:
		_agent_body.position = resolve_obstacles(
			clamp_to_bounds(_agent_body.position + velocity * delta))


func random_position() -> Vector2:
	while true:
		var p := super.random_position()
		if not inside_any_obstacle(p):
			return p
	return Vector2.ZERO  # unreachable (keeps the parser happy)


func _draw() -> void:
	super._draw()
	for rect in OBSTACLES:
		draw_rect(rect, Color(0.55, 0.42, 0.25), true)
		draw_rect(rect, Color(0.8, 0.62, 0.35), false, 2.0)
