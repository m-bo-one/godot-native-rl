# Memory chase game (#378): plain ChaseGame with blink-aware rendering. The TARGET is drawn
# solid only while the policy can actually see it (agent.is_target_visible()); while hidden the
# VIEWER gets a faint outline at the true position — so it's obvious the agent keeps homing on a
# target it can no longer observe (the LSTM memory made visible). Rendering only; every game
# mechanic is inherited unchanged.
extends "res://examples/chase_the_target/chase_game.gd"

@export var memory_agent_path: NodePath

var _memory_agent: Node = null

func _ready() -> void:
	super._ready()
	_memory_agent = get_node_or_null(memory_agent_path)

func _target_visible_now() -> bool:
	if _memory_agent != null and _memory_agent.has_method("is_target_visible"):
		return _memory_agent.is_target_visible()
	return true

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.30, 0.32, 0.42), false, 2.0)
	if _target_visible_now():
		draw_circle(get_target_pos(), touch_radius, Color(0.92, 0.33, 0.33))
	else:
		# Hidden from the POLICY — the viewer sees a faint ghost ring so the memory is legible.
		draw_arc(get_target_pos(), touch_radius, 0.0, TAU, 32, Color(0.92, 0.33, 0.33, 0.25), 2.0)
	draw_circle(get_agent_pos(), 15.0, Color(0.30, 0.80, 1.0))
