# Scripted expert for expert-demo recording (#46/#260 M4). Greedily steps toward the
# next-in-order tile (the same steering the sorter smoke checker uses, which reliably solves
# variable-count episodes), so demos can be generated headlessly with no human input and with no
# RL exploration gamble. Path-based extends (no bare class_name) so it resolves in headless/CLI
# runs — see CLAUDE.md.
extends "res://examples/sorter/sorter_agent.gd"

# Pure: map the offset-to-next-tile to a discrete move index.
# SorterAgent movement: 1=up, 2=down, 3=left, 4=right, 0=stay (ChaseObs.action_index_to_velocity).
static func expert_action_index(to: Vector2) -> int:
	if absf(to.x) > absf(to.y):
		return 4 if to.x > 0.0 else 3
	return 2 if to.y > 0.0 else 1

# godot_rl get_action() contract (RECORD_EXPERT_DEMOS mode): decide, store _action_index so the
# base _physics_process moves the body, and return the flat action array for recording.
func get_action() -> Array:
	if _game == null:
		return [0.0]
	var nxt: int = _game.next_target()
	if nxt <= 0:
		return [0.0]
	var to: Vector2 = _game.tile_position(nxt - 1) - _game.get_agent_pos()
	var idx := expert_action_index(to)
	_action_index = idx
	return [float(idx)]
