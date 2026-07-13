class_name ChaseMemoryObs
# Pure obs/visibility helpers for the memory chase (#378): a partially-observable chase where
# the target is visible only for a few decisions per cycle, so a recurrent policy must REMEMBER
# where it was and integrate its own motion while blind. No node state.

const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")
const OBS_SIZE := 6

# Deterministic decision-count blink cycle: visible for the first `visible_steps` decisions of
# each (visible_steps + hidden_steps) window. step < 0 is clamped to 0 (episodes start visible).
static func visible_for_step(step: int, visible_steps: int, hidden_steps: int) -> bool:
	if visible_steps <= 0:
		return false
	if hidden_steps <= 0:
		return true
	var cycle := visible_steps + hidden_steps
	return posmod(maxi(step, 0), cycle) < visible_steps

# 6-dim obs: [agent_x, agent_y, dir_x, dir_y, dist, visible]. Agent position is ALWAYS observed;
# the target-derived slots are zeroed while hidden. The visible slots reuse ChaseObs' encoding
# exactly so the visible-phase geometry matches the plain chase example.
static func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2, visible: bool) -> Array:
	var base: Array = ChaseObs.compute_obs(agent_pos, target_pos, arena_size)
	if visible:
		base.append(1.0)
		return base
	return [base[0], base[1], 0.0, 0.0, 0.0, 0.0]
