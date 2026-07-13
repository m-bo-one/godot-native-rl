# Memory chase agent (#378): ChaseAgent with a partially-observable, blinking target. The target
# is fully observed for `visible_steps` decisions, then hidden for `hidden_steps` (target-derived
# obs slots zeroed + a visible flag), so a recurrent (LSTM) policy must REMEMBER where the target
# was and integrate its own motion while blind — a feed-forward policy has nothing to steer by.
# Game, reward, and actions are plain chase; only the observation is restricted.
extends "res://examples/chase_the_target/chase_agent.gd"

const MemObs = preload("res://examples/chase_the_target/chase_memory_obs.gd")

@export var visible_steps := 2
@export var hidden_steps := 6
## Regression-only ablation: zero the recurrent hidden state before EVERY decision, so the same
## trained net runs memoryless. The memory-is-load-bearing check compares this against normal
## state carry — if memory were decorative the two scores would match. Never set in shipped scenes.
@export var ablate_memory := false

# The blink cycle advances per DECISION (one get_obs per decision in both TRAINING and
# NCNN_INFERENCE), not per physics frame — n_steps counts frames. The flag inside the obs always
# tells the truth about what the policy sees, so an extra startup probe (get_obs_space) merely
# phase-shifts the schedule; it can never lie to the policy.
var _decision_count := 0
var _last_visible := true

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
	_last_visible = MemObs.visible_for_step(_decision_count, visible_steps, hidden_steps)
	_decision_count += 1
	if ablate_memory:
		reset_recurrent_state()
	return {"obs": MemObs.compute_obs(
		_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size, _last_visible)}

# For the deploy scene's blink rendering: whether the LAST decision's obs contained the target.
func is_target_visible() -> bool:
	return _last_visible

func reset() -> void:
	super.reset()
	_decision_count = 0
	_last_visible = true

func get_debug_status() -> Dictionary:
	var status := super.get_debug_status()
	status["target_visible"] = _last_visible
	return status
