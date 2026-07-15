extends Node
# Test stub (#379): a Node agent (as NcnnSync.agents_training requires) that wraps a REAL
# NcnnControllerCore and reproduces an example agent's _physics_process lifecycle at the horizon —
# step() fires the horizon, then the agent calls reset() in the SAME frame (chase_agent.gd:137),
# BEFORE the Sync reads the step. Exposes the three duck-typed methods the Sync reads through so a
# wire-level test can prove the horizon still reaches the wire as truncated=[true].

const CoreScript = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

var _core = CoreScript.new()

# Drive the core to a horizon end and reset in the same frame, exactly as an agent does.
func drive_horizon_then_reset(reset_after: int) -> void:
	for _i in range(reset_after + 1):  # n_steps ends at reset_after + 1 > reset_after -> horizon fires
		_core.step(reset_after)
	_core.reset()  # in-frame reset (before the Sync reads) — must NOT clear truncated (#379)

func get_truncated() -> bool:
	return _core.get_truncated()

func get_done() -> bool:
	return _core.get_done()

func set_done_false() -> void:
	_core.set_done_false()
