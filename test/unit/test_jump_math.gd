extends SceneTree
# Unit tests for the pure jump-launch reward (#286): the launch is rewarded only while a solid
# hurdle is close ahead, is clamped, and never rewards downward motion.

const Harness = preload("res://test/harness.gd")
const JumpMath = preload("res://examples/quadruped_walk/jump_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# In-zone, rising -> rewards the (clamped) upward velocity.
	h.assert_true(absf(JumpMath.approach_jump_reward(2.0, 1.0, 2.0, 3.0) - 2.0) < 1e-6,
		"in-zone rising rewards vy")
	h.assert_true(absf(JumpMath.approach_jump_reward(5.0, 0.5, 2.0, 3.0) - 3.0) < 1e-6,
		"upward velocity is capped")
	# dist == zone boundary is inclusive; dist == 0 (at the wall) is inclusive.
	h.assert_true(absf(JumpMath.approach_jump_reward(1.0, 2.0, 2.0, 3.0) - 1.0) < 1e-6,
		"zone boundary inclusive")
	h.assert_true(absf(JumpMath.approach_jump_reward(1.0, 0.0, 2.0, 3.0) - 1.0) < 1e-6,
		"at the wall (dist 0) inclusive")

	# Out of zone (too far ahead) -> no reward.
	h.assert_eq(JumpMath.approach_jump_reward(2.0, 3.0, 2.0, 3.0), 0.0, "beyond zone -> 0")
	# Hurdle already behind (dist < 0) -> no reward (don't pay launches after clearing).
	h.assert_eq(JumpMath.approach_jump_reward(2.0, -0.5, 2.0, 3.0), 0.0, "hurdle behind -> 0")
	# Falling / no upward motion -> no reward even in-zone.
	h.assert_eq(JumpMath.approach_jump_reward(-1.0, 1.0, 2.0, 3.0), 0.0, "downward in-zone -> 0")
	h.assert_eq(JumpMath.approach_jump_reward(0.0, 1.0, 2.0, 3.0), 0.0, "no vertical vel -> 0")

	h.finish(self)
