extends SceneTree
# Unit tests for the pure variable-length entity-observation helper (#46 M1).

const Harness = preload("res://test/harness.gd")
const M = preload("res://addons/godot_native_rl/sensors/entity_obs_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	# obs_size = N*F + N (entity block + presence flags).
	h.assert_eq(M.obs_size(3, 2), 9, "obs_size 3x2 = 9")
	h.assert_eq(M.obs_size(4, 0), 4, "obs_size flags-only = N")

	# Fewer than N: real row then zero-padding, flags mark presence.
	var one: Array = M.build_obs([{"dist": 1.0, "feat": [0.5, -0.5]}], 3, 2)
	h.assert_eq(one, [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "one entity padded + flags")

	# More than N: keep the nearest N, sorted ascending by distance.
	var capped: Array = M.build_obs([
		{"dist": 5.0, "feat": [5.0]},
		{"dist": 1.0, "feat": [1.0]},
		{"dist": 3.0, "feat": [3.0]}], 2, 1)
	h.assert_eq(capped, [1.0, 3.0, 1.0, 1.0], "nearest-2 sorted + flags")

	# Ties keep original input order (stable).
	var tie: Array = M.build_obs([
		{"dist": 2.0, "feat": [10.0]},
		{"dist": 2.0, "feat": [20.0]}], 2, 1)
	h.assert_eq(tie, [10.0, 20.0, 1.0, 1.0], "ties keep input order")

	# No entities: all zeros, all flags 0.
	var none: Array = M.build_obs([], 2, 2)
	h.assert_eq(none, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "no entities -> zeros + zero flags")

	# Defensive width fit: a wrong-width feat row is truncated/padded to F.
	var fit: Array = M.build_obs([{"dist": 0.0, "feat": [1.0, 2.0, 3.0]}], 1, 2)
	h.assert_eq(fit, [1.0, 2.0, 1.0], "row fit to feat width")

	h.finish(self)
