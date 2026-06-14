extends SceneTree
# Unit test for the FlyBy goal-marker color helper (pure, no scene).

const Harness = preload("res://test/harness.gd")
const M = preload("res://examples/fly_by/fly_by_markers.gd")

func _initialize() -> void:
	var h = Harness.new()

	# The active target is the bright gold; everything else is the dim pending color. They differ.
	h.assert_eq(M.color_for(true), M.CURRENT, "current goal -> CURRENT color")
	h.assert_eq(M.color_for(false), M.PENDING, "other goals -> PENDING color")
	h.assert_true(M.CURRENT != M.PENDING, "current and pending are visually distinct")

	h.finish(self)
