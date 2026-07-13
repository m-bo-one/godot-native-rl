extends SceneTree

# DemoLegend (#229): a drop-in CanvasLayer legend for the example play scenes — 1-3 short lines
# telling a first-time viewer which entity is the RL agent, what the goal is, and that inference
# is native ncnn. Cosmetic only (never touches obs/actions); must be inert headless.

const Harness = preload("res://test/harness.gd")
const DemoLegend = preload("res://addons/godot_native_rl/debug/demo_legend.gd")

func _initialize() -> void:
	var h := Harness.new()

	var legend = DemoLegend.new()
	legend.lines = PackedStringArray(["BLUE = RL agent", "RED = target"])
	get_root().add_child(legend)
	await process_frame

	# The composed text carries every line plus the standing native-ncnn footer. WITHOUT a
	# PolicyDebugOverlay in the scene, the F3 hint must NOT appear (it would be a false
	# instruction — the overlay is what F3 toggles).
	var text: String = legend.legend_text()
	h.assert_true(text.contains("BLUE = RL agent"), "first line present")
	h.assert_true(text.contains("RED = target"), "second line present")
	h.assert_true(text.contains("ncnn"), "native-ncnn footer present")
	h.assert_true(not text.contains("F3"), "no F3 hint without a PolicyDebugOverlay in the scene")

	# With an overlay present (debug build — headless test runs are), the hint appears.
	var overlay := CanvasLayer.new()
	overlay.set_script(load("res://addons/godot_native_rl/debug/policy_debug_overlay.gd"))
	get_root().add_child(overlay)
	await process_frame
	h.assert_true(legend.legend_text().contains("F3"), "F3 hint present with a PolicyDebugOverlay")
	overlay.free()

	# The label node exists and carries the composed text (renders bottom-left when drawn).
	var label: Label = legend.get_node_or_null("Panel/Label")
	h.assert_true(label != null, "legend builds a Panel/Label")
	h.assert_eq(label.text, text, "label text == legend_text()")

	# Empty lines still yields the footer (a scene can use it as a bare ncnn badge).
	var bare = DemoLegend.new()
	get_root().add_child(bare)
	await process_frame
	h.assert_true(bare.legend_text().contains("ncnn"), "bare legend still shows the ncnn badge")

	h.finish(self)
