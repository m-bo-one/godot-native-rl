extends SceneTree
# CurriculumController.parse_stages_arg: pure cmdline-token parser (#198) for the launch-time
# `curriculum_stages=<res://…>` override. Args are injected so the parse is unit-testable
# headlessly (mirrors RunSpeed's parse convention). No node/scene involved.

const Harness = preload("res://test/harness.gd")
const Controller = preload("res://addons/godot_native_rl/training/curriculum_controller.gd")

func _initialize() -> void:
	var h = Harness.new()

	# token present -> returns the path
	h.assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["speedup=8", "curriculum_stages=res://a/b.json", "action_repeat=8"])),
		"res://a/b.json", "extracts curriculum_stages value")
	# absent -> empty
	h.assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["speedup=8", "action_repeat=8"])),
		"", "absent -> empty string")
	# bare key without =value -> empty
	h.assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["curriculum_stages"])),
		"", "bare key without =value -> empty string")
	# no args -> empty
	h.assert_eq(Controller.parse_stages_arg(PackedStringArray([])),
		"", "no args -> empty string")
	# single token parses
	h.assert_eq(Controller.parse_stages_arg(
		PackedStringArray(["curriculum_stages=res://x.json"])),
		"res://x.json", "single token parses")

	h.finish(self)
