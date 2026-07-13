extends CanvasLayer

# DemoLegend (#229): a drop-in legend for example play scenes. Shows 1-3 short lines telling a
# first-time viewer which entity is the RL agent and what the goal is, plus a standing
# "native ncnn inference" footer. The F3 hint is appended ONLY when it is true — a
# PolicyDebugOverlay is in the scene AND this is a debug build (the overlay is debug-gated), so
# a release/web export never advertises a keybinding that does nothing. Cosmetic only: never
# touches obs/actions, inert headless, safe to add to any deploy scene.
#
# Usage: add the node, set `lines`, e.g.
#   lines = ["BLUE square = RL agent", "RED square = moving target"]

const FOOTER := "native ncnn inference"
const F3_HINT := " — F3 for live obs/actions"

@export var lines: PackedStringArray = PackedStringArray()

var _label: Label = null


func legend_text() -> String:
	var parts: Array = []
	for line in lines:
		parts.append(String(line))
	parts.append(FOOTER + (F3_HINT if _f3_available() else ""))
	return "\n".join(parts)


# True only when pressing F3 would actually do something: the overlay node exists in the scene
# and the build is a debug build (PolicyDebugOverlay no-ops in release builds).
func _f3_available() -> bool:
	if not OS.is_debug_build() or not is_inside_tree():
		return false
	var root := get_tree().root
	return _find_overlay(root) != null


func _find_overlay(node: Node) -> Node:
	if node.get_script() != null and String(node.get_script().resource_path).ends_with("policy_debug_overlay.gd"):
		return node
	for child in node.get_children():
		var found := _find_overlay(child)
		if found != null:
			return found
	return null


func _ready() -> void:
	layer = 90  # above the game canvas; bottom-left placement keeps clear of the F3 overlay (top-left)
	var panel := PanelContainer.new()
	panel.name = "Panel"
	# Bottom-left, out of the way of HUDs (top) and the F3 overlay (left column starts on toggle).
	panel.anchors_preset = Control.PRESET_BOTTOM_LEFT
	panel.anchor_top = 1.0
	panel.anchor_bottom = 1.0
	panel.offset_left = 8.0
	panel.offset_top = -8.0
	panel.offset_bottom = -8.0
	panel.grow_vertical = Control.GROW_DIRECTION_BEGIN
	panel.self_modulate = Color(1, 1, 1, 0.85)
	add_child(panel)
	_label = Label.new()
	_label.name = "Label"
	_label.text = legend_text()
	_label.add_theme_font_size_override("font_size", 13)
	panel.add_child(_label)
