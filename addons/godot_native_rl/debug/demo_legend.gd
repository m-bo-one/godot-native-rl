extends CanvasLayer

# DemoLegend (#229): a drop-in legend for example play scenes. Shows 1-3 short lines telling a
# first-time viewer which entity is the RL agent and what the goal is, plus a standing footer
# ("native ncnn inference - F3 for live obs/actions"). Cosmetic only: never touches obs/actions,
# inert headless, safe to add to any deploy scene. Pair with PolicyDebugOverlay (the F3 target).
#
# Usage: add the node, set `lines`, e.g.
#   lines = ["BLUE square = RL agent", "RED square = moving target"]

const FOOTER := "native ncnn inference — F3 for live obs/actions"

@export var lines: PackedStringArray = PackedStringArray()

var _label: Label = null


func legend_text() -> String:
	var parts: Array = []
	for line in lines:
		parts.append(String(line))
	parts.append(FOOTER)
	return "\n".join(parts)


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
