extends Node
# Autoload (registered in the shipped examples project.godot, #226): returns to the launcher menu
# from any demo, so a user never gets stuck inside a demo with no way back.
#
# Escape (ui_cancel) works on DESKTOP, but a WEB export can't rely on it — browsers reserve the
# Escape key (it exits fullscreen / pointer-lock) so the keydown often never reaches Godot. So we
# ALSO show a small on-screen "Menu" button that returns to the launcher on click/tap — which works
# on web, touch, AND desktop. The button is hidden while on the launcher itself.

const LAUNCHER := "res://examples/launcher.tscn"

var _layer: CanvasLayer
var _button: Button

func _ready() -> void:
	# Build the always-on-top back button. As an autoload child it persists across scene changes, so
	# it's present on every demo without each scene wiring it.
	_layer = CanvasLayer.new()
	_layer.layer = 128  # above the demo + any in-scene overlays
	add_child(_layer)
	_button = Button.new()
	_button.text = "☰ Menu"           # hamburger glyph + label
	_button.focus_mode = Control.FOCUS_NONE  # never steal keyboard focus from the demo
	_button.set_anchors_preset(Control.PRESET_TOP_LEFT)
	_button.position = Vector2(12, 12)
	_button.pressed.connect(go_to_launcher)
	_layer.add_child(_button)

func _process(_delta: float) -> void:
	# Don't show "Menu" while already on the launcher.
	if _button != null:
		_button.visible = not _on_launcher()

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		go_to_launcher()

func _on_launcher() -> bool:
	if not is_inside_tree():
		return false
	var current := get_tree().current_scene
	return current != null and current.scene_file_path == LAUNCHER

# Switch to the launcher (no-op if already there or the launcher scene is missing). Public so the
# button + the Escape handler share one path.
func go_to_launcher() -> void:
	if _on_launcher():
		return
	if ResourceLoader.exists(LAUNCHER):
		get_tree().change_scene_to_file(LAUNCHER)
