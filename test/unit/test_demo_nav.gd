extends SceneTree
# Guards the demo_nav back-button (#265-adjacent web fix): the autoload builds an on-screen Menu
# button (so web/touch users can return to the launcher when Escape is unavailable) that does not
# steal keyboard focus from the demo.

const Harness = preload("res://test/harness.gd")
const DemoNav = preload("res://examples/demo_nav.gd")

func _initialize() -> void:
	var h := Harness.new()

	var nav = DemoNav.new()
	get_root().add_child(nav)
	nav._ready()  # _ready does NOT auto-fire from _initialize add_child in a headless SceneTree (gotcha)

	# It builds exactly one CanvasLayer (always-on-top) holding the back button.
	var layers: Array = nav.get_children().filter(func(c): return c is CanvasLayer)
	h.assert_eq(layers.size(), 1, "demo_nav builds one CanvasLayer")
	h.assert_true((layers[0] as CanvasLayer).layer >= 100, "back-button layer renders above the demo")

	# The button exists and never steals keyboard focus from the demo.
	var btn = nav._button
	h.assert_true(btn is Button, "demo_nav builds a Button")
	h.assert_eq(btn.focus_mode, Control.FOCUS_NONE, "back button does not steal keyboard focus")
	h.assert_true(btn.pressed.is_connected(nav.go_to_launcher), "button press routes to go_to_launcher")

	# With no scene loaded we are NOT on the launcher, so go_to_launcher is not a self-no-op trap and
	# the button would be shown.
	h.assert_true(not nav._on_launcher(), "_on_launcher false when not on the launcher")

	nav.free()
	h.finish(self)
