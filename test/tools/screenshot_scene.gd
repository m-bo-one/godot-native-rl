extends SceneTree

# Screenshot tool for the examples polish pass (#229): loads a scene, lets it run a bit, then
# saves the viewport to PNG and quits. Needs a rendering context (run under xvfb-run; headless
# can't render). Not part of the test suite — a manual verification tool.
#
#   xvfb-run -a godot --path . --script res://test/tools/screenshot_scene.gd \
#       -- --scene=res://examples/3dball/ball_balance.tscn --out=/tmp/3dball.png --frames=150

var _scene_path := ""
var _out_path := "/tmp/screenshot.png"
var _frames := 150
var _elapsed := 0

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--scene="):
			_scene_path = arg.split("=", true, 1)[1]
		elif arg.begins_with("--out="):
			_out_path = arg.split("=", true, 1)[1]
		elif arg.begins_with("--frames="):
			_frames = int(arg.split("=", true, 1)[1])
	if _scene_path == "":
		push_error("screenshot_scene: pass --scene=res://... after --")
		quit(1)
		return
	var err := change_scene_to_file(_scene_path)
	if err != OK:
		push_error("screenshot_scene: cannot load %s (err %d)" % [_scene_path, err])
		quit(1)

func _process(_delta: float) -> bool:
	_elapsed += 1
	if _elapsed < _frames:
		return false
	var img := get_root().get_texture().get_image()
	var save_err := img.save_png(_out_path)
	if save_err != OK:
		push_error("screenshot_scene: cannot save %s (err %d)" % [_out_path, save_err])
		quit(1)
		return false
	print("saved: ", _out_path)
	quit(0)
	return false
