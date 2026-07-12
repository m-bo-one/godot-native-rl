extends Node
# Screenshot capture for the solid-hurdle jumping demo (#286). Renders the deployed net on the
# solid course and saves viewport frames while the creature is NEAR a hurdle (approach -> leap ->
# land), so the saved set focuses on the jumps. Each filename encodes torso z / height / distance
# to the next hurdle, so the best leap frames are pickable without opening them.
#
# Run (needs a display -> xvfb):
#   xvfb-run -a godot --path . res://examples/quadruped_walk/quadruped_jump_capture.tscn \
#     -- out=/abs/dir frames=1600

@export var game_path: NodePath
@export var track_path: NodePath

var _game
var _track
var _frame := 0
var _shot := 0
var _out_dir := ""
var _max_frames := 1600
const WARMUP := 20
const NEAR_AHEAD := 3.0    ## start capturing this far before a hurdle
const NEAR_BEHIND := 2.5   ## keep capturing until this far past it

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_track = get_node_or_null(track_path)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("out="):
			_out_dir = a.substr("out=".length())
		elif a.begins_with("frames="):
			_max_frames = maxi(1, int(a.substr("frames=".length())))
	if _out_dir.is_empty():
		_out_dir = OS.get_user_data_dir().path_join("jump_shots")
	DirAccess.make_dir_recursive_absolute(_out_dir)

func _process(_delta: float) -> void:
	_frame += 1
	if _frame < WARMUP:
		return
	if _frame > _max_frames + WARMUP:
		print("CAPTURE DONE: %d shots -> %s" % [_shot, _out_dir])
		get_tree().quit(0)
		return
	if _game == null:
		return
	var tz: float = _game.torso_pos().z
	var dist: float = _track.dist_to_next_hurdle(tz) if _track != null else 1e9
	# Capture only within the approach/clear window of a hurdle, every other frame.
	if dist > NEAR_AHEAD or dist < -NEAR_BEHIND or _frame % 2 != 0:
		return
	await RenderingServer.frame_post_draw
	var img: Image = get_viewport().get_texture().get_image()
	if img == null:
		return
	var ty: float = _game.torso_pos().y
	var path := _out_dir.path_join("shot_%04d_z%05.1f_y%.2f_d%+.1f.png" % [_shot, tz, ty, dist])
	img.save_png(path)
	_shot += 1
