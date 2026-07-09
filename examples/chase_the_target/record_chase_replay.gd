extends Node
# Headless driver (#40): records ONE chase episode from the deployed ncnn net into a
# gnrl_replay_v1 JSON, so scripts/record_replay_video.sh can then render it to a video clip.
# The trained ChaseAgent (control_mode=3) is driven by the sibling Sync (NCNN_INFERENCE mode);
# a ReplayRecorder taps its inference_step signal (no trainer, no socket). After `frames` physics
# ticks it flushes the buffered episode, copies the newest ring file to `out`, and quits.
#
# Run: godot --headless --path . res://examples/chase_the_target/record_chase_replay.tscn \
#        -- out=user://chase_replay.json frames=900

const Recorder = preload("res://addons/godot_native_rl/training/replay_recorder.gd")

@export var game_path: NodePath
@export var agent_path: NodePath

const REC_DIR := "user://chase_replay_rec"

var _frames := 900
var _out := "user://chase_replay.json"
var _recorder
var _agent: Node
var _count := 0
var _finished := false

func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("frames="):
			_frames = maxi(1, int(a.substr("frames=".length())))
		elif a.begins_with("out="):
			_out = a.substr("out=".length())

	_agent = get_node_or_null(agent_path)
	var game := get_node_or_null(game_path)
	if _agent == null or game == null:
		printerr("record_chase_replay: missing agent/game node")
		get_tree().quit(1)
		return

	_recorder = Recorder.new()
	# Absolute paths so they resolve wherever the recorder sits in the tree. game_path drives
	# initial_state via get_replay_state(); sync_path points at the (inference-mode) Sync so the
	# recorder's _ready attach_sync is a no-op tap (its training signals never fire here) rather
	# than a spurious "no Sync found" error — the inference path below does the actual capture.
	_recorder.game_path = game.get_path()
	var sync := get_node_or_null(NodePath("../Sync"))
	if sync != null:
		_recorder.sync_path = sync.get_path()
	_recorder.out_dir = REC_DIR
	_recorder.keep_last = 2
	add_child(_recorder)
	_recorder.attach_agent(_agent)

func _physics_process(_delta: float) -> void:
	if _finished:
		return
	_count += 1
	if _count < _frames:
		return
	_finished = true
	_recorder.flush_inference_episodes()
	var written := _newest_ring_file()
	if written.is_empty():
		printerr("record_chase_replay: no replay episode was written (episode shorter than one decision window?)")
		get_tree().quit(1)
		return
	# Copy the newest ring file to the requested output path.
	var bytes := FileAccess.get_file_as_bytes(written)
	var f := FileAccess.open(_out, FileAccess.WRITE)
	if f == null:
		printerr("record_chase_replay: cannot write '%s'" % _out)
		get_tree().quit(1)
		return
	f.store_buffer(bytes)
	f.close()
	print("RECORDED REPLAY: %s (%d bytes) from %s" % [_out, bytes.size(), written])
	get_tree().quit(0)

func _newest_ring_file() -> String:
	var dir := DirAccess.open(REC_DIR)
	if dir == null:
		return ""
	var best := ""
	for name in dir.get_files():
		if name.ends_with(".json") and name > best:
			best = name
	return REC_DIR.path_join(best) if not best.is_empty() else ""
