extends Node
# Headless behavioral floor: runs BOTH trained policies (seeker via its model, hider via its) under
# ncnn inference and asserts the SEEKER actively PURSUES — it both (a) keeps moving and (b) keeps the
# hider in line-of-sight. The motion gate (`min_seeker_motion`) is the key #274 regression: the prior
# binary-LOS reward had a degenerate static optimum and the trained seeker FROZE in place, which the
# old LOS-only floor still passed (a frozen-in-sight seeker counts as LOS). After the dense
# distance-closing reward + reduced occluders the seeker moves ~4.5 px/frame (near max) and closes for
# occasional catches; a frozen seeker fails the motion floor. The golden-inference test
# (test/unit/test_hide_seek_multipolicy_golden_inference.gd) is the precise deterministic regression;
# this is the behavioral one. Deterministic via the seeded spawn RNG.

@export var game_path: NodePath
@export var seeker_path: NodePath
@export var hider_path: NodePath
@export var frames_to_run := 3000
@export var min_los_fraction := 0.08      ## seeker holds line-of-sight at least this often
@export var min_seeker_motion := 0.0      ## px the seeker must travel (freeze guard); 0 = off (opt-in)
@export var min_catches := 0              ## the seeker must close for >= this many catches; 0 = off
@export var rng_seed := 1  ## seed the game's spawn RNG so the run is deterministic

var _game
var _seeker_node
var _frames := 0
var _catches := 0
var _los_frames := 0
var _was_caught_last := false
var _seeker_motion := 0.0
var _seeker_prev := Vector2.ZERO

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var seeker = get_node_or_null(seeker_path)
	var hider = get_node_or_null(hider_path)
	if _game == null or seeker == null or hider == null:
		_fail("could not resolve game/seeker/hider nodes")
		return
	for a in [seeker, hider]:
		if a._ncnn_runner == null or not a._ncnn_runner.is_model_loaded():
			_fail("a trained ncnn model is not loaded")
			return
	# Deterministic run: seed the spawn RNG and re-spawn so the run is reproducible
	# (the policies are argmax-deterministic; spawn positions were the only nondeterminism).
	if _game.has_method("seed_rng"):
		_game.seed_rng(rng_seed)
		_game.reset_positions()
	_seeker_prev = _game.seeker_pos()

func _physics_process(_delta: float) -> void:
	if _game == null:
		return
	# Rising-edge catch count + accumulate the seeker's path length (the freeze guard).
	var caught: bool = _game.was_caught()
	if caught and not _was_caught_last:
		_catches += 1
	_was_caught_last = caught
	if _game.has_los():
		_los_frames += 1
	var sp: Vector2 = _game.seeker_pos()
	_seeker_motion += sp.distance_to(_seeker_prev)
	_seeker_prev = sp
	_frames += 1
	if _frames >= frames_to_run:
		var los_frac := float(_los_frames) / float(_frames)
		var report := "los=%.1f%% (%d/%d), seeker_motion=%.0fpx, catches=%d" % [100.0 * los_frac, _los_frames, _frames, _seeker_motion, _catches]
		# The seeker must MOVE (not freeze), keep sight, and close for catches. The motion gate is the
		# #274 regression — a frozen-in-sight seeker passes LOS but fails motion.
		# Motion + catch gates are opt-in (0 = off) so the shared pettingzoo eval keeps its lenient
		# LOS-only floor; the hide&seek multipolicy eval scene sets the strict #274 thresholds.
		if min_seeker_motion > 0.0 and _seeker_motion < min_seeker_motion:
			_fail("%s — seeker_motion below %.0fpx floor (FROZE — the #274 regression)" % [report, min_seeker_motion])
		elif los_frac < min_los_fraction:
			_fail("%s below LOS floor %.0f%% — seeker did not pursue" % [report, 100.0 * min_los_fraction])
		elif _catches < min_catches:
			_fail("%s below catch floor %d — seeker pursued but never closed" % [report, min_catches])
		else:
			print("MULTI-POLICY HIDE&SEEK PASSED (%s)" % report)
			get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("MULTI-POLICY HIDE&SEEK FAILED: %s" % reason)
	get_tree().quit(1)
