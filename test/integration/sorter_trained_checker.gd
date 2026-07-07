extends Node
# Trained Sorter behavioral check (#46/#260 M4): under ncnn inference the attention-encoder policy
# must SOLVE variable-count episodes — visit the numbered tiles in ascending order — and not by
# luck (correct visits must clearly outnumber wrong-order visits). The net is a behavior-cloned
# copy of the scripted expert (vanilla PPO stalls on the sparse strict-ordering reward); this pins
# that the cloned policy deploys through ncnn and reproduces the solving behavior.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 2400
@export var min_solved := 3          ## episodes fully sorted under ncnn
@export var min_correct := 12        ## total correct-order visits
@export var max_wrong_ratio := 0.5   ## wrong visits must stay under half the correct ones
## Anti-livelock: if no correct visit lands within this window, reseed the episode so a single
## unlucky layout can't stall the whole check (episodes are ~200-360 frames at the slow gait).
@export var episode_timeout_frames := 500

var _game
var _agent
var _frames := 0
var _last_correct := 0
var _since_progress := 0
var _timeouts := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")
		return
	_game.seed_rng(3)
	# Re-roll the first layout from the seeded stream (world._ready already rolled one from the
	# unseeded RNG in tree order — the documented chase/gridworld gotcha).
	_game.reset_episode()

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	_frames += 1
	var correct: int = _game.correct_visits
	if correct != _last_correct:
		_last_correct = correct
		_since_progress = 0
	else:
		_since_progress += 1
		if _since_progress >= episode_timeout_frames:
			_timeouts += 1
			_since_progress = 0
			_game.reset_episode()
	if _frames >= frames_to_run:
		var solved: int = _game.episodes_solved
		var wrong: int = _game.wrong_visits
		print("Sorter trained stats: solved=%d correct=%d wrong=%d timeouts=%d frames=%d"
			% [solved, correct, wrong, _timeouts, _frames])
		if solved < min_solved:
			_fail("only %d episodes solved (need %d)" % [solved, min_solved])
			return
		if correct < min_correct:
			_fail("only %d correct visits (need %d)" % [correct, min_correct])
			return
		if float(wrong) > float(correct) * max_wrong_ratio:
			_fail("%d wrong vs %d correct — not visiting in order" % [wrong, correct])
			return
		print("TRAINED SORTER PASSED (solved=%d correct=%d wrong=%d in %d frames)"
			% [solved, correct, wrong, _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("TRAINED SORTER FAILED: %s" % reason)
	get_tree().quit(1)
