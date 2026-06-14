extends Node
# Sequential generation "race" (#60 M4). Rather than race several articulated creatures at once —
# which is unstable in a shared Jolt physics space (3 ragdolls contend for the solver budget and all
# gaits collapse; see PR #219) — ONE creature runs each training generation in turn in clean solo
# physics, and we record how far each got. The leaderboard then shows the learning arc as a
# standings table. Robust + honest: each distance is the creature's true unobstructed reach.
#
# Generations are (param, bin, label) triples. Between phases we swap_model() the agent and reset the
# world. Ranking is the pure, unit-tested race_math.gd.

const R = preload("res://examples/quadruped_walk/race_math.gd")

@export var game_path: NodePath
@export var agent_path: NodePath
@export var label_path: NodePath
@export var frames_per_gen := 4000
# Parallel arrays (kept simple for the .tscn): one entry per generation.
@export var gen_params: Array[String] = []
@export var gen_bins: Array[String] = []
@export var gen_labels: Array[String] = []

signal race_finished(results: Array)  # Array of {"label": String, "distance": float}

var _game
var _agent
var _label: Label
var _phase := -1
var _frames := 0
var _max_z := 0.0
var _results: Array = []
var _done := false

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	_label = get_node_or_null(label_path) as Label
	# Defer so the agent's ncnn runner is created (its _ready ran) before the first swap_model.
	call_deferred("_start_phase", 0)

func results() -> Array:
	return _results

func is_done() -> bool:
	return _done

func _start_phase(i: int) -> void:
	_phase = i
	_frames = 0
	_max_z = 0.0
	if _agent != null and i < gen_params.size():
		_agent.swap_model(gen_params[i], gen_bins[i])
	if _game != null:
		_game.reset_positions()

func _physics_process(_delta: float) -> void:
	if _done or _game == null or _agent == null:
		return
	_max_z = maxf(_max_z, _game.torso_pos().z)
	_frames += 1
	_update_label()
	if _frames >= frames_per_gen:
		var label: String = gen_labels[_phase] if _phase < gen_labels.size() else ("gen %d" % _phase)
		_results.append({"label": label, "distance": _max_z})
		if _phase + 1 < gen_params.size():
			_start_phase(_phase + 1)
		else:
			_finish()

func _finish() -> void:
	_done = true
	_update_label()
	race_finished.emit(_results)

func _update_label() -> void:
	if _label == null:
		return
	var total: int = gen_params.size()
	var lines: Array = []
	if _done:
		lines.append("🏁 GENERATION RACE — FINISHED")
	else:
		# Prominent "now playing" banner with a live countdown, so it's unmistakable that the race
		# auto-cycles through generations (the creature looks similar each gen, so without this the
		# switch is easy to miss and the race reads as "stuck on 500k").
		lines.append("🏁 GENERATION RACE")
		if _phase >= 0 and _phase < gen_labels.size():
			var frames_left: int = maxi(0, frames_per_gen - _frames)
			var secs: int = R.countdown_seconds(frames_left, Engine.physics_ticks_per_second)
			lines.append(R.format_now(_phase + 1, total, gen_labels[_phase], _max_z, secs))
		lines.append("")
	# Finished generations, ranked.
	if _results.size() > 0:
		lines.append("FINAL STANDINGS" if _done else "STANDINGS")
		var dists: Array = []
		for r in _results:
			dists.append(r["distance"])
		var order := R.standings(dists)
		for rank in range(order.size()):
			var r = _results[order[rank]]
			lines.append(R.format_row(rank + 1, r["label"], r["distance"]))
	_label.text = "\n".join(lines)
