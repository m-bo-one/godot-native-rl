extends "res://test/integration/trained_checker_base.gd"
# Memory-chase behavioral gate (#378), two modes in one checker:
#  - normal (require_memory=true):  the trained LSTM net must catch >= min_catches — proving the
#    recurrent deploy path (state carry through run_inference_multi) drives real behavior.
#  - ablated (require_memory=false, scene sets agent.ablate_memory=true): the SAME net with its
#    hidden state zeroed before every decision must catch <= max_catches_ablated. Same weights,
#    same graph — only the carried memory differs. If memory were decorative the two scores would
#    match; the gap is the proof that the LSTM memory is load-bearing.

@export var require_memory := true
@export var min_catches := 5
@export var max_catches_ablated := 3

func _label() -> String:
	return "TRAINED MEMORY CHASE" if require_memory else "ABLATED MEMORY CHASE"

func _passed() -> bool:
	if require_memory:
		return int(_game.catches) >= min_catches
	return int(_game.catches) <= max_catches_ablated

func _report() -> String:
	if require_memory:
		if _passed():
			return "%d catches in %d frames with LSTM memory" % [int(_game.catches), _frames]
		return "only %d catches in %d frames (need %d) — recurrent net did not chase from memory" % [int(_game.catches), _frames, min_catches]
	if _passed():
		return "memory ablated: %d catches in %d frames (<= %d) — memory is load-bearing" % [int(_game.catches), _frames, max_catches_ablated]
	return "memory ablated but still %d catches in %d frames (> %d) — memory looks decorative, gate the net harder" % [int(_game.catches), _frames, max_catches_ablated]
