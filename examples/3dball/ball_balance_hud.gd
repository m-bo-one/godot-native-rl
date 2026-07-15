extends Label
# Minimal balance HUD for the 3DBall play scene (#229): how long the trained net has kept the
# ball up, the best streak, and the fall count. Reads the game's cosmetic counters only.

@export var game_path: NodePath
var _game

func _ready() -> void:
	_game = get_node_or_null(game_path)

func _process(_delta: float) -> void:
	if _game == null:
		text = "balance: n/a"
		return
	text = "balanced: %4.1f s   best: %4.1f s   falls: %d" % [
		_game.balance_frames / 60.0, _game.best_balance_frames / 60.0, _game.falls]
