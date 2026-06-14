extends Node3D
# Cosmetic goal markers for the FlyBy play scene. FlyByGame keeps its goals as bare Node3D markers
# so the sim stays headless-deterministic and unit-testable; this node renders each one as a glowing,
# semi-transparent checkpoint orb (sized to the actual reach radius) and highlights the goal the plane
# is currently flying toward — so a viewer can SEE what the policy is doing (it was previously a plane
# banking around empty space). Connects to the game's `goal_reached` signal to advance the highlight.
# Purely visual: omitting this node changes nothing about the sim.

const CURRENT := Color(1.0, 0.85, 0.1)   ## bright gold = the active target
const PENDING := Color(0.2, 0.6, 1.0)    ## dim cyan = upcoming goals

@export var game_path: NodePath
@export var goals_path: NodePath

var _game
var _markers: Array = []  # MeshInstance3D per goal, indexed like the game's goals

# Pure: which color a goal gets, given whether it's the plane's current target.
static func color_for(is_current: bool) -> Color:
	return CURRENT if is_current else PENDING

func _ready() -> void:
	_game = get_node_or_null(game_path)
	var goals_parent: Node = get_node_or_null(goals_path)
	if goals_parent == null:
		push_warning("FlyByMarkers: goals_path not set — no markers rendered.")
		return
	var radius: float = _game.goal_radius if _game != null else 6.0
	for child in goals_parent.get_children():
		if child is Node3D:
			var orb := _make_orb(radius)
			child.add_child(orb)
			_markers.append(orb)
	if _game != null and _game.has_signal("goal_reached"):
		_game.goal_reached.connect(_refresh)
	_refresh()

func _make_orb(radius: float) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.emission_enabled = true
	mi.material_override = mat
	return mi

# Recolor every orb so the plane's current target glows brightly and the rest stay dim. Called once
# at startup and again each time a goal is reached (goal_index has already advanced by then).
func _refresh(_unused = null) -> void:
	var current: int = _game.goal_index if _game != null else 0
	for i in range(_markers.size()):
		var mat: StandardMaterial3D = _markers[i].material_override
		var is_current := (i == current)
		var c := color_for(is_current)
		mat.albedo_color = Color(c.r, c.g, c.b, 0.35)
		mat.emission = c
		mat.emission_energy_multiplier = 2.5 if is_current else 0.5
