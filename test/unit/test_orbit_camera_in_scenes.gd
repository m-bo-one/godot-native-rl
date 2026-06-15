extends SceneTree
# Structure regression (#265): every 3D demo that should have the orbit camera carries exactly one
# node running orbit_camera.gd. Scenes are instantiated WITHOUT entering the tree (no _ready / no
# ncnn / no inference) — we only assert the node is wired in.

const Harness = preload("res://test/harness.gd")
const ORBIT := "res://addons/godot_native_rl/camera/orbit_camera.gd"

const SCENES: Array[String] = [
	"res://examples/quadruped_walk/quadruped_walk_track.tscn",
	"res://examples/quadruped_walk/quadruped_hurdles_track.tscn",
	"res://examples/quadruped_walk/quadruped_race.tscn",
	"res://examples/quadruped_walk/hexapod_walk_track.tscn",
	"res://examples/rover_3d/rover_3d.tscn",
	"res://examples/fly_by/fly_by.tscn",
]

func _count(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == ORBIT:
		n += 1
	for c in node.get_children():
		n += _count(c)
	return n

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var root := packed.instantiate()
		h.assert_eq(_count(root), 1, "%s has exactly one OrbitCamera" % path)
		root.free()
	h.finish(self)
