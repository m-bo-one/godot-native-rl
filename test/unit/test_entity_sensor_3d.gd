extends SceneTree
# Unit tests for EntitySensor3D (#46 M1). Detached nodes -> local-transform fallback, headless.

const Harness = preload("res://test/harness.gd")
const Sensor = preload("res://addons/godot_native_rl/sensors/entity_sensor_3d.gd")
const Stub = preload("res://test/unit/stubs/entity_feature_stub_3d.gd")

func _make(pos: Vector3, feats: Array) -> Node3D:
	var s: Node3D = Stub.new()
	s.position = pos
	s.features = feats
	return s

func _initialize() -> void:
	var h = Harness.new()

	# F = (x,y,z); no extras. obs_size = N*3 + N.
	var sensor = Sensor.new()
	sensor.max_entities = 2
	sensor.max_distance = 10.0
	sensor.include_z = true
	sensor.extra_feature_count = 0
	h.assert_eq(sensor.feature_width(), 3, "feature_width = 3 (x,y,z)")
	h.assert_eq(sensor.obs_size(), 8, "obs_size = 2*3 + 2")

	# include_z = false drops the z axis -> 2 position features.
	var sz = Sensor.new()
	sz.include_z = false
	sz.extra_feature_count = 0
	h.assert_eq(sz.feature_width(), 2, "feature_width = 2 when include_z=false")

	# One entity at (10,0,0): scaled (1,0,0); flags [1,0].
	var objs: Array[Node3D] = []
	objs.append(_make(Vector3(10, 0, 0), []))
	sensor.objects_to_observe = objs
	var obs: Array = sensor.get_observation()
	h.assert_eq(obs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "one entity offset + flags")

	# Extra features appended after the position block.
	var s2 = Sensor.new()
	s2.max_entities = 1
	s2.max_distance = 10.0
	s2.include_z = true
	s2.extra_feature_count = 1
	h.assert_eq(s2.feature_width(), 4, "feature_width = 3 pos + 1 extra")
	var objs2: Array[Node3D] = []
	objs2.append(_make(Vector3(0, 10, 0), [0.5]))
	s2.objects_to_observe = objs2
	var obs2: Array = s2.get_observation()
	# offset (0,10,0) -> scaled (0,1,0); extra 0.5; flag [1].
	h.assert_eq(obs2, [0.0, 1.0, 0.0, 0.5, 1.0], "3D offset + extra + flag")

	h.finish(self)
