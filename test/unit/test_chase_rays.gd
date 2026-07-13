extends SceneTree

# chase_rays (#364): the GDScript half of the twin parity contract.
# 1. resolve_aabb blocking parity — same cases as test_chase_rays_twin_env.py.
# 2. REAL physics rays == the analytic fixture: StaticBody2D boxes + a real RaycastSensor2D must
#    reproduce test/fixtures/chase_rays_golden_obs.json (computed by the NumPy twin's slab method)
#    within 2e-3 — this is the engine-physics-matches-analytic finding the issue asks for.

const Harness = preload("res://test/harness.gd")
const ChaseRaysGame = preload("res://examples/chase_the_target/chase_rays_game.gd")
const RaycastSensor2DScript = preload("res://addons/godot_native_rl/sensors/raycast_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- resolve_aabb parity (mirrors the Python cases exactly) ---
	var box := Rect2(250, 150, 100, 100)
	h.assert_eq(ChaseRaysGame.resolve_aabb(Vector2(260, 190), box), Vector2(250, 190), "push out min axis (x)")
	h.assert_eq(ChaseRaysGame.resolve_aabb(Vector2(300, 245), box), Vector2(300, 250), "push out y when shallower")
	h.assert_eq(ChaseRaysGame.resolve_aabb(Vector2(100, 100), box), Vector2(100, 100), "outside unchanged")

	# Layout constant matches the fixture's geometry source (4 boxes, 100x100, known centers).
	h.assert_eq(ChaseRaysGame.OBSTACLES.size(), 4, "four obstacles")
	h.assert_eq(ChaseRaysGame.OBSTACLES[0], Rect2(250, 150, 100, 100), "obstacle 0 rect")
	h.assert_eq(ChaseRaysGame.OBSTACLES[3], Rect2(650, 350, 100, 100), "obstacle 3 rect")

	# --- Real physics rays vs the committed analytic fixture ---
	var world := Node2D.new()
	get_root().add_child(world)
	for rect in ChaseRaysGame.OBSTACLES:
		var body := StaticBody2D.new()
		body.collision_layer = 1
		body.position = rect.get_center()
		var shape := CollisionShape2D.new()
		var rs := RectangleShape2D.new()
		rs.size = rect.size
		shape.shape = rs
		body.add_child(shape)
		world.add_child(body)
	var sensor = RaycastSensor2DScript.new()
	sensor.n_rays = 8
	sensor.cone_degrees = 315.0
	sensor.ray_length = 300.0
	sensor.collision_mask = 1
	world.add_child(sensor)
	await physics_frame
	await physics_frame

	var fixture: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string("res://test/fixtures/chase_rays_golden_obs.json"))
	var cases: Array = fixture["cases"]
	h.assert_true(cases.size() >= 4, "fixture has cases")
	var worst := 0.0
	for case in cases:
		sensor.global_position = Vector2(case["agent"][0], case["agent"][1])
		await physics_frame
		var obs: Array = sensor.get_observation()
		h.assert_eq(obs.size(), 8, "8 rays at agent %s" % str(case["agent"]))
		for i in range(8):
			var err: float = absf(float(obs[i]) - float(case["rays"][i]))
			worst = maxf(worst, err)
			if err > 2e-3:
				h.assert_true(false, "ray %d at %s: physics %f vs analytic %f" % [
					i, str(case["agent"]), float(obs[i]), float(case["rays"][i])])
	h.assert_true(worst <= 2e-3, "real physics rays match the analytic twin (worst |err| %f)" % worst)

	h.finish(self)
