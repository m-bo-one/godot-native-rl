extends SceneTree
# Unit tests for the #60 M2 hurdles example: HurdleTrack layout/progress/curriculum and the
# hurdles agent's fixed-size obs contract (29 + 6 rays = 35), with and without a sensor.

const Harness = preload("res://test/harness.gd")
const Track = preload("res://examples/quadruped_walk/hurdle_track.gd")
const Game = preload("res://examples/quadruped_walk/quadruped_game.gd")
const Agent = preload("res://examples/quadruped_walk/quadruped_hurdles_agent.gd")
const Sensor = preload("res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd")

func _initialize() -> void:
	var h = Harness.new()

	# --- pure layout ---
	h.assert_eq(Track.hurdle_layout(0, 8.0, 8.0), [], "0 hurdles -> empty layout")
	h.assert_eq(Track.hurdle_layout(-2, 8.0, 8.0), [], "negative count -> empty layout")
	h.assert_eq(Track.hurdle_layout(3, 8.0, 8.0), [8.0, 16.0, 24.0], "3 hurdles spaced 8 from 8")

	# hurdle_height_at (#252): uniform when max <= base; ramps base->max across the count otherwise.
	h.assert_true(absf(Track.hurdle_height_at(0, 6, 0.3, 0.0) - 0.3) < 1e-6, "height uniform when max=0")
	h.assert_true(absf(Track.hurdle_height_at(5, 6, 0.3, 0.2) - 0.3) < 1e-6, "height uniform when max<base")
	h.assert_true(absf(Track.hurdle_height_at(0, 6, 0.3, 0.55) - 0.3) < 1e-6, "first hurdle = base")
	h.assert_true(absf(Track.hurdle_height_at(5, 6, 0.3, 0.55) - 0.55) < 1e-6, "last hurdle = max")
	h.assert_true(absf(Track.hurdle_height_at(1, 6, 0.3, 0.55) - 0.35) < 1e-6, "second hurdle ramps")

	# --- track node: build / progress / curriculum ---
	var track = Track.new()
	track.hurdle_count = 2
	track.hurdle_spacing = 10.0
	track.start_z = 5.0
	get_root().add_child(track)
	track.rebuild()  # _ready doesn't fire from _initialize in headless SceneTree tests
	h.assert_eq(track.get_child_count(), 2, "2 hurdle bodies built")
	h.assert_eq(track.count_newly_passed(0.0), 0, "nothing passed at start")
	h.assert_eq(track.count_newly_passed(6.0), 1, "first hurdle passed")
	h.assert_eq(track.count_newly_passed(6.0), 0, "passed hurdle pays once")
	h.assert_eq(track.count_newly_passed(40.0), 1, "second hurdle passed")
	track.reset_progress()
	h.assert_eq(track.count_newly_passed(40.0), 2, "reset re-arms both")

	# --- solid mode + jump distance (#286) ---
	# Default (perception-only): hurdle bodies are collision_layer 2, mask 0 (byte-identical to #60 M2).
	var pbody: StaticBody3D = track.get_child(0)
	h.assert_eq(pbody.collision_layer, 2, "perception-only hurdle on layer 2")
	h.assert_eq(pbody.collision_mask, 0, "perception-only hurdle mask 0")

	var solid = Track.new()
	solid.solid = true
	solid.hurdle_count = 2
	solid.hurdle_spacing = 10.0
	solid.start_z = 5.0
	get_root().add_child(solid)
	solid.rebuild()
	var sbody: StaticBody3D = solid.get_child(0)
	# Solid: layer 3 = bit 1 (creature's default mask 1 collides) + bit 2 (sensor mask 2 still reads).
	h.assert_eq(sbody.collision_layer, 3, "solid hurdle on layer 3 (creature + sensor)")
	# dist_to_next_hurdle: first hurdle near-face at z = 5 - depth/2 (depth default 0.3 -> 4.85).
	h.assert_true(absf(solid.dist_to_next_hurdle(0.0) - 4.85) < 1e-6, "dist to first hurdle near-face")
	h.assert_true(solid.dist_to_next_hurdle(4.85) <= 0.0, "dist <= 0 at/after the near-face")
	solid.count_newly_passed(6.0)  # pass the first
	h.assert_true(absf(solid.dist_to_next_hurdle(6.0) - 8.85) < 1e-6, "dist to SECOND hurdle after passing first")
	solid.count_newly_passed(40.0)  # pass all
	h.assert_true(solid.dist_to_next_hurdle(40.0) > 1e8, "no hurdles ahead -> large sentinel")
	solid.queue_free()

	track.apply_curriculum({"hurdle_count": 4, "hurdle_height": 0.3, "hurdle_spacing": 6.0})
	h.assert_eq(track.get_child_count(), 4, "curriculum rebuild -> 4 bodies")
	h.assert_eq(track.zs(), [5.0, 11.0, 17.0, 23.0], "curriculum spacing applied")
	var first_body: StaticBody3D = track.get_child(0)
	h.assert_eq(first_body.collision_layer, 2, "hurdles on collision layer 2")
	h.assert_true(absf(first_body.position.y - 0.15) < 0.0001, "body centered at half height")

	track.apply_curriculum({"hurdle_count": 0})
	h.assert_eq(track.get_child_count(), 0, "stage-0 flat builds nothing")
	h.assert_eq(track.count_newly_passed(100.0), 0, "flat track never pays")

	# --- agent obs contract ---
	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)
	var game = Game.new()
	get_root().add_child(game)
	game.set_finish(finish)
	game.build_now()

	var agent = Agent.new()
	get_root().add_child(agent)
	agent.set_game(game)
	h.assert_eq(agent.expected_obs_size(), 35, "obs contract 29 + 6 rays")
	var obs_no_sensor: Array = agent.get_obs()["obs"]
	h.assert_eq(obs_no_sensor.size(), 35, "no sensor -> zero-filled to 35")
	for i in range(29, 35):
		h.assert_eq(float(obs_no_sensor[i]), 0.0, "ray slot %d zero without sensor" % i)

	var sensor = Sensor.new()
	sensor.n_rays_width = 3
	sensor.n_rays_height = 2
	sensor.ray_length = 6.0
	get_root().add_child(sensor)
	sensor.set_cast_fn_for_test(func(_o: Vector3, _d: Vector3) -> float: return 3.0)
	agent._sensor = sensor
	var obs: Array = agent.get_obs()["obs"]
	h.assert_eq(obs.size(), 35, "with sensor -> still 35")
	h.assert_true(absf(float(obs[29]) - 0.5) < 0.0001, "closeness 1 - 3/6 = 0.5 in ray slot")

	h.finish(self)
