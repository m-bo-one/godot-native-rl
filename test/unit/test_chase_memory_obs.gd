extends SceneTree
# Unit tests for the memory-chase pure helpers (#378): blink schedule + blanked obs encoding.

const Harness = preload("res://test/harness.gd")
const MemObs = preload("res://examples/chase_the_target/chase_memory_obs.gd")
const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- visible_for_step: 2 visible / 6 hidden cycle ---
	var pattern: Array = []
	for step in range(10):
		pattern.append(MemObs.visible_for_step(step, 2, 6))
	h.assert_eq(pattern, [true, true, false, false, false, false, false, false, true, true],
		"2v/6h cycle: visible first, wraps at 8")
	h.assert_true(MemObs.visible_for_step(-3, 2, 6), "negative step clamps to 0 (visible)")
	h.assert_true(MemObs.visible_for_step(5, 2, 0), "hidden_steps=0 -> always visible")
	h.assert_true(not MemObs.visible_for_step(0, 0, 6), "visible_steps=0 -> never visible")

	# --- compute_obs: visible phase = ChaseObs encoding + flag 1 ---
	var arena := Vector2(1000, 600)
	var agent := Vector2(100, 100)
	var target := Vector2(700, 400)
	var vis: Array = MemObs.compute_obs(agent, target, arena, true)
	var base: Array = ChaseObs.compute_obs(agent, target, arena)
	h.assert_eq(vis.size(), 6, "obs is 6-dim")
	for i in range(5):
		h.assert_eq(vis[i], base[i], "visible slot %d matches ChaseObs" % i)
	h.assert_eq(vis[5], 1.0, "visible flag = 1")

	# --- compute_obs: hidden phase zeroes target-derived slots, keeps agent pos ---
	var hid: Array = MemObs.compute_obs(agent, target, arena, false)
	h.assert_eq(hid.size(), 6, "hidden obs is 6-dim")
	h.assert_eq(hid[0], base[0], "agent x still observed while hidden")
	h.assert_eq(hid[1], base[1], "agent y still observed while hidden")
	for i in range(2, 6):
		h.assert_eq(hid[i], 0.0, "hidden slot %d zeroed" % i)

	h.finish(self)
