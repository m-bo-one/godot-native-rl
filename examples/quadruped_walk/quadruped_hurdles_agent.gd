class_name QuadrupedHurdlesAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://examples/quadruped_walk/quadruped_agent.gd"

# #60 M2: the M1 quadruped + forward hurdle perception and a clear-the-hurdle bonus.
# Obs = M1's 29 + 6 closeness rays (RaycastSensor3D, collision_mask = hurdle layer 2) = 35.
# The sensor is a world-scene node (the rig is code-built, so it can't be a torso child in the
# scene): each frame we snap its position to the torso and keep its fixed, level orientation
# (scene sets rotation.y = PI so the ray fan's local -Z looks down +Z, the running direction).

const RAY_OBS_SIZE := 6

@export var hurdle_track_path: NodePath
@export var ray_sensor_path: NodePath
@export var curriculum_path: NodePath  ## this world's CurriculumController (per-world under ParallelArena)
@export var clear_bonus := 1.0
@export var sensor_height := 0.5  ## ray origin Y above the ground (level, torso-independent)
# Jump-launch shaping for the SOLID-hurdle task (#286), gated OFF by default so the shipped
# perception-only hurdles net is byte-identical. jump_weight>0 rewards upward torso velocity while a
# hurdle is within jump_zone metres ahead, nudging the creature to launch AT the wall it must clear.
const JumpMath = preload("res://examples/quadruped_walk/jump_math.gd")
@export var jump_weight := 0.0
@export var jump_zone := 2.0     ## metres ahead of a hurdle where an upward launch is rewarded
@export var jump_vy_cap := 3.0   ## clamp the per-step launch contribution
# Anti-bypass (#252): instead of a hard out-of-lane terminal (which ended episodes before the policy
# could learn to recover -> collapsed to a degenerate out-of-lane policy), keep the creature in-lane
# with a CONTINUOUS lateral-position penalty: |torso_x| beyond lane_soft costs lane_weight per metre,
# so drifting toward the hurdle's edge to run BESIDE it is never worth it, but the episode keeps going
# and a smooth gradient pulls the creature back to center. Clears are paid only within lane_half_width.
@export var lane_half_width := 2.5
@export var lane_soft := 1.2          ## |torso_x| beyond this starts being penalized
@export var lane_weight := 0.6        ## penalty per metre of lateral excess
@export var lane_excess_cap := 2.0    ## cap the penalized excess (metres) so a transient can't run the reward away
@export var lane_bound := 6.0         ## |torso_x| beyond this ends the episode (wide safety net for blowups)

var _track
var _sensor
var _warned_no_sensor := false
var _curriculum: Node = null
var _episode_reward := 0.0
var _episode_clears := 0

func _ready() -> void:
	super._ready()
	_track = get_node_or_null(hurdle_track_path)
	_sensor = get_node_or_null(ray_sensor_path)
	# Per-world controller via path (each tiled world is self-contained); group fallback for
	# single-world scenes that keep the controller at the top level.
	_curriculum = get_node_or_null(curriculum_path)
	if _curriculum == null and is_inside_tree():
		_curriculum = get_tree().get_first_node_in_group("CURRICULUM")

func get_info() -> Dictionary:
	if _curriculum == null:
		return {}
	return {"curriculum_stage": _curriculum.stage_index()}

func expected_obs_size() -> int:
	return OBS_SIZE + RAY_OBS_SIZE

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(expected_obs_size())
	z.fill(0.0)
	return z

# Fixed-size ray slice: zero-filled without a sensor (one warning), padded/truncated otherwise —
# the wire contract must not drift with scene wiring.
func _ray_obs() -> Array:
	var out: Array = []
	if _sensor == null:
		if not _warned_no_sensor:
			push_warning("QuadrupedHurdlesAgent: ray_sensor_path not set — zero-filled ray observations.")
			_warned_no_sensor = true
	else:
		out = _sensor.get_observation()
	while out.size() < RAY_OBS_SIZE:
		out.append(0.0)
	if out.size() > RAY_OBS_SIZE:
		out.resize(RAY_OBS_SIZE)
	return out

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	var base: Array = super.get_obs()["obs"]
	base.append_array(_ray_obs())
	return {"obs": base}

func _snap_sensor() -> void:
	if _sensor == null or _game == null:
		return
	var p: Vector3 = _game.torso_pos()
	if _sensor.is_inside_tree():
		_sensor.global_position = Vector3(p.x, sensor_height, p.z)
	else:
		_sensor.position = Vector3(p.x, sensor_height, p.z)

# Full override of QuadrupedAgent's loop (GDScript can't skip one super level): same v3
# locomotion reward + hurdle-clear bonus, and the corrected terminal ordering — the fall
# penalty must NOT be zeroed before the sync reads it (#207; reward+done are read together).
func _physics_process(_delta: float) -> void:
	_core.step(reset_after)  # the controller layer's episode bookkeeping
	if _game == null:
		return
	if _action.size() == ACTION_COUNT:
		_game.apply_motors(_action)
	_snap_sensor()
	var reward_before := reward
	accumulate_reward()
	reward += forward_weight * _game.forward_velocity()
	reward -= lateral_weight * absf(_game.lateral_velocity())
	reward += upright_weight * _game.upright()
	reward -= energy_penalty * _sum_abs(_action)
	# Tile-offset-safe LOCAL torso position (global X/Z carry the ParallelArena tile offset, which
	# would constant-max the lane penalty + mis-count clears on every tile but the first — the bug
	# that stalled every hurdles retrain at a flat -12.6). (#252)
	var torso_local: Vector3 = _game.torso_local_pos()
	var torso_x: float = torso_local.x
	# Soft, BOUNDED lane keeping: a capped continuous penalty beyond lane_soft pulls the creature back
	# to center (no tight terminal — that was unlearnable) while the cap stops a transient excursion or
	# physics spike from running the reward away.
	reward -= lane_weight * minf(maxf(0.0, absf(torso_x) - lane_soft), lane_excess_cap)
	# Jump-launch shaping (#286): reward upward torso velocity while a solid hurdle is close ahead, so
	# the creature learns to push up AT the wall. Gated OFF (jump_weight 0) for the perception-only net.
	if jump_weight > 0.0 and _track != null:
		var dist: float = _track.dist_to_next_hurdle(torso_local.z)
		reward += jump_weight * JumpMath.approach_jump_reward(_game.vertical_velocity(), dist, jump_zone, jump_vy_cap)
	# Clears are only paid IN-LANE: the bonus can't be earned by drifting past the hurdle's edge.
	if _track != null and absf(torso_x) <= lane_half_width:
		var cleared: int = _track.count_newly_passed(torso_local.z)
		_episode_clears += cleared
		reward += clear_bonus * cleared
	# Terminal on a fall OR a wide lateral blowout (far past the hurdles) — the wide bound leaves room
	# to drift and recover but still ends a truly off-track/haywire episode.
	if _is_fallen() or absf(torso_x) > lane_bound:
		reward -= fall_penalty
		done = true
		needs_reset = true
	_episode_reward += reward - reward_before
	if needs_reset:
		needs_reset = false
		if _curriculum != null:
			_curriculum.record_episode(_episode_reward, _episode_clears > 0)
		_episode_reward = 0.0
		_episode_clears = 0
		_game.reset_positions()
		if _track != null:
			_track.reset_progress()
		reset()
		# NO zero_reward() here: this step's reward (incl. the fall penalty and any clear
		# bonus) is read by the sync together with done. Zeroing would wipe it (#207).
		if reward_source != null:
			reward_source.reset()
