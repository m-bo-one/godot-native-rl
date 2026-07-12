class_name JumpMath
extends RefCounted
# Pure jump-launch reward math for the solid-hurdle jumping task (#286). Kept scene-free so it is
# unit-tested in test/unit/test_jump_math.gd.

# Reward the upward launch ONLY while a solid hurdle is close ahead — this nudges the creature to
# push its torso up AT the wall rather than bounce randomly. `vy` is world-frame torso vertical
# velocity, `dist` is the signed distance to the next un-passed hurdle (positive = ahead), `zone` is
# how far ahead of the hurdle the launch is rewarded, `cap` clamps the per-step contribution. Zero
# when there is no hurdle ahead within the zone (dist < 0 means the hurdle is already behind).
static func approach_jump_reward(vy: float, dist: float, zone: float, cap: float) -> float:
	if dist < 0.0 or dist > zone:
		return 0.0
	return clampf(vy, 0.0, cap)
