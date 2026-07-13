# NumPy env twin with raycast obs — analytic ray-vs-AABB (#364)

**Status:** approved-by-implementer (autonomous)
**Issue:** [#364](https://github.com/minigraphx/godot-native-rl/issues/364) (follow-up to #37)
**Date:** 2026-07-13

## Problem

#37 proved the "twin" approach (train in pure NumPy, no Godot/socket, deploy back via ncnn) for a
*kinematic* env. Most non-trivial envs use `RaycastSensor` observations — the approach only
generalizes if the ray math can be reproduced in NumPy exactly enough to preserve sim-to-deploy
transfer. The original backlog item 31 explicitly listed "kinematics + **analytic raycast-vs-AABB**
+ reward"; this issue proves that missing axis.

## Design — `chase_rays`: chase + solid AABB obstacles + a surround RaycastSensor2D

Per the issue's "add rays to chase" option — chase already has a proven twin, so the delta is
exactly the ray axis:

- **Geometry:** 4 fixed 100×100 AABB obstacles in the 1000×600 arena (centers (300,200), (700,200),
  (300,400), (700,400)) — deterministic, leaves corridors.
- **Blocking (new dynamics):** the agent is blocked by the boxes. CODE-side AABB resolution (pure
  `resolve_aabb`: push out along the minimal-penetration axis, fixed obstacle order), mirrored
  exactly in NumPy — like chase's kinematics, NOT engine physics, so the twin can be exact. The
  target relocates on catch, rejection-sampled outside the boxes (an in-box target could be
  uncatchable). Spawns likewise.
- **Rays (new obs):** the REAL `RaycastSensor2D` node (8 rays, `cone_degrees=315` → 45° apart
  surround, `ray_length=300`, closeness encoding) casting against StaticBody2D boxes on layer 1.
  The physics bodies exist only for the sensor; motion blocking stays code-side.
- **Obs:** `ChaseObs` 5 floats + 8 closenesses = **13**. Action space unchanged (5 discrete).

## The parity contract (transfer-critical)

A committed golden fixture `test/fixtures/chase_rays_golden_obs.json` — K agent positions → the 8
expected closenesses, computed from the ANALYTIC slab-method ray-vs-AABB:

- Python (`test_chase_rays_twin_env.py`): `ray_closenesses(agent)` must reproduce the fixture
  exactly (it generated it).
- GDScript (`test_chase_rays.gd`): the scene's REAL physics `RaycastSensor2D` obs at the same
  positions must match the fixture within 2e-3 (Godot's ray-vs-box is analytic; the tolerance
  covers solver epsilons). This pins engine-physics == NumPy-analytic, the finding the issue asks
  for.
- Blocking parity: `resolve_aabb` unit-tested with identical cases on both sides.

## Files

- `scripts/chase_rays_twin_env.py` — constants, `ray_aabb_distance` (slab), `ray_closenesses`,
  `resolve_aabb`, `ChaseRaysTwinEnv` (reuses chase_twin_env's obs/action/reward math).
- `scripts/train_chase_rays_twin.{py,sh}` — SubprocVecEnv PPO → TorchScript → ncnn (mirrors #37).
- `examples/chase_the_target/chase_rays_game.gd` (extends chase_game: obstacles, blocking,
  rejection spawn, draw), `chase_rays_agent.gd` (extends chase_agent: obs = base 5 + sensor),
  `chase_rays.tscn` (watchable play scene + legend).
- `test/integration/trained_chase_rays_scene.tscn` — behavioral regression (reuses the chase
  checker; catches ≥ threshold in the REAL engine with REAL physics rays).
- Committed net `examples/chase_the_target/models/chase_rays_twin.ncnn.{param,bin}`.

## Honest limit (documented on the issue)

The twin matches only the analytic cases it reimplements — arbitrary Godot collision shapes /
Jolt physics envs remain out of scope (no analytic form; see #60).
