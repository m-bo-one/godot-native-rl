# Quadruped: jumping gait over SOLID hurdles (#286)

**Status:** approved-by-implementer (autonomous)
**Issue:** [#286](https://github.com/minigraphx/godot-native-rl/issues/286) (follow-up to #277)
**Date:** 2026-07-09

## Problem

The shipped hurdles net (#60 M2) clears hurdles that are **perception-only** — `StaticBody3D` on
collision layer 2 with `collision_mask = 0`, so the closeness RaycastSensor reads them but the
creature never physically collides; #277 was a cosmetic visual fix. #286 asks for the real thing:
**physically-solid hurdles** the quadruped must actually **leap over**, and a retrained **jumping
gait**.

## Feasibility / approach

Training a leap from scratch is very hard, but we can **warm-start from `models/quadruped_hurdles.zip`**
— the committed SB3 checkpoint of the current hurdles runner (same 35-dim obs), which already runs the
course and approaches hurdles. So the fine-tune only has to learn the **launch**, not locomotion. A
**low→tall solid curriculum** (start with steppable bumps, ramp height) plus **jump-launch reward
shaping** gives it a gradient to discover the vertical push. Obs is UNCHANGED (35-dim: 29 base + 6
closeness rays) so the warm-start loads directly and the wire contract is stable.

## Design (all additions are backward-compatible; the shipped perception-only demo is untouched)

### 1. Solid hurdles — `HurdleTrack.solid` (default `false`)
- New `@export var solid := false`. When true, `_make_hurdle` sets `body.collision_layer = 3`
  (bit 1 → the creature's default `collision_mask = 1` collides; bit 2 → the RaycastSensor's
  `collision_mask = 2` still reads it) and renders a **solid visible wall at the true collision
  height** (instead of the #277 decoupled low gate). Default `false` = the exact current
  perception-only behavior, so `quadruped_hurdles_*` scenes and the shipped net are byte-identical.
- New `dist_to_next_hurdle(torso_z) -> float` — signed distance to the next un-passed hurdle ahead
  (large sentinel if none), for the jump-timing reward. Pure over `zs()` + `_next_index`.

### 2. Jump reward — `QuadrupedHurdlesAgent`, gated by `jump_weight` (default `0.0`)
- `@export var jump_weight := 0.0` (0 → shipped hurdles net unchanged), `jump_zone := 2.0`
  (metres ahead of a hurdle where an upward launch is rewarded), `jump_vy_cap := 3.0`.
- In the loop: `reward += jump_weight * JumpMath.approach_jump_reward(vy, dist, jump_zone, jump_vy_cap)`
  where `vy = _game.vertical_velocity()` and `dist = _track.dist_to_next_hurdle(torso_local.z)`.
  Rewards upward torso velocity ONLY when a hurdle is close ahead — nudging the creature to push up
  at the wall, not bounce randomly. The solid wall + the (raised) clear bonus do the rest: you can't
  get the forward/clear reward without going over.
- `QuadrupedGame.vertical_velocity() -> float` = `torso.linear_velocity.y` (mirrors
  `forward_velocity`/`lateral_velocity`).

### 3. `JumpMath` (pure, unit-tested)
`approach_jump_reward(vy, dist, zone, cap)` = `clampf(vy, 0, cap)` when `0 <= dist <= zone`, else `0`.

### 4. Curriculum — `quadruped_jump_curriculum.json` (solid, low→tall)
Stages promote on **`success_rate`** (fraction of episodes clearing ≥1 hurdle — the right signal when
a solid wall gates forward reward): flat → tiny (0.08) → low (0.15) → med (0.22) → tall (0.30, solid).

### 5. Scenes
- `quadruped_jump_world.tscn` — the hurdles world with `Hurdles.solid = true`, `Agent.jump_weight > 0`,
  a more forgiving `fall_height` (crouched landings aren't a fall), curriculum → the jump json.
- `quadruped_jump_train_parallel.tscn` — `ParallelArena` × 8 + `Sync` (mirrors the hurdles one).
- `quadruped_jump_track.tscn` — deploy scene (trained net + a behavioral checker) for validation +
  the screenshots.

## Training

`OUT=models/quadruped_jump SCENE=res://examples/quadruped_walk/quadruped_jump_train_parallel.tscn
INIT_FROM=models/quadruped_hurdles.zip ENT_COEF=0.01 TIMESTEPS=<budget> ./scripts/train_quadruped.sh`
→ TorchScript → `export_to_ncnn.py --atol 0.05`. Deploy net + behavioral regression committed.

## Verification

- `test/unit/test_jump_math.gd` — the pure jump-reward gate.
- Headless: obs size still 35 with jump gating on; solid collision layer = 3.
- **Screenshots** of the trained gait clearing solid hurdles in `quadruped_jump_track.tscn` (rendered
  under xvfb), sent to the user + used for the honest result writeup.

## Honest expectation

This is a hard Jolt articulated-locomotion task. The deliverable is: solid-hurdle infrastructure +
jump shaping/curriculum (correct, tested), a real training run warm-started from the runner, and an
**honest, screenshot-backed** report of the achieved gait — whatever height it reliably clears — not a
guaranteed 0.5 m leap. If it plateaus, the env + finding are still the #286 outcome.

Closes #286 (or reports the achieved milestone + keeps a follow-up if the tall-hurdle target isn't met).

## Result (2026-07-09, 1.5M steps, warm-started from the hurdles runner)

The curriculum raced flat→tiny→low→med→tall by ~278k steps (all 8 worlds), confirming it clears
**solid** hurdles up through med (0.22 m). The trained net learned a real — if **wild, lunging** —
leap:

| Solid hurdle height | Best reach | Hurdles cleared (of 4) |
|---|---|---|
| 0.20 m | 17.0 m | **2** (reliable, identical across 3 runs) |
| 0.25 m | 15.3 m | 1 |
| 0.30 m | 8.5 m | **1** (a genuine leap over a torso-height solid wall) |

Screenshots (rendered under xvfb) show the arc: crouch at the wall → explosive launch → airborne
apex (~1.5 m) clearing the 0.30 m solid barrier. So the milestone — **solid hurdles + a genuine
jumping gait** (vs the old pass-through) — is met at the low end of the 0.3–0.5 m range for 1–2
hurdles. It does **not** yet clear the full solid course or 0.5 m, and the gait is an ungainly dive
that lands hard (hence the drop-off after 1–2). Committed: the net (`quadruped_jump.ncnn.*`), the
0.30 m deploy demo, and a robust 0.20 m behavioral regression (2 clears / 17 m).

**Kept open on #286:** reliable multi-hurdle 0.3–0.5 m leaping with a cleaner landing gait (needs
more training / reward work on landing stability + a stricter clear gate). This pass ships the env +
shaping + a real first-milestone net.
