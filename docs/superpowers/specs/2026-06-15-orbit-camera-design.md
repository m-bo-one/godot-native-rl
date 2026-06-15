# Orbit camera for the 3D demos — design (#265)

## Purpose

The 3D demos currently use fixed follow cameras, so a viewer is locked to one angle. A
**user-controlled orbit camera** lets them rotate, zoom, and inspect the trained behavior from any
angle — see the quadruped's gait from the side, confirm it goes *over* the hurdles (not beside),
watch the rover approach its goal. This is a demo/DX polish item under the #229 umbrella.

## Scope

**In scope:** a reusable `OrbitCamera` node + wiring it into the existing single-creature 3D demos
(quadruped walk / hurdles / race, rover_3d, fly_by).

**Explicitly out of scope (deferred / not wanted):**
- The **multi-creature showcase scene** (N trained creatures tiled far apart, orbit overhead) — a
  separate follow-up spec. It's the compelling application but a bigger piece (new scene + Jolt
  tiling), and it builds *on* the node this spec delivers.
- A **detached free-fly** mode (fly anywhere independent of the creature) — YAGNI for single-creature
  inspection; it's mainly useful for the showcase, so it ships with that follow-up if ever.
- The **2D demos** (chase, coop, gridworld, ball_chase, hide_and_seek, visual_chase) — fixed
  top-down is already clear; orbit does not apply.

## Architecture

A single reusable node, `OrbitCamera` (`extends Camera3D`), living in the addon
(`addons/godot_native_rl/...`) so game devs using the addon get it too. It supersedes
`examples/quadruped_walk/quadruped_camera.gd` on the demos it's wired into. It has **two modes**:

- **follow** (default): azimuth/elevation/distance held at configurable defaults (≈ the current
  behind-and-above framing), pivot tracks the target each frame. Identical feel to today's follow
  cam, so default demo framing is unchanged.
- **orbit**: the user drives azimuth/elevation (drag) and distance (scroll); the pivot still tracks
  the moving creature, so it stays framed while you rotate around it.

A key toggles follow ↔ orbit.

### Camera math (pure, testable)

State is spherical around the pivot: `azimuth` (yaw, radians), `elevation` (pitch, clamped to
~±80° so it never flips over the pole), `distance` (clamped `[min_distance, max_distance]`). A pure
static helper computes the world position:

```
orbit_position(pivot: Vector3, azimuth, elevation, distance) -> Vector3
  = pivot + distance * Vector3(
        cos(elevation) * sin(azimuth),
        sin(elevation),
        cos(elevation) * cos(azimuth))
```

Each frame: `global_position = orbit_position(pivot, az, el, dist)`, then `look_at(pivot, UP)`.
Mirrors the existing `fly_by_camera.gd` `follow_position` pattern (pure helper + thin node loop).

### Pivot resolution (uniform across demos)

The camera reads a duck-typed `get_camera_pivot() -> Vector3` from its game node (resolved via the
existing `get_parent()` / optional `game_path` convention). Each game gains a one-liner:
- `QuadrupedGame.get_camera_pivot()` → `torso_pos()`
- `RoverGame.get_camera_pivot()` → its body's position
- `FlyByGame.get_camera_pivot()` → the plane transform origin

If the method is absent the camera holds still (and warns once) — same defensive style as the
existing cameras (which no-op when the target method is missing).

## Controls

Handled in `_unhandled_input` so the camera never eats UI / policy-overlay / launcher keys.

- **Rotate (orbit):** right-mouse-button drag — horizontal Δ → azimuth, vertical Δ → elevation
  (clamped). `orbit_sensitivity` export.
- **Zoom:** mouse wheel up/down → `distance` ∓ step, clamped.
- **Mode toggle:** `toggle_key` export, default **C** (F3 = policy-debug overlay, Esc = launcher
  nav are taken; C is free + intuitive). Flips follow ↔ orbit.
- **Convenience:** an RMB-drag or a scroll while in follow mode auto-switches to orbit, so the user
  doesn't have to press C first.

## Headless / inertness

Headless runs receive no input events, so the camera stays in **follow** mode and behaves exactly
like the current follow cam — purely cosmetic, zero effect on observations / rewards / training /
CI. It never reads or writes game state beyond the pivot position.

## Per-demo integration

- **quadruped_walk / quadruped_hurdles / quadruped_race / hexapod_walk** (every track scene that
  currently uses `quadruped_camera.gd`, since `OrbitCamera` supersedes it): the Camera3D node's
  script becomes `OrbitCamera`; follow-mode defaults reproduce the current offset (≈ behind-and-
  above), so framing is unchanged and orbit just adds manual control. The race model-swaps one
  creature at a time; the pivot tracks whichever is active.
- **rover_3d:** Camera3D script → `OrbitCamera`, pivot = rover body. Rover heads to a goal (not a
  straight line); orbit-follow tracks it fine.
- **fly_by:** the plane *turns*, and `fly_by_camera.gd` trails its *heading* (flattened) so you
  always see its back against the sky — better than a fixed-angle follow for a turning plane. So
  fly_by **keeps** `fly_by_camera.gd` as its default camera and **adds** an `OrbitCamera` as a
  second camera that the toggle key makes `current`. (Approach: unify on the quadrupeds + rover;
  coexist on fly_by.)

## Testing

- **Unit (headless):** the pure `orbit_position()` helper — known spherical→cartesian positions, the
  elevation/distance clamps, and that the camera ends up `distance` from the pivot looking at it.
- **Scene-structure regression:** each target demo scene loads and carries exactly one camera with
  the `OrbitCamera` script (placement-independent, like `test_overlay_in_examples.gd`).
- **No-regression:** the existing quadruped / rover golden + behavioral regressions must stay green
  unchanged (default follow framing preserved; camera never touches game state).

## Risks / notes

- **Default-angle parity:** follow-mode defaults must visually match the current cams or the demos'
  framing changes. Mitigation: convert the existing fixed offset (e.g. `Vector3(4, 3.5, -9)`) to the
  equivalent spherical az/el/dist and use it as the default; eyeball the result.
- **fly_by two-camera wiring:** ensure exactly one camera is `current` at a time and the toggle
  swaps cleanly; the scene-structure test still expects one `OrbitCamera` present (not necessarily
  current).
- This is cosmetic only — no protocol, training, or deploy-contract impact.
