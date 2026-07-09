# JAX/NumPy Gymnasium chase "twin" — train without Godot (#37)

**Status:** approved-by-implementer (autonomous batch)
**Issue:** [#37](https://github.com/minigraphx/godot-native-rl/issues/37) (Backlog item 31)
**Date:** 2026-07-08

## Problem

Training goes through the Godot socket bridge: every env step is a JSON round-trip to a running
engine. For a **simple kinematic** env (chase) the dynamics are trivial arithmetic — the socket +
engine are pure overhead. Backlog item 31 asks for a pure-Python Gymnasium **"twin"** of a simple
example that trains with **no Godot and no socket** (so SB3 vectorizes over many in-process copies at
far higher throughput), then deploys the trained policy **back into Godot** via ncnn — which only
works if the twin reproduces the Godot dynamics closely enough. Validating that **sim-to-deploy gap**
is the whole point.

## Why NumPy, not JAX

The throughput win here is **removing the socket + engine**, not GPU kernels: chase's step is a
handful of float ops, so a plain NumPy env under SB3's `SubprocVecEnv`/`DummyVecEnv` already runs
orders of magnitude more steps/sec than the bridge, with **zero new heavy deps** (numpy is in
`.venv-train`). A JAX `vmap`/`jit` backend would help only at very large batch on GPU and adds a big
dependency — deferred as a noted extension, **tracked as #361**. The issue title is "JAX/NumPy";
NumPy is the lean choice.

## Exact dynamics to replicate (transfer-critical)

Read from `chase_game.gd` / `chase_agent.gd` / `chase_obs.gd` / `run_speed.gd`:

- **Obs (5-dim, must be byte-identical to `ChaseObs.compute_obs`):**
  `[(ax/W-0.5)*2, (ay/H-0.5)*2, dir.x, dir.y, clamp(dist/‖arena‖, 0, 1)]`, `dir = normalize(target-agent)`.
- **Action (Discrete 5):** `0`=noop, `1`=(0,-s), `2`=(0,+s), `3`=(-s,0), `4`=(+s,0)`, `s=move_speed=300`.
- **Cadence:** `run_speed` scales physics-ticks **and** time_scale together, so `_physics_process`
  delta stays **1/60** at any speedup. One env step = `action_repeat=8` sub-frames (train **and**
  deploy use 8), each moving `vel/60` (= 5 px) → **40 px/step**, matching `touch_radius=40`.
- **Per sub-frame** (mirrors `chase_agent._physics_process`): move + clamp to `[0,W]×[0,H]`; add
  progress `(prev_dist-cur_dist)/‖arena‖`; subtract `step_penalty=0.001`; add any pending catch
  bonus; if `cur_dist < 40`: relocate target to a random cell, `+1` catch bonus queued to the **next**
  sub-frame, rebase the progress baseline (matches `ProgressShapingTerm.on_event`).
- **Episode:** truncates at `reset_after=1000` steps (chase never "terminates" — a catch relocates).

## Design

- **`scripts/chase_twin_env.py`** — pure NumPy helpers (`compute_obs`, `action_to_velocity`,
  constants) + `ChaseTwinEnv(gymnasium.Env)` (`Discrete(5)` / `Box((5,))`). The helpers are the
  single source of the Godot-matching math and are unit-tested against hand-computed Godot values.
- **`scripts/train_chase_twin.py`** — SB3 PPO over `make_vec_env(ChaseTwinEnv, n_envs=N)` (no Godot,
  no socket), then export the deterministic actor to ncnn via the shared `export_policy` (#52) +
  `export_to_ncnn.py`. Because obs/action match chase exactly, the `.ncnn.{param,bin}` drops straight
  into the existing chase deploy scenes.
- **`scripts/train_chase_twin.sh`** — orchestrator (pure Python; prints steps/sec).
- **Validation (the sim-to-deploy proof):** the twin-trained net is committed and run through the
  **existing Godot chase behavioral checker** (`trained_chase_scene`-style, `min_catches`) — if it
  catches in the real engine, the twin matched. Wired into `run_tests.sh`.

## Testing

- `test/python/test_chase_twin_env.py` (stdlib+numpy, no Godot): obs matches the Godot formula on
  fixed inputs; action→velocity table; a full step moves exactly 40 px in-axis; catch relocates +
  awards the bonus; episode truncates at 1000; Gymnasium API shapes (reset/step) + `check_env`.
- A committed twin-trained ncnn net + a Godot behavioral regression (catches ≥ N in 1800 frames).

## Files

| File | Change |
|---|---|
| `scripts/chase_twin_env.py` | **new** NumPy Gymnasium twin + pure helpers |
| `scripts/train_chase_twin.py` / `.sh` | **new** Godot-free trainer + orchestrator |
| `test/python/test_chase_twin_env.py` | **new** dynamics + API tests |
| `examples/chase_the_target/models/chase_twin.ncnn.*` | **new** trained deploy net |
| `test/integration/trained_chase_twin_scene.tscn` + checker reuse | **new** sim-to-deploy regression |
| `CLAUDE.md`, `docs/BACKLOG.md` | document + check off item 31 |

Closes #37.
