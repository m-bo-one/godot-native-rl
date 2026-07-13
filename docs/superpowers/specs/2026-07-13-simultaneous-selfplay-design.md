# Simultaneous two-sided self-play — mid-run cross-freezing (#189)

**Status:** approved-by-implementer (autonomous)
**Issue:** [#189](https://github.com/minigraphx/godot-native-rl/issues/189) (deferred from the #29 league design)
**Date:** 2026-07-13

## Problem

The #29 league trains ONE side per phase against frozen ncnn ghosts, restarting phases to grow the
opponent pool. #189 asks for the simultaneous form: **both learners live in one run** (the
multi-policy trainer), with **each side's snapshot periodically frozen into the opponent pool
mid-run** — classic league bookkeeping without phase restarts.

## Design

The multi-policy trainer (`train_hide_seek_multipolicy.py`, #26/#73) already trains both policies
simultaneously; the delta is exactly the mid-run freezing:

- **`--snapshot_every N`** (env-steps; default 0 = off, existing behavior byte-identical) +
  **`--pool_dir`** (default `models/selfplay_pool` — the #29 layout).
- Pure `snapshot_updates(total_updates, steps_per_update, every)` picks the update indices whose
  cumulative env-steps first cross each multiple of `every` (unit-tested; off for `every <= 0`).
- At each boundary, `freeze_snapshot` exports EVERY live learner's actor →
  TorchScript → `export_to_ncnn` (pnnx + parity) into `pool_dir/<policy>/<policy>_live_uNNNNNN.ncnn.*`
  and registers it in that pool's `pool.json` ELO ledger at the learner rating
  (`selfplay_phase.register_snapshot` — the same artifacts the alternating league writes per
  phase, so `SelfPlayManager` / `pick_opponent` / `rescan_pool()` consume the pool unchanged).
- `train_hide_seek_multipolicy.sh` forwards `SNAPSHOT_EVERY` / `POOL_DIR`.

## What this deliberately does NOT do

Mixing frozen ghosts into the live run (per-episode live-vs-ghost opponent sampling) is out of
scope: toggling an agent between TRAINING and NCNN_INFERENCE mid-run changes the wire's agent
count, which the protocol cannot express today. The pool this feature grows is consumed by the
EXISTING league flows (alternating phases, eval scenes); revisit ghost-mixing if/when the
protocol grows per-episode agent masking.

## Verification

- Pure schedule + naming unit tests (`test_train_hide_seek_multipolicy.py`).
- Live proof (documented in the PR): a short two-sided run with `SNAPSHOT_EVERY` small produces
  ≥2 snapshots per side (ncnn pairs present, `pool.json` ledgers valid, ratings at learner
  default) while training proceeds to the normal final export.

## Result (2026-07-13, live proof)

Short live run (`TIMESTEPS=6144 NUM_STEPS=256`, single 2-agent world, `SNAPSHOT_EVERY=2048`):
snapshots froze at updates 4, 8 and 12 for BOTH sides — `models/selfplay_pool/{seeker,hider}/`
each gained 3 `*_live_u*.ncnn.{param,bin}` pairs (pnnx parity OK per snapshot) + a valid
`pool.json` with 3 members at rating 1200, while the run continued to the standard final export.
No phase restarts.
