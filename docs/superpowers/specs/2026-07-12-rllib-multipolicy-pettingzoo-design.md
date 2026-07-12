# RLlib multi-policy PPO via the PettingZoo adapter (#123)

**Status:** approved-by-implementer (autonomous)
**Issue:** [#123](https://github.com/minigraphx/godot-native-rl/issues/123) (deferred sibling of #118; builds on #110 + #111)
**Date:** 2026-07-12

## Problem

Single-policy RLlib over the wire protocol shipped in #110, and `GodotParallelEnv`
(`scripts/godot_pettingzoo_env.py`) is a conformant PettingZoo `ParallelEnv` (#111) already driving
our own multi-policy trainer (#118). The missing canonical-upstream combination: **stock RLlib
multi-agent PPO trained *via* PettingZoo** — `ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv`
over our adapter, one RLlib policy (module) per `agent_policy_names` entry. That proves the adapter
against upstream's own multi-agent tooling, not just our custom loop.

## Design

### 1. `scripts/train_rllib_pettingzoo.py` (new)

Mirrors `train_rllib.py` (config NamedTuple, pure helpers, lazy heavy imports, new API stack,
`num_env_runners=0` = one env/one socket on the driver):

- **Space squeeze (pure factories + helpers):** RLlib's default `RLModule` wants `Box` obs and
  `Discrete` actions, but `GodotParallelEnv` exposes the raw handshake spaces
  (`Dict({'obs': Box})` / `Tuple(Discrete)`). A thin PettingZoo `ParallelEnv` wrapper
  (`SqueezedGodotParallelEnv`, built lazily by `make_squeezed_env_cls()`) squeezes per agent:
  `squeeze_obs_space` (`Dict['obs'] -> Box`), `squeeze_action_space`
  (single-entry `Tuple(Discrete) -> Discrete`, else fail loud), obs values to the `'obs'` float32
  vector, and scalar actions re-nested to the one-component action rows the inner env scatters.
- **Policy identity:** the env fills a module-level `AGENT_POLICY_REGISTRY` (`agent_id ->
  policy_name` from the wire's `agent_policy_names`) at construction; `policy_mapping_fn` reads it
  and **fails loud** on an unknown agent. Safe because `num_env_runners=0` keeps env + mapping fn
  in one process (documented constraint). `--policies` (default `seeker hider`) declares the
  expected module set up front (RLlib needs the module ids before the env exists); the env asserts
  the wire names are a subset — a scene/CLI mismatch fails loud instead of training a mislabeled
  policy.
- **Multi-agent config:** `.multi_agent(policies=set(cfg.policies), policy_mapping_fn=...)` on the
  same new-API-stack PPO knobs as #110.
- **Env meta for export:** after training, writes `<train_dir>/<experiment>/env_meta.json`
  (`obs_dim`, `nvec`, `policies`) from the live handshake so the export step never hardcodes dims.

### 2. `scripts/export_rllib_to_torchscript.py` (extended, backward-compatible)

- New `--module_id` (default `None` = old behavior). Selection extracted into a pure
  `pick_module_id(keys, requested)`: explicit id must exist (error lists the ids); otherwise
  `default_policy` if present, else the single key, else "ambiguous" error (which now names
  `--module_id` as the fix).

### 3. `scripts/train_rllib_pettingzoo.sh` (new)

Mirrors `train_rllib.sh`: trainer first (server), `sleep 20` (ray startup), headless Godot
(`SCENE` default `hide_and_seek_multipolicy_train_parallel.tscn` — the #73 scene-driven identity),
wait, then per policy in `env_meta.json`: RLModule → TorchScript (`--module_id <name>`) →
`export_to_ncnn.py` → `$OUTDIR/hide_seek_rllib_<name>.ncnn.{param,bin}`. `TIMESTEPS`/`SPEEDUP`/
`ACTION_REPEAT`/`BASE_PORT`/`EXPERIMENT`/`TRAIN_DIR`/`OUTDIR`/`SCENE`/`ATOL` overrides.

### 4. Verification

- `test/python/test_train_rllib_pettingzoo.py` — pure helpers with guarded gymnasium imports
  (mapping/registry, space squeezes incl. fail-loud cases, arg parsing, meta round-trip).
- `test/python/test_export_rllib_to_torchscript.py` — `pick_module_id` cases.
- `test/run_tests.sh` — guarded smoke (ray importable in `.venv-train`): tiny multi-policy run,
  asserts BOTH `hide_seek_rllib_{seeker,hider}.ncnn.{param,bin}` land. Skips in CI (no ray),
  runs locally after `setup_training.sh`.
- Live proof before merge: the smoke run end-to-end locally (ray installed on demand).

## Non-goals

- No committed trained fixtures/regression: #118 already gates the multi-policy deploy path with
  live-trained nets; this issue is about proving the **training interop**, mirroring how the #110
  single-policy backend ships as a guarded smoke + golden.
- Old API stack (`RayVectorGodotEnv`) remains out of scope (#110's decision).
