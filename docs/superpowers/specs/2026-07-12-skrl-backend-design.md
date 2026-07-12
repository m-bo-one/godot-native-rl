# SKRL training backend (#25, backlog item 19)

**Status:** approved-by-implementer (autonomous)
**Issue:** [#25](https://github.com/minigraphx/godot-native-rl/issues/25)
**Date:** 2026-07-12

## Problem

Backlog item 19: a SKRL training backend — the fifth stock trainer alongside SB3, CleanRL,
SampleFactory and RLlib, proving an unmodified skrl release trains over the godot_rl wire protocol
and deploys through the TorchScript→ncnn contract.

## Design

- **`scripts/train_skrl.py`** — mirrors `train_rllib.py` conventions (config NamedTuple, pure
  helpers, lazy heavy imports). Reuses the **same single-agent gymnasium adapter** the RLlib
  backend built (`train_rllib.make_godot_env_cls`: `Dict({'obs': Box}) -> Box`,
  `Tuple(Discrete) -> Discrete`, batch-of-one squeezed), wrapped by skrl's own `wrap_env`.
  Models are user-authored in skrl (`CategoricalMixin` policy + `DeterministicMixin` value over
  plain `nn.Sequential` trunks we own), so the deploy export is trivial and version-decoupled:
  the trained policy trunk (obs → raw logits) is traced to TorchScript + shape sidecar inline —
  no checkpoint introspection, unlike the RLlib exporter.
- **`requirements-skrl.txt`** — optional add-on for `.venv-train` (`skrl==2.1.*`), exactly the
  ray-add-on pattern (#126): base stack satisfies skrl's deps; CI omits it; `setup_training.sh`
  installs it locally.
- **`scripts/train_skrl.sh`** — trainer first (server), short `STARTUP_DELAY`, headless chase
  scene, wait, then `export_to_ncnn.py` (pnnx + parity).
- **Guarded smoke** in `run_tests.sh` on `import skrl` (like the ray smokes).

## skrl 2.1 API findings (introspected live — newer than any doc I trusted)

- v1's `PPO_DEFAULT_CONFIG` dict is **gone**; config is the `PPO_CFG` dataclass.
- `Model.compute` is **called positionally** (`self.compute(inputs, role)`) although the base
  class annotates `role` as keyword-only — subclasses must define `compute(self, inputs, role="")`.
- `wrap_env` auto-detects a plain gymnasium env; `SequentialTrainer` cfg is a plain dict
  (`{"timesteps", "headless", "disable_progressbar", "close_environment_at_exit"}`).
- Whole loop proven in-process on CartPole before wiring Godot (train + trace OK).

## Verification

- `test/python/test_train_skrl.py` — pure helpers (args, PPO_CFG kwargs mapping) + torch-guarded
  net-shape tests.
- Live end-to-end smoke (tiny run): chase scene → skrl PPO → TorchScript → ncnn parity — run
  locally before merge; guarded in CI.

## Scope note

The original backlog line said "multi-agent + JAX". Shipped as the **torch single-agent interop
proof**, matching the scope every other stock backend shipped with (CleanRL/SF/RLlib all started
single-agent chase). skrl's JAX flavor and its multi-agent trainers (IPPO/MAPPO) remain reachable
through the same adapter shapes if demand appears; the PettingZoo adapter (#111) is the natural
seam for skrl multi-agent later.
