#!/usr/bin/env python3
"""RLlib MULTI-POLICY PPO over the PettingZoo GodotParallelEnv adapter (#123).

The canonical-upstream multi-agent combination: `ray.rllib.env.wrappers.pettingzoo_env.
ParallelPettingZooEnv` wraps our `GodotParallelEnv` (#111), and RLlib's multi-agent PPO
(new API stack) trains one policy module per `agent_policy_names` entry — proving the
adapter against stock upstream tooling rather than our custom trainer (#118). Runs in
.venv-train with the optional ray add-on (requirements-rllib.txt, #126).

Space squeeze: RLlib's default RLModule wants Box obs / Discrete actions, but the adapter
exposes the raw handshake spaces (Dict({'obs': Box}) / Tuple(Discrete)). A thin PettingZoo
wrapper (SqueezedGodotParallelEnv, built by make_squeezed_env_cls) squeezes both per agent.

Policy identity travels IN THE AGENT ID (the canonical PettingZoo convention): the
squeezed env renames the adapter's integer agents to "<policy>_<index>" from the wire's
agent_policy_names, so policy_mapping_fn is a pure string parse — stateless, and immune
to the cloudpickle-by-value trap a module-global registry hits (ray pickles __main__
functions with private COPIES of their globals; found live, see docs/dev/gotchas.md).
--policies declares the module set up front (RLlib needs the ids before the env exists);
the env asserts the wire names are a subset, so a scene/CLI mismatch fails loud instead
of training a mislabeled policy. env_meta.json (obs_dim/nvec/policies) is still written
at construction for the per-policy export step.

Run this FIRST (the env opens the server on --base_port and waits), THEN launch the Godot
multi-policy scene (its Sync sets multi_policy=true, #73). See train_rllib_pettingzoo.sh.

Convention: heavy imports (ray / torch / numpy / gymnasium / pettingzoo / godot_rl) are
LAZY so the pure helpers stay unit-testable without those deps (test_train_rllib_pettingzoo.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple, Sequence

class RLlibPZConfig(NamedTuple):
    """Immutable run configuration (built from argv by parse_args)."""

    timesteps: int
    base_port: int
    speedup: int
    action_repeat: int
    seed: int
    experiment: str
    train_dir: str
    policies: tuple


def parse_args(argv: Sequence[str] | None = None) -> RLlibPZConfig:
    p = argparse.ArgumentParser(allow_abbrev=False,
                                description="Ray/RLlib multi-policy PPO via PettingZoo (hide & seek).")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", type=str, default="hide_seek_rllib")
    p.add_argument("--train_dir", type=str, default="logs/rllib")
    p.add_argument("--policies", type=str, nargs="+", default=["seeker", "hider"],
                   help="expected policy names (must cover the scene's agent_policy_names)")
    a = p.parse_args(argv)
    return RLlibPZConfig(
        timesteps=a.timesteps, base_port=a.base_port, speedup=a.speedup,
        action_repeat=a.action_repeat, seed=a.seed, experiment=a.experiment,
        train_dir=a.train_dir, policies=tuple(a.policies),
    )


# --- Pure helpers (no heavy deps) ---

def agent_ids_from_names(agent_policy_names: Sequence[str]) -> list:
    """PettingZoo agent ids carrying policy identity: wire agent i with policy P becomes
    "P_i". The inner adapter's integer index is recoverable (split on the LAST underscore),
    and policy_of() below is the whole policy_mapping_fn."""
    if not agent_policy_names:
        raise ValueError("agent_policy_names is empty — the scene emitted no agents")
    return [f"{name}_{i}" for i, name in enumerate(agent_policy_names)]


def policy_of(agent_id, *args, **kwargs) -> str:
    """The RLlib policy_mapping_fn: "<policy>_<index>" -> "<policy>". Pure and stateless
    (extra args accepted because RLlib passes the episode). Fails loud on a malformed id."""
    name, _, index = str(agent_id).rpartition("_")
    if not name or not index.isdigit():
        raise RuntimeError(f"agent id {agent_id!r} does not carry policy identity "
                           "(expected '<policy>_<index>')")
    return name


def inner_index_of(agent_id) -> int:
    """Inverse of agent_ids_from_names for the adapter's integer index."""
    return int(str(agent_id).rpartition("_")[2])


def validate_policies(agent_policy_names: Sequence[str], declared: Sequence[str]) -> None:
    """Every wire policy name must be a declared module id, or training would silently
    route agents to a policy that doesn't exist. Fails loud naming the offenders."""
    unknown = sorted(set(agent_policy_names) - set(declared))
    if unknown:
        raise ValueError(
            f"scene emitted policy names {unknown} not in --policies {sorted(set(declared))}; "
            "declare them (or fix the scene's policy_group exports)")


def squeeze_obs_space(space):
    """Dict({'obs': Box, ...}) -> the Box under 'obs' (what RLlib's default encoder eats)."""
    import gymnasium as gym

    if not isinstance(space, gym.spaces.Dict) or "obs" not in space.spaces:
        raise ValueError(f"expected a Dict obs space with an 'obs' key, got {space}")
    return space.spaces["obs"]


def squeeze_action_space(space):
    """Single-entry Tuple(Discrete) -> the Discrete. Multi-key / non-discrete fail loud
    (this backend mirrors #110's single-Discrete scope)."""
    import gymnasium as gym

    if not isinstance(space, gym.spaces.Tuple) or len(space.spaces) != 1:
        raise ValueError(f"expected a single-entry Tuple action space, got {space}")
    inner = space.spaces[0]
    if not isinstance(inner, gym.spaces.Discrete):
        raise ValueError(f"expected a Discrete action, got {inner}")
    return inner


def build_env_meta(obs_dim: int, nvec: Sequence[int], policies: Sequence[str]) -> dict:
    """The handshake facts the export step needs (written as env_meta.json, read by the .sh),
    so per-policy export never hardcodes dimensions."""
    return {"obs_dim": int(obs_dim), "nvec": [int(n) for n in nvec], "policies": list(policies)}


def write_env_meta(path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")


# PPO knobs are shared with the #110 single-policy backend — one definition, no drift
# (train_rllib.ppo_config_overrides only reads cfg.seed, which RLlibPZConfig also carries).
from train_rllib import ppo_config_overrides  # noqa: E402


# --- Env wrapper (lazy factory, mirrors train_rllib.make_godot_env_cls) ---

def make_squeezed_env_cls(declared_policies: Sequence[str]):
    """Build SqueezedGodotParallelEnv (lazy: imports pettingzoo/numpy/godot adapter here)."""
    import numpy as np
    from pettingzoo import ParallelEnv

    from godot_pettingzoo_env import GodotParallelEnv

    class SqueezedGodotParallelEnv(ParallelEnv):
        """GodotParallelEnv with per-agent Dict obs -> Box and Tuple(Discrete) -> Discrete
        (the shapes RLlib's default RLModule consumes), and agents renamed to
        "<policy>_<index>" so identity travels in the id (policy_of == the mapping fn).
        Writes env_meta.json (for the per-policy export step) into config['run_dir'] —
        the env is the only place the wire handshake facts exist."""

        metadata = {"render_modes": [], "name": "SqueezedGodotParallelEnv"}

        def __init__(self, config=None):
            import os

            cfg = dict(config or {})
            self._inner = GodotParallelEnv(
                port=int(cfg.get("base_port", 11008)),
                seed=int(cfg.get("seed", 0)),
                config={"action_repeat": int(cfg.get("action_repeat", 8)),
                        "speedup": int(cfg.get("speedup", 8))},
            )
            validate_policies(self._inner.agent_policy_names, declared_policies)
            # Identity-carrying ids, positionally aligned with the inner adapter's int agents.
            self.possible_agents = agent_ids_from_names(self._inner.agent_policy_names)
            self._inner_id = {a: self._inner.possible_agents[i]
                              for i, a in enumerate(self.possible_agents)}
            self.agents = self.possible_agents[:]
            self.observation_spaces = {
                a: squeeze_obs_space(self._inner.observation_spaces[self._inner_id[a]])
                for a in self.possible_agents}
            self.action_spaces = {
                a: squeeze_action_space(self._inner.action_spaces[self._inner_id[a]])
                for a in self.possible_agents}
            self._np = np
            first = self.possible_agents[0]
            meta = build_env_meta(int(self.observation_spaces[first].shape[0]),
                                  [int(self.action_spaces[first].n)], declared_policies)
            write_env_meta(os.path.join(cfg["run_dir"], "env_meta.json"), meta)
            print("env_meta:", meta)

        def observation_space(self, agent):
            return self.observation_spaces[agent]

        def action_space(self, agent):
            return self.action_spaces[agent]

        def reset(self, seed=None, options=None):
            obs, infos = self._inner.reset(seed=seed, options=options)
            self.agents = self.possible_agents[:]
            return ({a: self._squeeze_obs(obs[self._inner_id[a]]) for a in self.possible_agents},
                    {a: infos.get(self._inner_id[a], {}) for a in self.possible_agents})

        def step(self, actions):
            # Scalar Discrete -> the one-component action row the inner env scatters.
            nested = {self._inner_id[a]: self._np.asarray([int(v)], dtype=self._np.int64)
                      for a, v in actions.items()}
            obs, rewards, terminations, truncations, infos = self._inner.step(nested)
            acted = [a for a in self.possible_agents if self._inner_id[a] in obs]
            return ({a: self._squeeze_obs(obs[self._inner_id[a]]) for a in acted},
                    {a: rewards[self._inner_id[a]] for a in acted},
                    {a: terminations[self._inner_id[a]] for a in acted},
                    {a: truncations[self._inner_id[a]] for a in acted},
                    {a: infos.get(self._inner_id[a], {}) for a in acted})

        def close(self):
            self._inner.close()

        def _squeeze_obs(self, o):
            return self._np.asarray(o["obs"], dtype=self._np.float32)

    return SqueezedGodotParallelEnv


def main(argv: Sequence[str] | None = None) -> int:
    import os

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    cfg = parse_args(argv)
    overrides = ppo_config_overrides(cfg)
    env_cls = make_squeezed_env_cls(cfg.policies)

    exp_dir = os.path.abspath(os.path.join(cfg.train_dir, cfg.experiment))
    env_config = {"base_port": cfg.base_port, "speedup": cfg.speedup,
                  "action_repeat": cfg.action_repeat, "seed": cfg.seed,
                  "run_dir": exp_dir}
    register_env("godot_pettingzoo", lambda c: ParallelPettingZooEnv(env_cls(c)))
    policy_mapping_fn = policy_of  # identity travels in the agent id — stateless

    ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=2)
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment("godot_pettingzoo", env_config=env_config)
        .multi_agent(policies=set(cfg.policies), policy_mapping_fn=policy_mapping_fn)
        .env_runners(num_env_runners=overrides["num_env_runners"])
        .framework(overrides["framework"])
        .debugging(seed=overrides["seed"])
        .training(
            lr=overrides["lr"],
            gamma=overrides["gamma"],
            train_batch_size_per_learner=overrides["train_batch_size"],
            minibatch_size=overrides["minibatch_size"],
            num_epochs=overrides["num_epochs"],
            entropy_coeff=overrides["entropy_coeff"],
        )
    )
    algo = config.build_algo()

    sampled = 0
    iteration = 0
    while sampled < cfg.timesteps:
        result = algo.train()
        iteration = int(result.get("training_iteration", iteration + 1))
        env_runner_results = result.get("env_runners", {})
        if "num_env_steps_sampled_lifetime" not in env_runner_results:
            raise RuntimeError(
                "env_runners/num_env_steps_sampled_lifetime missing from RLlib result "
                f"(keys: {sorted(env_runner_results)}); the step-counter key moved — "
                "update train_rllib_pettingzoo.py for this ray version.")
        sampled = int(env_runner_results["num_env_steps_sampled_lifetime"])
        returns = env_runner_results.get("agent_episode_returns_mean",
                                         env_runner_results.get("episode_return_mean", float("nan")))
        print(f"iter {iteration} steps={sampled}/{cfg.timesteps} returns={returns}")

    ckpt_dir = os.path.join(exp_dir, f"checkpoint_{iteration:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    algo.save_to_path(ckpt_dir)
    print("checkpoint:", ckpt_dir)
    # (env_meta.json was written by the env at construction.)

    algo.stop()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
