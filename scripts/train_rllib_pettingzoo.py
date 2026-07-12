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

Policy identity: the env writes agent_policies.json (agent_id -> policy_name from the
wire) at construction; policy_mapping_fn lazily READS THE FILE and fails loud on an
unknown agent. A file — not a module global — because ray's register_env/config
cloudpickles __main__ functions BY VALUE: the env creator and the mapping fn each get a
private COPY of any module-level dict, so a registry the env fills is invisible to the
mapping fn (found live; see docs/dev/gotchas.md). --policies declares the module set up
front (RLlib needs the ids before the env exists); the env asserts the wire names are a
subset, so a scene/CLI mismatch fails loud instead of training a mislabeled policy.

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

def mapping_from_names(agent_policy_names: Sequence[str]) -> dict:
    """agent_id (adapter index) -> policy name, from the wire's agent_policy_names."""
    if not agent_policy_names:
        raise ValueError("agent_policy_names is empty — the scene emitted no agents")
    return {i: str(name) for i, name in enumerate(agent_policy_names)}


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


def write_agent_policies(path, mapping: dict) -> None:
    """Persist the wire's agent_id -> policy_name mapping (the env -> policy_mapping_fn channel;
    a FILE because cloudpickle-by-value makes module globals private per pickled function)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({str(k): v for k, v in mapping.items()}) + "\n")


def read_agent_policies(path) -> dict:
    """Inverse of write_agent_policies (JSON keys back to the adapter's int agent ids)."""
    return {int(k): v for k, v in json.loads(Path(path).read_text()).items()}


def make_policy_mapping_fn(registry_path):
    """RLlib policy_mapping_fn over the agent_policies.json the env writes at construction.
    Lazily (re-)reads the file — it doesn't exist until the env has done the wire handshake —
    and fails loud on a missing file or unknown agent."""
    cache: dict = {}

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id not in cache:
            try:
                cache.update(read_agent_policies(registry_path))
            except OSError as e:
                raise RuntimeError(
                    f"agent-policy registry {registry_path} unreadable ({e}) — the env writes it "
                    "at construction; was the env created with the same registry_path?") from e
        if agent_id not in cache:
            raise RuntimeError(f"agent {agent_id!r} missing from {registry_path} (have {cache})")
        return cache[agent_id]

    return policy_mapping_fn


def build_env_meta(obs_dim: int, nvec: Sequence[int], policies: Sequence[str]) -> dict:
    """The handshake facts the export step needs (written as env_meta.json, read by the .sh),
    so per-policy export never hardcodes dimensions."""
    return {"obs_dim": int(obs_dim), "nvec": [int(n) for n in nvec], "policies": list(policies)}


def write_env_meta(path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2) + "\n")


def ppo_config_overrides(cfg: RLlibPZConfig) -> dict:
    """Same modest new-API-stack knobs as the #110 single-policy backend."""
    return {
        "framework": "torch",
        "num_env_runners": 0,
        "seed": cfg.seed,
        "train_batch_size": 512,
        "minibatch_size": 128,
        "num_epochs": 4,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "entropy_coeff": 0.01,
    }


# --- Env wrapper (lazy factory, mirrors train_rllib.make_godot_env_cls) ---

def make_squeezed_env_cls(declared_policies: Sequence[str]):
    """Build SqueezedGodotParallelEnv (lazy: imports pettingzoo/numpy/godot adapter here)."""
    import numpy as np
    from pettingzoo import ParallelEnv

    from godot_pettingzoo_env import GodotParallelEnv

    class SqueezedGodotParallelEnv(ParallelEnv):
        """GodotParallelEnv with per-agent Dict obs -> Box and Tuple(Discrete) -> Discrete,
        the shapes RLlib's default RLModule consumes. Writes agent_policies.json (for the
        policy_mapping_fn) and env_meta.json (for the per-policy export step) into
        config['run_dir'] — the env is the only place the wire handshake facts exist."""

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
            self.possible_agents = list(self._inner.possible_agents)
            self.agents = self.possible_agents[:]
            self.observation_spaces = {
                a: squeeze_obs_space(self._inner.observation_spaces[a]) for a in self.possible_agents}
            self.action_spaces = {
                a: squeeze_action_space(self._inner.action_spaces[a]) for a in self.possible_agents}
            self._np = np
            run_dir = cfg["run_dir"]
            write_agent_policies(os.path.join(run_dir, "agent_policies.json"),
                                 mapping_from_names(self._inner.agent_policy_names))
            first = self.possible_agents[0]
            meta = build_env_meta(int(self.observation_spaces[first].shape[0]),
                                  [int(self.action_spaces[first].n)], declared_policies)
            write_env_meta(os.path.join(run_dir, "env_meta.json"), meta)
            print("env_meta:", meta)

        def observation_space(self, agent):
            return self.observation_spaces[agent]

        def action_space(self, agent):
            return self.action_spaces[agent]

        def reset(self, seed=None, options=None):
            obs, infos = self._inner.reset(seed=seed, options=options)
            self.agents = self.possible_agents[:]
            return {a: self._squeeze_obs(o) for a, o in obs.items()}, infos

        def step(self, actions):
            # Scalar Discrete -> the one-component action row the inner env scatters.
            nested = {a: self._np.asarray([int(v)], dtype=self._np.int64) for a, v in actions.items()}
            obs, rewards, terminations, truncations, infos = self._inner.step(nested)
            return ({a: self._squeeze_obs(o) for a, o in obs.items()},
                    rewards, terminations, truncations, infos)

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
    registry_path = os.path.join(exp_dir, "agent_policies.json")
    env_config = {"base_port": cfg.base_port, "speedup": cfg.speedup,
                  "action_repeat": cfg.action_repeat, "seed": cfg.seed,
                  "run_dir": exp_dir}
    register_env("godot_pettingzoo", lambda c: ParallelPettingZooEnv(env_cls(c)))
    policy_mapping_fn = make_policy_mapping_fn(registry_path)

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
    # (env_meta.json + agent_policies.json were written by the env at construction.)

    algo.stop()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
