#!/usr/bin/env python3
"""Train the Chase The Target agent with SKRL PPO over the godot-rl wire protocol (#25).

Fifth stock training backend alongside SB3, CleanRL, SampleFactory and RLlib. Ecosystem
interop: an unmodified skrl release (2.1.x, torch flavor) trains against our env through
the same single-agent gymnasium adapter the RLlib backend uses (train_rllib.make_godot_env_cls
— Dict({'obs': Box}) -> Box, Tuple(Discrete) -> Discrete, batch-of-one squeezed), wrapped by
skrl's own `wrap_env`. Runs in .venv-train with the optional skrl add-on
(requirements-skrl.txt; torch/gymnasium come from the base stack).

Unlike RLlib, WE define the policy/value nets (skrl models are user-authored nn.Modules with
skrl mixins), so the deploy export is trivial and version-decoupled: the trained policy's
`net` (obs -> raw action logits) is traced to TorchScript + a shape sidecar right here —
no checkpoint introspection — then scripts/export_to_ncnn.py converts with parity check.

skrl 2.1 gotchas (introspected live): PPO config is the PPO_CFG dataclass (the v1
PPO_DEFAULT_CONFIG dict is gone), and Model.compute is CALLED POSITIONALLY
(`self.compute(inputs, role)`) even though the base class annotates `role` keyword-only —
define `compute(self, inputs, role="")`.

Run this FIRST (the env opens the server on --base_port and waits), THEN launch the Godot
chase training scene. See scripts/train_skrl.sh for orchestration.

Convention: heavy imports (skrl / torch / gymnasium / godot_rl) are LAZY so the pure
helpers stay unit-testable without those deps (test/python/test_train_skrl.py).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import NamedTuple, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


class SKRLConfig(NamedTuple):
    """Immutable run configuration (built from argv by parse_args)."""

    timesteps: int
    base_port: int
    speedup: int
    action_repeat: int
    seed: int
    rollouts: int
    out: str


def parse_args(argv: Sequence[str] | None = None) -> SKRLConfig:
    p = argparse.ArgumentParser(allow_abbrev=False, description="SKRL PPO for chase (#25).")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollouts", type=int, default=512,
                   help="on-policy rollout length (env steps between PPO updates)")
    p.add_argument("--out", type=str, default="models/chase_skrl_policy.pt")
    a = p.parse_args(argv)
    return SKRLConfig(timesteps=a.timesteps, base_port=a.base_port, speedup=a.speedup,
                      action_repeat=a.action_repeat, seed=a.seed, rollouts=a.rollouts, out=a.out)


def ppo_cfg_kwargs(cfg: SKRLConfig) -> dict:
    """skrl PPO_CFG kwargs (pure: no skrl import). Field names pinned to skrl 2.1's PPO_CFG
    dataclass; magnitudes borrowed from the other chase backends (interop proof, not a
    leaderboard)."""
    return {
        "rollouts": cfg.rollouts,
        "learning_epochs": 4,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 2.5e-4,
        "entropy_loss_scale": 0.01,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_loss_scale": 0.5,
    }


def actor_net(obs_dim: int, n_actions: int):
    """The policy trunk: obs -> raw action logits. A plain nn.Sequential we own end-to-end,
    so this exact module is what gets traced for ncnn deploy (no framework introspection)."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, n_actions))


def value_net(obs_dim: int):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, 1))


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    from export_to_ncnn import write_shape_sidecar
    from train_rllib import make_godot_env_cls  # the shared single-agent gymnasium adapter

    cfg = parse_args(argv)
    set_seed(cfg.seed)

    gym_env_cls = make_godot_env_cls()
    env = wrap_env(gym_env_cls({"base_port": cfg.base_port, "speedup": cfg.speedup,
                                "action_repeat": cfg.action_repeat, "seed": cfg.seed}))
    device = env.device
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    print(f"obs_dim={obs_dim} n_actions={n_actions} device={device}")

    class Policy(CategoricalMixin, Model):
        def __init__(self, observation_space, action_space, device=None):
            Model.__init__(self, observation_space=observation_space,
                           action_space=action_space, device=device)
            CategoricalMixin.__init__(self, unnormalized_log_prob=True)
            self.net = actor_net(obs_dim, n_actions)

        def compute(self, inputs, role=""):  # positional `role` — see module docstring
            return self.net(inputs["observations"]), {}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device=None):
            Model.__init__(self, observation_space=observation_space,
                           action_space=action_space, device=device)
            DeterministicMixin.__init__(self)
            self.net = value_net(obs_dim)

        def compute(self, inputs, role=""):
            return self.net(inputs["observations"]), {}

    policy = Policy(env.observation_space, env.action_space, device)
    value = Value(env.observation_space, env.action_space, device)
    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=env.num_envs, device=device)
    # Experiment logs/checkpoints under the gitignored logs/ (skrl's default is ./runs/ in cwd,
    # which would land tensorboard events + .pt checkpoints inside the repo).
    from skrl.agents.torch.base import ExperimentCfg

    ppo_cfg = PPO_CFG(**ppo_cfg_kwargs(cfg))
    ppo_cfg.experiment = ExperimentCfg(directory="logs/skrl")
    agent = PPO(models={"policy": policy, "value": value}, memory=memory,
                observation_space=env.observation_space, action_space=env.action_space,
                device=device, cfg=ppo_cfg)
    trainer = SequentialTrainer(env=env, agents=agent,
                                cfg={"timesteps": cfg.timesteps, "headless": True,
                                     "disable_progressbar": True,
                                     "close_environment_at_exit": True})
    trainer.train()
    print("training done")

    # Deploy export: trace the raw-logits trunk (deterministic actor = argmax game-side).
    net = policy.net.to("cpu").eval()
    shape = [1, obs_dim]
    with torch.no_grad():
        scripted = torch.jit.trace(net, torch.zeros(*shape, dtype=torch.float32))
    pt_path = pathlib.Path(cfg.out)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, shape)
    print("exported TorchScript to:", pt_path)
    print("wrote shape sidecar:    ", sidecar)
    print("next: export_to_ncnn.py", pt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
