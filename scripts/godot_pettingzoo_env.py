#!/usr/bin/env python3
"""Expose a Godot env (godot_rl bridge) as a PettingZoo ParallelEnv for multi-policy training.

Provides the FUNCTIONALITY of godot_rl's GDRLPettingZooEnv without importing it: our own adapter so
we own the lifecycle and dependency surface. One agent == one Godot AIController instance; each agent
maps to a policy via `agent_policy_names` (emitted on the wire since 2026-06-03).

Fixed-population semantics (matches godot_rl + upstream GDRLPettingZooEnv): every agent is present
every step; an agent whose episode has finished receives a zero action until all agents are done.
`terminations`/`truncations` are the REAL Gymnasium split since #12: the wire carries an additive
`truncated` field and the env is built over TruncationAwareGodotEnv (godot_env_truncation.py),
which splits it out (stock godot_rl 0.8.2 still sees plain `done` — unchanged).

Design: docs/superpowers/specs/2026-06-09-pettingzoo-multipolicy-interop-design.md

The constructor accepts an injected `godot_env` (for tests / custom construction); when omitted it
builds a real GodotEnv from `config` (lazy import so the module imports without a socket).
"""
from __future__ import annotations

import functools
from typing import Dict, Optional

# Top-level (not lazy like torch/godot_rl): the base class must be resolved at class-definition
# time, so `GodotParallelEnv(ParallelEnv)` forces this import. Intentional convention exception.
from pettingzoo import ParallelEnv


class GodotParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "GodotParallelEnv"}

    def __init__(
        self,
        port: Optional[int] = None,
        show_window: bool = False,
        seed: int = 0,
        config: Optional[Dict] = None,
        godot_env=None,
    ) -> None:
        config = config or {}
        if godot_env is not None:
            self.godot_env = godot_env
        else:
            # Truncation-aware subclass (#12): step() yields the real (terminated, truncated)
            # split from the wire's additive field instead of upstream's done-duplication.
            from godot_rl.core.godot_env import GodotEnv
            from godot_env_truncation import TruncationAwareGodotEnv

            if port is None:
                port = GodotEnv.DEFAULT_PORT
            reserved = {"env_path", "show_window", "action_repeat", "speedup", "seed", "port"}
            # Forward any non-reserved config keys to GodotEnv as kwargs (reserved keys are passed explicitly above/below).
            extra = {k: v for k, v in config.items() if k not in reserved}
            self.godot_env = TruncationAwareGodotEnv(
                env_path=config.get("env_path"),
                show_window=show_window,
                action_repeat=config.get("action_repeat", 1),
                speedup=config.get("speedup", 1),
                convert_action_space=False,
                seed=seed,
                port=port,
                **extra,
            )

        self.render_mode = None
        self.possible_agents = list(range(self.godot_env.num_envs))
        self.agents = self.possible_agents[:]
        self.agent_policy_names = list(self.godot_env.agent_policy_names)
        self.observation_spaces = {
            agent: self.godot_env.observation_spaces[i]
            for i, agent in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            agent: self.godot_env.tuple_action_spaces[i]
            for i, agent in enumerate(self.possible_agents)
        }
        import numpy as np

        # Zero action for a done agent: one int64 per Tuple(Discrete,...) component (matches the
        # (len(nvec),) action rows the trainer scatters). Explicit dtype, not np.zeros_like(sample()).
        self._zero_actions = {
            agent: np.zeros(len(self.action_spaces[agent].spaces), dtype=np.int64)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        self.godot_env.close()

    def reset(self, seed=None, options=None):
        # seed: GodotEnv seeds at construction; reset-time re-seeding is unsupported by the bridge.
        godot_obs, godot_infos = self.godot_env.reset()
        self.agents = self.possible_agents[:]
        observations = {agent: godot_obs[i] for i, agent in enumerate(self.possible_agents)}
        infos = {agent: godot_infos[i] for i, agent in enumerate(self.possible_agents)}
        return observations, infos

    def step(self, actions):
        # Fixed-population: self.agents is NOT pruned on per-agent termination (godot_rl keeps every
        # agent every step; done agents get zero actions).
        godot_actions = [
            actions[agent] if agent in actions else self._zero_actions[agent]
            for agent in self.possible_agents
        ]
        godot_obs, godot_rewards, godot_dones, godot_truncations, godot_infos = self.godot_env.step(
            godot_actions, order_ij=True
        )
        observations = {agent: godot_obs[i] for i, agent in enumerate(self.possible_agents) if agent in actions}
        rewards = {agent: godot_rewards[i] for i, agent in enumerate(self.possible_agents) if agent in actions}
        terminations = {agent: bool(godot_dones[i]) for i, agent in enumerate(self.possible_agents) if agent in actions}
        # Real per-agent truncations (#12). With an injected plain GodotEnv (or a pre-#12 scene)
        # slot 4 duplicates `done`; TruncationAwareGodotEnv makes it the true Gymnasium flag.
        truncations = {agent: bool(godot_truncations[i]) for i, agent in enumerate(self.possible_agents) if agent in actions}
        infos = {agent: godot_infos[i] for i, agent in enumerate(self.possible_agents) if agent in actions}
        return observations, rewards, terminations, truncations, infos
