#!/usr/bin/env python3
"""Train the Sorter agent with a single-file CleanRL PPO using the attention encoder (#46/#258).

The Sorter env emits a variable-length entity observation (EntitySensor2D block:
[n*feat zero-padded entities][n presence flags]); the shared trunk is the AttentionEncoder
(scripts/attention_encoder.py), which reshapes the flat obs internally, so the CleanRL loop keeps
passing flat obs unchanged. On completion the policy is exported to native ncnn via the DIRECT
hand-written exporter (scripts/export_statedict_to_ncnn.py::attention_policy_layers, through
spike_attention_ncnn.export_encoder_policy) — NOT ONNX/pnnx (pnnx decomposes the hand-built
attention; the direct path is byte-pinned to the #307 ncnn-verified fixtures).

Design: docs/superpowers/specs/2026-07-07-attention-encoder-m2-torch-side-design.md

Reuses the tested pure PPO helpers from scripts/train_cleanrl.py (compute_gae, layer_init,
num_updates, _split_categoricals, obs_dim, act_layout). Heavy imports (torch/numpy/godot_rl) are
lazy inside main() so build_sorter_agent stays unit-testable; build_sorter_agent needs torch and
is imported at module load only where torch exists.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene.
See scripts/train_sorter.sh for orchestration.
"""
from __future__ import annotations

import argparse
from typing import NamedTuple, Sequence

# Shared, unit-tested PPO helpers (DRY — do not re-implement).
from train_cleanrl import (  # noqa: F401
    act_layout,
    compute_gae,
    layer_init,
    num_updates,
    obs_dim,
    _split_categoricals,
)


class SorterConfig(NamedTuple):
    timesteps: int
    speedup: int
    action_repeat: int
    seed: int
    num_steps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    num_minibatches: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    n_entities: int
    feat: int
    embed_dim: int
    num_heads: int
    save_model_path: str
    outdir: str
    stem: str


def parse_args(argv: Sequence[str] | None = None) -> SorterConfig:
    p = argparse.ArgumentParser(allow_abbrev=False,
                                description="CleanRL single-file PPO for Sorter (attention encoder).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--n_entities", type=int, default=6, help="max entities (Sorter max_tiles)")
    p.add_argument("--feat", type=int, default=4, help="per-entity feature width")
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--save_model_path", type=str, default="models/sorter_attention.pt")
    p.add_argument("--outdir", type=str, default="models",
                   help="directory for the exported <stem>.ncnn.{param,bin}")
    p.add_argument("--stem", type=str, default="sorter_attention")
    a = p.parse_args(argv)
    return SorterConfig(**vars(a))


def build_sorter_agent(obs_dim: int, n_act: int, embed_dim: int = 16, num_heads: int = 2,
                       n_entities: int = 6, feat: int = 4):
    """actor/critic nn.Module with an AttentionEncoder trunk. Lazy torch import.

    Asserts n_entities*feat + n_entities == obs_dim (the fixed EntitySensor obs contract).
    """
    import torch.nn as nn

    from attention_encoder import AttentionEncoder

    assert n_entities * feat + n_entities == obs_dim, (
        "obs_dim %d != n*f+n (n=%d f=%d)" % (obs_dim, n_entities, feat))

    class Agent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AttentionEncoder(n_entities, feat, embed_dim, num_heads)
            self.actor = layer_init(nn.Linear(embed_dim, n_act), std=0.01)
            self.critic = layer_init(nn.Linear(embed_dim, 1), std=1.0)

        def logits(self, obs):
            return self.actor(self.encoder(obs))

        def value(self, obs):
            return self.critic(self.encoder(obs)).squeeze(-1)

    return Agent()


def export_sorter_policy(agent, outdir: str, stem: str):
    """Export the trained encoder + actor head to native ncnn via the direct exporter."""
    from spike_attention_ncnn import export_encoder_policy

    return export_encoder_policy(agent.encoder, agent.actor.weight, agent.actor.bias, outdir, stem)


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = CleanRLGodotEnv(env_path=None, show_window=False, seed=cfg.seed, n_parallel=1,
                          speedup=cfg.speedup, action_repeat=cfg.action_repeat)
    n_envs = env.num_envs
    observation_dim = obs_dim(env.single_observation_space)
    total_logits, nvec = act_layout(env.single_action_space)
    print("obs_dim=%d action_logits=%d nvec=%s num_envs=%d"
          % (observation_dim, total_logits, nvec, n_envs))

    agent = build_sorter_agent(observation_dim, total_logits, cfg.embed_dim, cfg.num_heads,
                               cfg.n_entities, cfg.feat).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    num_steps = cfg.num_steps
    batch_size = num_steps * n_envs
    minibatch_size = max(1, batch_size // cfg.num_minibatches)
    updates = num_updates(cfg.timesteps, num_steps, n_envs)
    if updates == 0:
        raise SystemExit("timesteps (%d) < one rollout batch (num_steps*num_envs=%d) — nothing to "
                         "train; raise --timesteps or lower --num_steps." % (cfg.timesteps, batch_size))
    print("running %d updates (batch_size=%d, minibatch_size=%d)"
          % (updates, batch_size, minibatch_size))

    obs_buf = torch.zeros((num_steps, n_envs, observation_dim), device=device)
    actions_buf = torch.zeros((num_steps, n_envs, len(nvec)), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, n_envs), device=device)
    rewards_buf = torch.zeros((num_steps, n_envs), device=device)
    dones_buf = torch.zeros((num_steps, n_envs), device=device)
    values_buf = torch.zeros((num_steps, n_envs), device=device)

    next_obs_np, _ = env.reset(cfg.seed)
    next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
    next_done = torch.zeros(n_envs, device=device)

    for update in range(updates):
        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                logits = agent.logits(next_obs)
                value = agent.value(next_obs)
            dists = _split_categoricals(logits, nvec)
            sampled = [d.sample() for d in dists]
            action = torch.stack(sampled, dim=1)
            logprob = sum(d.log_prob(a) for d, a in zip(dists, sampled))
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            action_np = action.cpu().numpy().astype(np.int64)
            next_obs_np, reward, terminations, truncations, _ = env.step(action_np)
            done = np.logical_or(np.asarray(terminations), np.asarray(truncations)).astype(np.float32)
            rewards_buf[step] = torch.tensor(np.asarray(reward, dtype=np.float32), device=device)
            next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
            next_done = torch.tensor(done, device=device)

        with torch.no_grad():
            next_value = agent.value(next_obs)
        advantages_np, returns_np = compute_gae(
            rewards_buf.cpu().numpy(), values_buf.cpu().numpy(), dones_buf.cpu().numpy(),
            next_value.cpu().numpy(), next_done.cpu().numpy(), cfg.gamma, cfg.gae_lambda)
        advantages = torch.tensor(advantages_np, device=device)
        returns = torch.tensor(returns_np, device=device)

        b_obs = obs_buf.reshape(-1, observation_dim)
        b_actions = actions_buf.reshape(-1, len(nvec))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        v_loss = torch.tensor(0.0)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                logits = agent.logits(b_obs[mb_inds])
                dists = _split_categoricals(logits, nvec)
                mb_actions = b_actions[mb_inds]
                new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                entropy = sum(d.entropy() for d in dists)
                new_value = agent.value(b_obs[mb_inds])

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        steps_done = (update + 1) * batch_size
        print("update %d/%d steps=%d mean_reward=%.4f value_loss=%.4f"
              % (update + 1, updates, steps_done, float(rewards_buf.mean()), float(v_loss.detach())))

    pt_path = pathlib.Path(cfg.save_model_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), pt_path)
    print("Saved torch policy to:", pt_path)

    pathlib.Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    param, binp = export_sorter_policy(agent, cfg.outdir, cfg.stem)
    print("Exported ncnn to:", param, binp)

    env.close()


if __name__ == "__main__":
    main()
