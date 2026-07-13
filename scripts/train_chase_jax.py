#!/usr/bin/env python3
"""Train chase on the JAX batch twin — whole-batch jit rollouts, no Godot, no socket (#361).

PureJaxRL-style single-file PPO over chase_twin_jax.batched_step: rollouts, GAE and the PPO
update are all jit-compiled JAX, so a large batch of envs advances as one kernel (and the same
code runs unchanged on GPU). The trained flax actor is converted weight-for-weight to a torch
nn.Sequential (unit-tested parity seam), traced to TorchScript + shape sidecar, and converted
to ncnn by scripts/export_to_ncnn.py — the identical deploy contract as the NumPy twin (#37).

    .venv-train/bin/python scripts/train_chase_jax.py --timesteps 400000 --num_envs 64

Deps: the OPTIONAL requirements-jax.txt add-on for .venv-train (jax/flax/optax). Heavy imports
are lazy so the module imports (for the export-parity unit test) without running a training.
Honest benchmark: prints measured env-steps/s — on CPU the win over the NumPy twin appears at
larger batch; on GPU the same code scales further (#361's claim is measured, not assumed).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import flax.linen as nn
import jax
import jax.numpy as jnp


class Actor(nn.Module):
    """obs -> raw action logits; mirrors the deploy MLP (2x64 tanh) of the other chase nets."""

    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(self.n_actions)(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(1)(x).squeeze(-1)


def params_to_torch_actor(params, obs_dim: int, n_actions: int):
    """flax Actor params -> an equivalent torch nn.Sequential (Linear/Tanh; kernel transposed).
    The unit-tested deploy seam: torch(x) == flax.apply(params, x)."""
    import numpy as np
    import torch
    import torch.nn as tnn

    layers = params["params"]
    seq = tnn.Sequential(
        tnn.Linear(obs_dim, 64), tnn.Tanh(),
        tnn.Linear(64, 64), tnn.Tanh(),
        tnn.Linear(64, n_actions),
    )
    with torch.no_grad():
        for torch_layer, name in ((seq[0], "Dense_0"), (seq[2], "Dense_1"), (seq[4], "Dense_2")):
            torch_layer.weight.copy_(torch.from_numpy(np.asarray(layers[name]["kernel"]).T.copy()))
            torch_layer.bias.copy_(torch.from_numpy(np.asarray(layers[name]["bias"]).copy()))
    return seq.eval()


class Rollout(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    values: jnp.ndarray


def compute_gae(rewards, values, last_value, gamma: float, lam: float):
    """GAE over a (T, N) rollout with no terminals (the twin only truncates — matches the
    NumPy twin's 1000-decision horizon handled by rollout slicing)."""
    def scan_fn(carry, xs):
        reward, value, next_value = xs
        delta = reward + gamma * next_value - value
        adv = delta + gamma * lam * carry
        return adv, adv

    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    _, advs = jax.lax.scan(scan_fn, jnp.zeros_like(last_value),
                           (rewards, values, next_values), reverse=True)
    return advs, advs + values


def main() -> int:
    import time

    import numpy as np
    import optax
    import torch

    import chase_twin_jax as twin
    from export_to_ncnn import write_shape_sidecar

    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=400_000, help="total env-steps (decisions x envs)")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=128, help="rollout length per update")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--out", type=str, default="models/chase_jax_twin.pt")
    args = p.parse_args()

    key = jax.random.PRNGKey(args.seed)
    actor = Actor(n_actions=twin.N_ACTIONS)
    critic = Critic()
    key, ka, kc = jax.random.split(key, 3)
    dummy = jnp.zeros((1, twin.OBS_DIM))
    actor_params = actor.init(ka, dummy)
    critic_params = critic.init(kc, dummy)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = tx.init((actor_params, critic_params))

    key, kr = jax.random.split(key)
    state = twin.reset_batch(args.num_envs, kr)
    obs = jnp.asarray(np.asarray(twin._batched_obs(state.agent, state.target)))

    def policy_step(params, state, obs, key):
        logits = actor.apply(params[0], obs)
        key, sub = jax.random.split(key)
        actions = jax.random.categorical(sub, logits)
        log_probs = jax.nn.log_softmax(logits)[jnp.arange(obs.shape[0]), actions]
        values = critic.apply(params[1], obs)
        new_state, new_obs, rewards, _ = twin.batched_step(state, actions)
        return new_state, new_obs, key, Rollout(obs, actions, log_probs, rewards, values)

    @jax.jit
    def collect(params, state, obs, key):
        def scan_fn(carry, _):
            state, obs, key = carry
            state, new_obs, key, transition = policy_step(params, state, obs, key)
            return (state, new_obs, key), transition
        (state, obs, key), rollout = jax.lax.scan(scan_fn, (state, obs, key), None, args.num_steps)
        last_value = critic.apply(params[1], obs)
        return state, obs, key, rollout, last_value

    def loss_fn(params, batch, advs, returns):
        logits = actor.apply(params[0], batch.obs)
        log_all = jax.nn.log_softmax(logits)
        log_probs = log_all[jnp.arange(batch.obs.shape[0]), batch.actions]
        ratio = jnp.exp(log_probs - batch.log_probs)
        norm_adv = (advs - advs.mean()) / (advs.std() + 1e-8)
        pg = -jnp.minimum(ratio * norm_adv,
                          jnp.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * norm_adv).mean()
        values = critic.apply(params[1], batch.obs)
        v_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy = -(jnp.exp(log_all) * log_all).sum(-1).mean()
        return pg + v_loss - args.ent_coef * entropy

    @jax.jit
    def update(params, opt_state, rollout, last_value, key):
        advs, returns = compute_gae(rollout.rewards, rollout.values, last_value,
                                    args.gamma, args.gae_lambda)
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), rollout)
        advs = advs.reshape(-1)
        returns = returns.reshape(-1)
        batch_size = flat.obs.shape[0]
        mb_size = batch_size // args.minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, batch_size)

            def minibatch(carry, idx):
                params, opt_state = carry
                sel = jax.lax.dynamic_slice_in_dim(perm, idx * mb_size, mb_size)
                mb = jax.tree.map(lambda x: x[sel], flat)
                grads = jax.grad(loss_fn)(params, mb, advs[sel], returns[sel])
                updates, opt_state = tx.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), None

            (params, opt_state), _ = jax.lax.scan(minibatch, (params, opt_state),
                                                  jnp.arange(args.minibatches))
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(epoch, (params, opt_state, key),
                                                   None, args.epochs)
        return params, opt_state, key

    params = (actor_params, critic_params)
    steps_per_update = args.num_envs * args.num_steps
    n_updates = max(1, args.timesteps // steps_per_update)
    t0 = time.time()
    for u in range(n_updates):
        state, obs, key, rollout, last_value = collect(params, state, obs, key)
        params, opt_state, key = update(params, opt_state, rollout, last_value, key)
        if u % 10 == 0 or u == n_updates - 1:
            mean_r = float(rollout.rewards.mean()) * args.num_steps
            catches = int(np.asarray(state.catches).sum())
            print(f"update {u + 1}/{n_updates} mean_rollout_return={mean_r:.2f} "
                  f"cumulative_catches={catches}")
    total = n_updates * steps_per_update
    dt = time.time() - t0
    print("Trained %d env-steps in %.1fs (%.0f steps/s, %d envs, JAX %s, no Godot)"
          % (total, dt, total / max(dt, 1e-9), args.num_envs, jax.devices()[0].platform))

    torch_actor = params_to_torch_actor(params[0], twin.OBS_DIM, twin.N_ACTIONS)
    shape = [1, twin.OBS_DIM]
    with torch.no_grad():
        scripted = torch.jit.trace(torch_actor, torch.zeros(*shape, dtype=torch.float32))
    pt_path = pathlib.Path(args.out)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, shape)
    print("exported TorchScript to:", pt_path)
    print("wrote shape sidecar:    ", sidecar)
    print("next: export_to_ncnn.py", pt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
