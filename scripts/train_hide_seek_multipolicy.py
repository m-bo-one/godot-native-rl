#!/usr/bin/env python3
"""Train Hide & Seek with TWO distinct policies (seeker + hider) over the godot-rl bridge.

A custom single-file multi-policy PPO (sibling of scripts/train_cleanrl.py). CleanRLGodotEnv
vectorizes over the N Godot agents as N parallel envs; this trainer reads agent_policy_names, routes
each agent index to its policy, maintains one PPO learner per distinct name, and exports each actor
to TorchScript (+ a shape sidecar) for scripts/export_to_ncnn.py --via torchscript -> native ncnn.

Run this FIRST (opens the server on 11008, waits), THEN launch the Godot scene (the multi-policy
train scene self-declares multi_policy on its Sync node — no --multi-policy cmdline gate; see #73).
See scripts/train_hide_seek_multipolicy.sh. Design:
docs/superpowers/specs/2026-06-05-multi-policy-trained-example-design.md

Heavy imports (torch/numpy/godot_rl) are LAZY so the pure helpers stay unit-testable. The pure PPO
helpers (compute_gae, num_updates, layer_init, _build_agent) are reused from train_cleanrl; export is
TorchScript (not ONNX) so it stays in the numpy<2 world stable-baselines3 requires.

ASSUMPTION: all policies share one observation and action shape (every learner is built from
env.single_observation_space / single_action_space, i.e. agent 0's). True for Hide & Seek (seeker and
hider have identical 15-float obs + a 5-way discrete `move`). For heterogeneous policies, build each
learner from its own role's spaces instead.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Sequence


def policy_index_map(agent_policy_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map each distinct policy name to the agent indices using it (first-seen key order,
    ascending indices). e.g. ["seeker","hider","seeker"] -> {"seeker":[0,2],"hider":[1]}."""
    out: Dict[str, List[int]] = {}
    for i, name in enumerate(agent_policy_names):
        out.setdefault(name, []).append(i)
    return out


def split_by_policy(batched, index_map: Dict[str, List[int]]):
    """Slice a (n_agents, ...) array into {name: array[indices]} per policy. Lazy numpy."""
    import numpy as np

    arr = np.asarray(batched)
    return {name: arr[idx] for name, idx in index_map.items()}


def stitch_actions(per_policy_actions, index_map: Dict[str, List[int]], n_agents: int):
    """Inverse of split_by_policy for actions: scatter each policy's (n_p, action_dim) actions back
    into a single (n_agents, action_dim) int64 array in agent order. Lazy numpy."""
    import numpy as np

    first = next(iter(per_policy_actions.values()))
    action_dim = int(np.asarray(first).shape[1])
    out = np.zeros((n_agents, action_dim), dtype=np.int64)
    for name, idx in index_map.items():
        out[idx] = np.asarray(per_policy_actions[name])
    return out


def export_actor_as_torchscript(agent, observation_dim: int, pt_path) -> None:
    """Trace a CleanRL-style agent's deterministic actor (obs -> raw action logits) to TorchScript
    and write the `<pt>.shape.json` sidecar so `export_to_ncnn.py <pt>` auto-derives the inputshape.

    ONNX-free on purpose: torch 2.x's `torch.onnx.export` pulls in onnxscript/onnx, which require
    numpy>=2 and collide with stable-baselines3 (numpy<2); the pnnx TorchScript path avoids that
    entirely. Output is the raw logits (length sum(nvec)); the deploy-side ActionDecode argmaxes per
    segment, exactly as the ONNX path would. Lazy torch import keeps the module unit-test-light.
    """
    import pathlib

    import torch

    from export_to_ncnn import write_shape_sidecar  # import-light (no torch/onnx at module load)

    class TracedActor(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs):
            return self.inner.logits(obs)

    actor = TracedActor(agent).to("cpu").eval()
    shape = [1, observation_dim]
    with torch.no_grad():
        scripted = torch.jit.trace(actor, torch.zeros(*shape, dtype=torch.float32))
    pt_path = pathlib.Path(pt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    write_shape_sidecar(pt_path, shape)


def snapshot_updates(total_updates: int, steps_per_update: int, every: int) -> List[int]:
    """Which (0-based) update indices trigger a mid-run pool snapshot (#189): the update whose
    cumulative env-steps first crosses each multiple of `every`. Empty when every <= 0 (off)."""
    if every <= 0 or steps_per_update <= 0:
        return []
    out: List[int] = []
    next_boundary = every
    for u in range(total_updates):
        done = (u + 1) * steps_per_update
        if done >= next_boundary:
            out.append(u)
            while next_boundary <= done:
                next_boundary += every
    return out


def snapshot_name(policy: str, update: int) -> str:
    """Pool-member name for a mid-run freeze: policy-scoped + zero-padded 1-based update index,
    so ls order == training order (the league's 'latest' pick_mode relies on registration order,
    but sortable names keep the pool dir humanly readable)."""
    return "%s_live_u%06d" % (policy, update + 1)


def freeze_snapshot(agent, observation_dim: int, policy: str, update: int, pool_base) -> None:
    """Freeze one learner's actor into the opponent pool MID-RUN (#189): TorchScript -> ncnn in
    pool_base/<policy>/ + register in that pool's ELO ledger (pool.json) at the learner rating —
    the same artifacts the alternating league (#29) produces per phase, so SelfPlayManager /
    pick_opponent consume this pool unchanged. The .pt intermediates are removed (the pool holds
    deploy ncnn only)."""
    import pathlib

    import export_to_ncnn
    import selfplay_phase

    pool_dir = pathlib.Path(pool_base) / policy
    pool_dir.mkdir(parents=True, exist_ok=True)
    # #371: derive a name that can't collide with a PRIOR run's pool member. Reruns restart the
    # update schedule from the same indices, so without this the export would overwrite an existing
    # snapshot's weights and register_snapshot_file would then raise on the duplicate name — killing
    # the (checkpoint-less) trainer and leaving the surviving ledger entry pointing at other weights.
    # Uniquify BEFORE exporting so old snapshots + ratings stay intact and the pool grows additively.
    name = selfplay_phase.unique_member_name(
        selfplay_phase.read_member_names(pool_dir), snapshot_name(policy, update))
    pt_path = pool_dir / f"{name}.pt"
    export_actor_as_torchscript(agent, observation_dim, pt_path)
    rc = export_to_ncnn.main([str(pt_path), "--outdir", str(pool_dir)])
    if rc != 0:
        raise RuntimeError(f"ncnn conversion failed for snapshot {name} (rc {rc})")
    pt_path.unlink(missing_ok=True)
    pathlib.Path(str(pt_path) + ".shape.json").unlink(missing_ok=True)

    ledger = selfplay_phase.register_snapshot_file(pool_dir, name)
    print(f"[snapshot] froze {name} into {pool_dir} (rating {ledger['members'][name]['rating']})")


class MultiPolicyConfig(NamedTuple):
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
    export_dir: str
    policy_names: tuple  # expected names, for a fail-fast sanity check against the wire
    snapshot_every: int  # env-steps between mid-run pool freezes (#189); 0 = off
    pool_dir: str


def parse_args(argv: Sequence[str] | None = None) -> "MultiPolicyConfig":
    p = argparse.ArgumentParser(allow_abbrev=False, description="Multi-policy PPO for hide & seek.")
    p.add_argument("--timesteps", type=int, default=800_000)
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
    p.add_argument("--export_dir", type=str, default="models")
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="env-steps between mid-run opponent-pool freezes of BOTH live learners "
                        "(#189 simultaneous self-play); 0 = off")
    p.add_argument("--pool_dir", type=str, default="models/selfplay_pool",
                   help="pool base dir (per-policy subdirs; the #29 league layout)")
    a = p.parse_args(argv)
    return MultiPolicyConfig(
        timesteps=a.timesteps, speedup=a.speedup, action_repeat=a.action_repeat, seed=a.seed,
        num_steps=a.num_steps, learning_rate=a.learning_rate, gamma=a.gamma,
        gae_lambda=a.gae_lambda, update_epochs=a.update_epochs, num_minibatches=a.num_minibatches,
        clip_coef=a.clip_coef, ent_coef=a.ent_coef, vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm, export_dir=a.export_dir,
        policy_names=("seeker", "hider"),
        snapshot_every=a.snapshot_every, pool_dir=a.pool_dir,
    )


def _policy_names_from_env(env) -> list:
    """Per-agent policy names from CleanRLGodotEnv's underlying GodotEnv(s). With n_parallel=1 the
    names live on env.envs[0].agent_policy_names (confirmed against godot_rl 0.8.2)."""
    for inner in getattr(env, "envs", []) or []:
        names = getattr(inner, "agent_policy_names", None)
        if names:
            return list(names)
    raise RuntimeError("could not read agent_policy_names from CleanRLGodotEnv.envs")


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

    import train_cleanrl as tc  # reuse compute_gae, num_updates, layer_init, _build_agent, etc.

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = CleanRLGodotEnv(
        env_path=None, show_window=False, seed=cfg.seed, n_parallel=1,
        speedup=cfg.speedup, action_repeat=cfg.action_repeat,
    )
    n_agents = env.num_envs
    observation_dim = tc.obs_dim(env.single_observation_space)
    total_logits, nvec = tc.act_layout(env.single_action_space)

    names = _policy_names_from_env(env)
    index_map = policy_index_map(names)
    print(f"obs_dim={observation_dim} logits={total_logits} nvec={nvec} "
          f"n_agents={n_agents} policies={ {k: len(v) for k, v in index_map.items()} }")
    for expected in cfg.policy_names:
        if expected not in index_map:
            raise RuntimeError(f"expected policy '{expected}' not on the wire (got {list(index_map)})")

    # Per-policy learners + rollout storage (one set per distinct policy name).
    agents, opts, bufs = {}, {}, {}
    num_steps = cfg.num_steps
    for name, idx in index_map.items():
        np_ = len(idx)
        ag = tc._build_agent(observation_dim, total_logits).to(device)
        agents[name] = ag
        opts[name] = torch.optim.Adam(ag.parameters(), lr=cfg.learning_rate, eps=1e-5)
        bufs[name] = dict(
            obs=torch.zeros((num_steps, np_, observation_dim), device=device),
            actions=torch.zeros((num_steps, np_, len(nvec)), dtype=torch.long, device=device),
            logprobs=torch.zeros((num_steps, np_), device=device),
            rewards=torch.zeros((num_steps, np_), device=device),
            dones=torch.zeros((num_steps, np_), device=device),
            values=torch.zeros((num_steps, np_), device=device),
        )

    updates = tc.num_updates(cfg.timesteps, num_steps, n_agents)
    print(f"running {updates} updates over {n_agents} agents")
    # #189 simultaneous self-play: cross-freeze BOTH live learners into the opponent pool at
    # these updates (no phase restarts — the pool grows during the run).
    freeze_at = set(snapshot_updates(updates, num_steps * n_agents, cfg.snapshot_every))
    if freeze_at:
        print(f"mid-run pool snapshots at updates {sorted(u + 1 for u in freeze_at)} -> {cfg.pool_dir}")

    next_obs_np, _ = env.reset(cfg.seed)
    next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
    next_done = torch.zeros(n_agents, device=device)

    def split_t(t):  # split a (n_agents, ...) torch tensor per policy
        return {name: t[idx] for name, idx in index_map.items()}

    for update in range(updates):
        for step in range(num_steps):
            no_split = split_t(next_obs)
            nd_split = split_t(next_done)
            per_policy_action = {}
            for name, idx in index_map.items():
                ag = agents[name]
                b = bufs[name]
                ob = no_split[name]
                b["obs"][step] = ob
                b["dones"][step] = nd_split[name]
                with torch.no_grad():
                    logits = ag.logits(ob)
                    value = ag.value(ob)
                dists = tc._split_categoricals(logits, nvec)
                sampled = [d.sample() for d in dists]
                action = torch.stack(sampled, dim=1)
                b["actions"][step] = action
                b["logprobs"][step] = sum(d.log_prob(a) for d, a in zip(dists, sampled))
                b["values"][step] = value
                per_policy_action[name] = action.cpu().numpy().astype(np.int64)
            full_action = stitch_actions(per_policy_action, index_map, n_agents)

            next_obs_np, reward, terminations, truncations, _ = env.step(full_action)
            done = np.logical_or(np.asarray(terminations), np.asarray(truncations)).astype(np.float32)
            reward_t = torch.tensor(np.asarray(reward, dtype=np.float32), device=device)
            for name, idx in index_map.items():
                bufs[name]["rewards"][step] = reward_t[idx]
            next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
            next_done = torch.tensor(done, device=device)

        # Per-policy PPO update (mirrors train_cleanrl, independently per learner).
        for name, idx in index_map.items():
            ag, opt, b = agents[name], opts[name], bufs[name]
            np_ = len(idx)
            with torch.no_grad():
                next_value = ag.value(next_obs[idx])
            adv_np, ret_np = tc.compute_gae(
                b["rewards"].cpu().numpy(), b["values"].cpu().numpy(), b["dones"].cpu().numpy(),
                next_value.cpu().numpy(), next_done[idx].cpu().numpy(), cfg.gamma, cfg.gae_lambda)
            advantages = torch.tensor(adv_np, device=device)
            returns = torch.tensor(ret_np, device=device)

            b_obs = b["obs"].reshape(-1, observation_dim)
            b_actions = b["actions"].reshape(-1, len(nvec))
            b_logprobs = b["logprobs"].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            batch_size = num_steps * np_
            minibatch_size = max(1, batch_size // cfg.num_minibatches)
            b_inds = np.arange(batch_size)
            for _ in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    logits = ag.logits(b_obs[mb])
                    dists = tc._split_categoricals(logits, nvec)
                    mb_actions = b_actions[mb]
                    new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                    entropy = sum(d.entropy() for d in dists)
                    new_value = ag.value(b_obs[mb])
                    logratio = new_logprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ag.parameters(), cfg.max_grad_norm)
                    opt.step()

        msg = " ".join(f"{name}_rew={float(bufs[name]['rewards'].mean()):.3f}" for name in index_map)
        print(f"update {update + 1}/{updates} {msg}")

        if update in freeze_at:
            for name in index_map:
                freeze_snapshot(agents[name], observation_dim, name, update, cfg.pool_dir)

    # Export each policy's actor to TorchScript (+ shape sidecar) for the ncnn pipeline.
    # TorchScript (not ONNX) so the export stays in the numpy<2 world stable-baselines3 needs.
    outdir = pathlib.Path(cfg.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in index_map:
        pt_path = outdir / f"hide_seek_{name}.pt"
        export_actor_as_torchscript(agents[name], observation_dim, pt_path)
        print("Exported TorchScript to:", pt_path)

    env.close()


if __name__ == "__main__":
    main()
