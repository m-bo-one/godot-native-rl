#!/usr/bin/env python3
"""Behavior cloning of the Sorter scripted expert with the attention encoder (#46/#260 M4).

Vanilla PPO stalls on the Sorter's sparse strict-ordering reward (two independent flat runs at the
random baseline). BC on the scripted expert — which reliably solves variable-count episodes — is
the reliable route to a *solving* deploy net that still uses the #46 attention encoder. Trains the
same AttentionEncoder + a single linear action head by cross-entropy to predict the expert action,
then exports via the M2 direct exporter (identical ncnn graph to the RL trainers).

Run: .venv-train/bin/python scripts/train_sorter_bc.py \
        --demos examples/sorter/demos/sorter_expert_demos.json \
        --outdir examples/sorter/models --stem sorter_attention
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False, description="BC of the Sorter expert (attention).")
    p.add_argument("--demos", required=True, help="gnrl_v1/godot_rl expert demo file")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--n_entities", type=int, default=6)
    p.add_argument("--feat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/sorter_attention_bc.pt")
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--stem", type=str, default="sorter_attention")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from load_expert_demos import load_demos, flatten_pairs
    from train_sorter_cleanrl import build_sorter_agent, export_sorter_policy

    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = flatten_pairs(load_demos(args.demos))
    obs = torch.tensor(np.asarray(x, dtype=np.float32))
    act = torch.tensor(np.asarray(y, dtype=np.int64)).reshape(-1)
    obs_dim = obs.shape[1]
    n_act = int(act.max().item()) + 1
    print("BC dataset: %d pairs, obs_dim=%d, n_act=%d" % (obs.shape[0], obs_dim, n_act))
    # Sorter has a fixed 5-way action head; guard against a demo set that never used a direction.
    n_act = max(n_act, 5)

    agent = build_sorter_agent(obs_dim, n_act, args.embed_dim, args.num_heads,
                               args.n_entities, args.feat)
    # BC trains only the encoder + actor (the critic is unused at deploy).
    params = list(agent.encoder.parameters()) + list(agent.actor.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        logits = agent.actor(agent.encoder(obs))
        loss = F.cross_entropy(logits, act)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            acc = (logits.argmax(dim=1) == act).float().mean().item()
            print("epoch %d/%d ce=%.4f train_acc=%.3f" % (epoch, args.epochs, float(loss), acc))

    pt_path = Path(args.save_model_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), pt_path)
    print("Saved BC policy to:", pt_path)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    param, binp = export_sorter_policy(agent, args.outdir, args.stem)
    print("Exported ncnn to:", param, binp)


if __name__ == "__main__":
    main()
