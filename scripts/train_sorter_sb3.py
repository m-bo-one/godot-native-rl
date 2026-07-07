#!/usr/bin/env python3
"""SB3 PPO Sorter trainer with the attention features extractor (#46/#259 M3).

Mirrors scripts/train_chase.py (SB3 PPO over StableBaselinesGodotEnv). The policy uses
AttentionFeaturesExtractor + net_arch=[] so the deploy actor is encoder -> a single linear head;
export reuses the M2 direct exporter (spike_attention_ncnn.export_encoder_policy). No VecNormalize,
so the deployed encoder sees the same raw obs distribution it trained on.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene.
See scripts/train_sorter_sb3.sh for orchestration.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence


def export_sb3_sorter_policy(model, outdir: str, stem: str = "sorter_attention_sb3"):
    """Export an SB3 attention policy (net_arch=[]) to ncnn via the M2 direct exporter.

    Pulls the shared AttentionEncoder (the features extractor) + the single-linear action head and
    hands them to spike_attention_ncnn.export_encoder_policy. Returns (param, bin) paths.
    """
    from spike_attention_ncnn import export_encoder_policy

    policy = model.policy
    enc = policy.features_extractor.encoder
    head = policy.action_net
    return export_encoder_policy(enc, head.weight, head.bias, outdir, stem)


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(allow_abbrev=False, description="SB3 PPO Sorter (attention encoder).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--n_entities", type=int, default=6)
    p.add_argument("--feat", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--save_model_path", type=str, default="models/sorter_attention_sb3.zip")
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--stem", type=str, default="sorter_attention_sb3")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    from attention_features_extractor import AttentionFeaturesExtractor

    args = parse_args(argv)
    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    env = StableBaselinesGodotEnv(env_path=None, show_window=False, seed=args.seed, n_parallel=1,
                                  speedup=args.speedup, action_repeat=args.action_repeat)
    env = VecMonitor(env)

    # net_arch=[] is load-bearing: it keeps action_net a single Linear(embed_dim, sum(nvec)) so the
    # deploy actor is exactly encoder -> head (the M2 export contract).
    model = PPO(
        "MultiInputPolicy", env, verbose=1, n_steps=args.n_steps, batch_size=64,
        policy_kwargs=dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(embed_dim=args.embed_dim, num_heads=args.num_heads,
                                           n_entities=args.n_entities, feat=args.feat),
            net_arch=[]),
    )
    model.learn(args.timesteps)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 policy to:", zip_path)

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    param, binp = export_sb3_sorter_policy(model, args.outdir, args.stem)
    print("Exported ncnn to:", param, binp)
    env.close()


if __name__ == "__main__":
    main()
