#!/usr/bin/env python3
"""Train chase_rays on the pure-NumPy twin — NO Godot, NO socket (#364).

Same shape as train_chase_twin.py (#37) over ChaseRaysTwinEnv: chase + solid AABB obstacles +
an analytic 8-ray surround obs (13-dim). The trained net deploys into the REAL engine where the
rays come from actual physics casts — the twin's slab-method distances match them exactly (see
test_chase_rays.gd), so the policy transfers.

    .venv-train/bin/python scripts/train_chase_rays_twin.py --timesteps 400000 --n_envs 8

Export: the deterministic actor -> TorchScript (`.pt` + shape sidecar) via the shared #52 exporter;
`scripts/export_to_ncnn.py` then converts to ncnn (run by scripts/train_chase_twin.sh).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_formats import export_policy, add_format_arg  # noqa: E402


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from chase_rays_twin_env import ChaseRaysTwinEnv

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/chase_rays_twin.zip")
    parser.add_argument("--out", type=str, default="models/chase_rays_twin.pt",
                        help="deploy export stem (suffix ignored); TorchScript by default")
    add_format_arg(parser)
    parser.set_defaults(format="torchscript")   # Godot-free path: ONNX-free by default
    args = parser.parse_args()

    # SubprocVecEnv gives real parallelism (chase's Python step is GIL-bound); DummyVecEnv for n=1.
    vec_cls = SubprocVecEnv if args.n_envs > 1 else None
    env = make_vec_env(ChaseRaysTwinEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)

    model = PPO(
        "MlpPolicy", env,
        n_steps=256, batch_size=256, gae_lambda=0.95, gamma=0.99,
        n_epochs=10, ent_coef=0.01, learning_rate=3e-4, verbose=1, seed=args.seed,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    dt = time.time() - t0
    print("Trained %d steps in %.1fs (%.0f steps/s, %d envs, no Godot)"
          % (args.timesteps, dt, args.timesteps / max(dt, 1e-9), args.n_envs))

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    for p in export_policy(model, args.out, args.format):
        print("Exported deploy model:", p)

    env.close()


if __name__ == "__main__":
    main()
