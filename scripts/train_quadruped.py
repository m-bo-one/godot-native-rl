#!/usr/bin/env python3
"""Train the quadruped-walk agent with Stable-Baselines3 PPO (continuous Box action) over the
godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_quadruped.sh for orchestration.

The action space is one continuous key ("motors", size 8 = the hinge motor targets). We trace the
deterministic actor (the action MEAN for a Box policy) to TorchScript, which export_to_ncnn.py
converts unchanged. The std is exported separately by scripts/export_action_dist.py for deploy-side
DiagGaussian sampling.

Export is TorchScript, NOT godot_rl's export_model_as_onnx (mirrors train_fly_by.py — see that file
+ docs/dev/gotchas.md for the ONNX-free rationale).

Locomotion note: reward shaping (motor_max_speed, joint limits, upright/energy/fall weights on
QuadrupedAgent) and the sample budget are tuned during the actual training run (PR2). The PPO
hyper-parameters here are a sane continuous-control starting point, not a converged recipe.
"""
import argparse
import pathlib
import sys

# Reuse the deterministic-actor TorchScript tracer (import stays light at module load: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ent_coef", type=float, default=0.0,
                   help="PPO entropy coefficient; >0 encourages exploration (helps the hurdles task "
                        "escape the deterministic local optimum a warm-started policy collapses into)")
    p.add_argument("--save_model_path", type=str, default="models/quadruped_walk.zip")
    p.add_argument("--pt_export_path", type=str, default="models/quadruped_walk.pt")
    # Periodic checkpointing: save an SB3 .zip every N total env-steps so you can watch the
    # creature at different learning stages (early flailing → late walking) — the #60 epic's
    # "generation race" (Milestone 4). 0 = off (only the final model is saved).
    p.add_argument("--checkpoint_freq", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="models/quadruped_walk_ckpts")
    # Warm-start: initialize the new policy from a previously trained SB3 .zip (e.g. train the
    # hurdles runner from the walk runner). Layers whose shapes match are copied; a wider input
    # layer (hurdles' 35-dim obs = walk's 29 + 6 hurdle rays) keeps the source columns and ZEROES
    # the new ones, so the net starts behaving exactly like the source (ignoring the rays) and then
    # learns to use them. Hugely more sample-efficient than training locomotion from scratch (#252).
    p.add_argument("--init_from", type=str, default="",
                   help="path to a source SB3 .zip to warm-start the policy from (shape-tolerant)")
    return p.parse_args(argv)


def remap_widen_input(src_sd: dict, tgt_sd: dict) -> dict:
    """Pure state-dict remap for warm-starting across a WIDER input layer. Returns a new dict over
    the target's keys: copy the source tensor where shapes match; for a 2D weight that only grew
    along the input dim (same out features, more in features) copy the source columns and zero the
    new ones; otherwise keep the target's (freshly initialized) tensor. Torch-free — operates on
    anything with `.shape`, `.clone()` and column assignment, so it's unit-testable with fakes."""
    out = {}
    for key, tgt in tgt_sd.items():
        src = src_sd.get(key)
        if src is None:
            out[key] = tgt
        elif tuple(src.shape) == tuple(tgt.shape):
            out[key] = src
        elif (len(src.shape) == 2 and len(tgt.shape) == 2
              and src.shape[0] == tgt.shape[0] and src.shape[1] < tgt.shape[1]):
            w = tgt.clone()
            w[:, :src.shape[1]] = src
            w[:, src.shape[1]:] = 0.0  # new inputs start ignored -> behaves like the source net
            out[key] = w
        else:
            out[key] = tgt  # unhandled mismatch -> keep target init
    return out


def warm_start_policy(model, src_path: str) -> None:
    """Load a source SB3 .zip and copy its policy weights into `model` via remap_widen_input."""
    from stable_baselines3 import PPO

    # SB3 re-appends ".zip", so an explicit "...zip" path becomes "...zip.zip" — strip it first.
    clean = src_path[:-4] if src_path.endswith(".zip") else src_path
    src = PPO.load(clean, device="cpu")
    new_sd = remap_widen_input(src.policy.state_dict(), model.policy.state_dict())
    model.policy.load_state_dict(new_sd)
    print("Warm-started policy from %s (shape-tolerant copy)" % src_path)


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from export_torchscript import export_policy_as_torchscript

    args = parse_args()

    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Continuous control over the tiled ParallelArena (godot-rl auto-detects n_agents from the
    # handshake and vectorizes). A larger rollout trains locomotion more reliably than the chase
    # defaults. Do NOT pass seed= to PPO (the env's seed() raises NotImplementedError).
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=args.ent_coef,
        learning_rate=3e-4,
        tensorboard_log="logs/sb3",
    )

    # CheckpointCallback's save_freq is in *callback calls* = vectorized steps, so divide the
    # requested total-env-step cadence by the number of parallel agents (auto-detected).
    callback = None
    if args.checkpoint_freq > 0:
        n_envs = getattr(env, "num_envs", 1) or 1
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1),
            save_path=args.checkpoint_dir,
            name_prefix="quadruped_walk",
        )
        print("Checkpointing every %d env-steps to %s (n_envs=%d)"
              % (args.checkpoint_freq, args.checkpoint_dir, n_envs))

    if args.init_from:
        warm_start_policy(model, args.init_from)

    model.learn(args.timesteps, callback=callback)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    _, sidecar = export_policy_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = action mean) to:", pt_path)
    print("Wrote shape sidecar:", sidecar)
    print("Convert to ncnn with: export_to_ncnn.py %s" % pt_path)

    env.close()


if __name__ == "__main__":
    main()
