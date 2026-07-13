#!/usr/bin/env python3
"""Train the memory chase (blinking target, #378) with sb3-contrib RecurrentPPO (PPO-LSTM).

The env hides the target's obs slots for most decisions (see chase_memory_agent.gd), so the
LSTM must carry where the target was — a feed-forward PPO has nothing to steer by while blind.
Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot training
scene. See scripts/train_chase_memory.sh for orchestration.

sb3-contrib is an opt-in add-on: .venv-train/bin/pip install -r requirements-recurrent.txt
"""
import argparse
import pathlib
import sys

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_recurrent_ppo import export_recurrent_actor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--save_model_path", type=str, default="models/chase_memory.zip")
    parser.add_argument("--outdir", type=str, default="models",
                        help="where the ncnn deploy artifacts land")
    parser.add_argument("--skip_export", action="store_true",
                        help="save the SB3 zip only (export separately via export_recurrent_ppo.py)")
    args = parser.parse_args()

    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # MultiInputLstmPolicy: the godot env exposes Dict{"obs": Box}. Single-layer LSTM — the
    # export path's sidecar writer supports exactly one ncnn LSTM layer (fails loud otherwise).
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.n_steps,  # RecurrentPPO batches whole sequences; keep it simple
        ent_coef=0.005,
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=1,
            net_arch=dict(pi=[32], vf=[32]),
        ),
        tensorboard_log="logs/sb3",
    )
    model.learn(args.timesteps)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved RecurrentPPO model to:", zip_path)
    env.close()

    if not args.skip_export:
        export_recurrent_actor(str(zip_path), pathlib.Path(args.outdir), zip_path.stem)


if __name__ == "__main__":
    main()
