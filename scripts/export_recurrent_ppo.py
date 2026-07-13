#!/usr/bin/env python3
"""Export an sb3-contrib RecurrentPPO checkpoint's LSTM actor to ncnn multi-IO + sidecar (#378).

The deploy graph is the actor half only: obs -> lstm_actor -> mlp_extractor.policy_net ->
action_net, wrapped as forward(obs, h, c) -> (logits, h', c') and exported via legacy ONNX
(opset 13) -> pnnx -> ncnn. pnnx preserves the LSTM state blobs as extra graph inputs/outputs
(in1/in2 -> out1/out2 — verified by the #378 spike and by scripts/make_synthetic_lstm.py), which
is exactly the `<stem>.recurrent.json` contract `NcnnControllerCore` consumes: the controller
feeds the carried state back each decision and zero-inits it at episode boundaries (matching
RecurrentPPO's episode_starts semantics game-side).

Deterministic-actor export: logits for Discrete (argmax game-side), the mean for continuous.
Single-layer LSTM only — a stacked LSTM would decompose into multiple ncnn LSTM layers with a
different state-blob layout, so we fail loud instead of writing a wrong sidecar.

Parity is verified the way the net will actually run: a random obs SEQUENCE replayed through
ncnn(python) with hidden state carried across steps, against the torch wrapper (and the wrapper
itself is asserted against model.predict(deterministic=True) so it provably IS the policy).

Usage:
  .venv-train/bin/python scripts/export_recurrent_ppo.py --checkpoint models/chase_memory.zip \
      [--outdir models] [--stem chase_memory] [--atol 0.01] [--seq 8]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_actor(policy):
    """The deterministic RecurrentPPO actor as an (obs, h, c) -> (out, h', c') module."""
    import torch.nn as nn

    class RecurrentActor(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.lstm = p.lstm_actor
            self.mlp = p.mlp_extractor.policy_net
            self.action_net = p.action_net

        def forward(self, obs, h_in, c_in):
            out, (h_out, c_out) = self.lstm(obs, (h_in, c_in))
            return self.action_net(self.mlp(out)), h_out, c_out

    return RecurrentActor(policy)


def export_recurrent_actor(checkpoint: str, outdir: Path, stem: str,
                           atol: float = 1e-2, seq_len: int = 8) -> list[Path]:
    """Export + carried-sequence parity check. Returns [param, bin, sidecar] paths."""
    import numpy as np
    import torch
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(checkpoint, device="cpu")
    policy = model.policy.eval()
    lstm = policy.lstm_actor
    if lstm.num_layers != 1:
        raise SystemExit(
            f"export_recurrent_ppo: n_lstm_layers={lstm.num_layers} unsupported — a stacked LSTM "
            "decomposes into multiple ncnn LSTM layers with a different state-blob layout. "
            "Retrain with n_lstm_layers=1 or extend the sidecar writer.")
    obs_dim = lstm.input_size
    hidden = lstm.hidden_size
    actor = build_actor(policy)

    rng = np.random.default_rng(0)
    obs_seq = [rng.standard_normal(obs_dim).astype(np.float32) for _ in range(seq_len)]

    # Torch reference with carried state; cross-check the wrapper against the policy's own
    # predict() (discrete only — continuous predict clips/squashes, the raw mean is the contract).
    h = torch.zeros(1, 1, hidden)
    c = torch.zeros(1, 1, hidden)
    ref = []
    with torch.no_grad():
        for o in obs_seq:
            out, h, c = actor(torch.from_numpy(o).reshape(1, 1, obs_dim), h, c)
            ref.append(out.reshape(-1).numpy().copy())
    import gymnasium.spaces as spaces
    if isinstance(model.action_space, spaces.Discrete):
        state = None
        episode_start = np.ones((1,), bool)
        sb3_actions = []
        for o in obs_seq:
            a, state = model.predict(o, state=state, episode_start=episode_start,
                                     deterministic=True)
            episode_start = np.zeros((1,), bool)
            sb3_actions.append(int(a))
        wrapper_actions = [int(np.argmax(l)) for l in ref]
        if sb3_actions != wrapper_actions:
            raise SystemExit("export_recurrent_ppo: extracted actor disagrees with "
                             f"model.predict ({wrapper_actions} vs {sb3_actions})")

    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / f"{stem}.onnx"
        torch.onnx.export(
            actor,
            (torch.zeros(1, 1, obs_dim), torch.zeros(1, 1, hidden), torch.zeros(1, 1, hidden)),
            str(onnx_path),
            input_names=["obs", "h_in", "c_in"],
            output_names=["out", "h_out", "c_out"],
            opset_version=13, dynamo=False,
        )
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"), str(onnx_path),
             "--outdir", str(outdir), "--skip-verify",
             "--inputshape", f"[1,1,{obs_dim}],[1,1,{hidden}],[1,1,{hidden}]"],
            check=False).returncode
        if rc != 0:
            raise SystemExit("export_recurrent_ppo: export_to_ncnn failed")

    param = outdir / f"{stem}.ncnn.param"
    bin_ = outdir / f"{stem}.ncnn.bin"

    # Carried-sequence parity through ncnn(python) — the single-step pnnx check can't see a
    # broken state loop, this can.
    import ncnn
    net = ncnn.Net()
    net.load_param(str(param))
    net.load_model(str(bin_))
    h_n = np.zeros(hidden, np.float32)
    c_n = np.zeros(hidden, np.float32)
    max_diff = 0.0
    for i, o in enumerate(obs_seq):
        # GOTCHA: ncnn.Mat(np_array) ALIASES the numpy buffer and extraction is lazy — the
        # arrays must stay referenced until after extract(), or freed buffers get silently
        # reused across input blobs (in1's dead buffer reallocated for in2 made the LSTM read
        # the CELL data as the hidden state — a 0.3 parity "failure" that wasn't real).
        obs_buf, h_buf, c_buf = o.copy(), h_n.copy(), c_n.copy()
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(obs_buf))
        ex.input("in1", ncnn.Mat(h_buf))
        ex.input("in2", ncnn.Mat(c_buf))
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        del obs_buf, h_buf, c_buf  # lifetime explicitly covers the extract calls above
        h_n = np.array(out1, np.float32).reshape(-1)
        c_n = np.array(out2, np.float32).reshape(-1)
        max_diff = max(max_diff, float(np.max(np.abs(
            np.array(out0).reshape(-1) - ref[i]))))
    print(f"[export_recurrent_ppo] torch vs ncnn max |diff| over {seq_len}-step "
          f"carried sequence: {max_diff:.6f}")
    if max_diff > atol:
        raise SystemExit(f"export_recurrent_ppo: PARITY FAIL ({max_diff:.6f} > atol {atol})")

    sidecar = {
        "obs_input": "in0",
        "obs_shape": [obs_dim],
        "action_output": "out0",
        "state_pairs": [
            {"in": "in1", "out": "out1", "shape": [hidden]},
            {"in": "in2", "out": "out2", "shape": [hidden]},
        ],
    }
    sidecar_path = outdir / f"{stem}.recurrent.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    for p in (param, bin_, sidecar_path):
        print("[export_recurrent_ppo] wrote:", p)
    return [param, bin_, sidecar_path]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--stem", default="",
                        help="output stem (default: checkpoint filename stem)")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seq", type=int, default=8)
    args = parser.parse_args()
    stem = args.stem or Path(args.checkpoint).stem
    export_recurrent_actor(args.checkpoint, Path(args.outdir), stem,
                           atol=args.atol, seq_len=args.seq)


if __name__ == "__main__":
    main()
