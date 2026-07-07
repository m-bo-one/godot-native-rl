# Integrated `--format` export on training scripts (#52)

**Status:** approved-by-implementer (autonomous batch; gate met)
**Issue:** [#52](https://github.com/minigraphx/godot-native-rl/issues/52)
**Date:** 2026-07-07

## Gate (met)

The issue is gated on "the standalone `export_torchscript.py` is shipped AND its e2e parity is
confirmed equal to the ONNX route." Both hold: `scripts/export_torchscript.py` ships a reusable
`export_policy_as_torchscript(model, pt_path)` (in-memory model → `.pt` + `.pt.shape.json`), and the
TorchScript→ncnn path is the proven deploy route for BallChase SAC / FlyBy / visual_chase / etc.

## Goal & non-goals

- **Goal:** a `--format {onnx,torchscript,both}` option (default `onnx`, behavior unchanged) on the
  SB3 trainers (`train_chase.py`, `train_rover.py`, `train_hide_seek.py`) and `export_checkpoint.py`,
  so a finished run can emit `.pt` + `.pt.shape.json` instead of / alongside ONNX in the hot path.
- **Non-goal:** touching the CleanRL/SF/RLlib/multipolicy trainers (they already export their own
  format); changing the default; any deploy-side change.

## Design

### 1. Shared dispatch helper — `scripts/export_formats.py`

- **Pure** `resolve_formats(fmt: str) -> tuple[str, ...]` — `"onnx"→("onnx",)`,
  `"torchscript"→("torchscript",)`, `"both"→("onnx","torchscript")`; raises `ValueError` on anything
  else. Unit-testable with no torch.
- `export_policy(model, out_stem, fmt="onnx") -> list[Path]` — for each resolved format, writes
  `<out_stem>.onnx` (via godot_rl `export_model_as_onnx`) and/or `<out_stem>.pt` + `.pt.shape.json`
  (via `export_torchscript.export_policy_as_torchscript`). Returns the written paths. Heavy imports
  lazy inside the function. `out_stem` is a path with any suffix stripped.

### 2. Wire into the four scripts

Each already has `--onnx_export_path`. Add `--format {onnx,torchscript,both}` (default `onnx`). At
the export site, replace the direct `export_model_as_onnx(export_model, onnx_path)` call with
`export_policy(export_model, <onnx_export_path without suffix>, args.format)` and print the written
paths. `train_chase.py` keeps its `deploy_export_checkpoint` best-model selection — the chosen
`export_model` is what gets exported in the requested format(s).

### 3. Docs

`CLAUDE.md` train-command lines note the `--format` flag; `README` if it lists the export step.

## Testing (TDD)

- `test/python/test_export_formats.py`:
  - Pure `resolve_formats` cases (onnx/torchscript/both/invalid) — no torch.
  - torch+SB3-gated: build a tiny `PPO("MlpPolicy", DummyBoxEnv)`, call `export_policy` with each
    fmt into a tempdir, assert the right files exist (`.onnx`; `.pt` + `.pt.shape.json`; both).
- A guarded end-to-end is unnecessary (the trainers' existing smokes already run ONNX; the new path
  is the format dispatch + the already-proven torchscript writer). Manually run one trainer with
  `--format torchscript` to confirm the wiring, and the full `run_tests.sh` for regression.

## Files

| File | Change |
|---|---|
| `scripts/export_formats.py` | **new** `resolve_formats` + `export_policy` |
| `scripts/train_chase.py` / `train_rover.py` / `train_hide_seek.py` / `export_checkpoint.py` | `--format` flag + dispatch |
| `test/python/test_export_formats.py` | **new** tests |
| `CLAUDE.md` | note the flag |

Closes #52.
