#!/usr/bin/env python3
"""Deploy-format dispatch for the SB3 training scripts (#52).

`--format {onnx,torchscript,both}` on the trainers routes a finished run's deterministic actor to
ONNX (godot_rl `export_model_as_onnx`) and/or TorchScript (`.pt` + `.pt.shape.json` via the shared
`export_torchscript.export_policy_as_torchscript`). Both land as `<stem>.<ext>` and flow unchanged
into `scripts/export_to_ncnn.py`. Default stays `onnx` so the trainers' hot path is unchanged.

Pure `resolve_formats` is torch-free (unit-testable); `export_policy` lazily imports the heavy
exporters only for the formats requested.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

SUPPORTED = ("onnx", "torchscript", "both")


def resolve_formats(fmt: str) -> tuple[str, ...]:
    """Map a --format value to the concrete formats to write. Raises ValueError on an unknown value."""
    if fmt == "onnx":
        return ("onnx",)
    if fmt == "torchscript":
        return ("torchscript",)
    if fmt == "both":
        return ("onnx", "torchscript")
    raise ValueError("unknown export format %r (expected one of %s)" % (fmt, ", ".join(SUPPORTED)))


def export_policy(model, out_stem: str, fmt: str = "onnx") -> list[pathlib.Path]:
    """Export a loaded SB3 model's deterministic actor in the requested format(s).

    `out_stem` is a path with any suffix stripped (e.g. models/chase_policy). Returns the written
    artifact paths (the `.onnx` and/or `.pt`; the `.pt.shape.json` sidecar rides alongside the `.pt`).
    """
    stem = pathlib.Path(out_stem).with_suffix("")
    written: list[pathlib.Path] = []
    formats = resolve_formats(fmt)
    if "onnx" in formats:
        from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
        onnx_path = stem.with_suffix(".onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_model_as_onnx(model, str(onnx_path))
        written.append(onnx_path)
    if "torchscript" in formats:
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from export_torchscript import export_policy_as_torchscript
        pt_path, _sidecar = export_policy_as_torchscript(model, stem.with_suffix(".pt"))
        written.append(pt_path)
    return written


def add_format_arg(parser) -> None:
    """Add the shared --format flag to an argparse parser (default onnx, behavior unchanged)."""
    parser.add_argument("--format", choices=SUPPORTED, default="onnx",
                        help="deploy export format: onnx (default), torchscript (.pt + shape "
                             "sidecar, ONNX-free), or both")


def _print_written(written: Sequence[pathlib.Path]) -> None:
    for p in written:
        print("Exported deploy model:", p)
