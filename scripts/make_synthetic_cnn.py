"""Generate a tiny seeded CNN and an ncnn golden fixture for image-inference tests.

Run under .venv-train (torch + onnxruntime + ncnn; shells out to .venv pnnx via
scripts/export_to_ncnn.py). Writes models/<stem>.ncnn.{param,bin} and
models/<stem>_golden.json (a fixed 8x8 image + golden logits/argmax) used by
test/unit/test_image_inference_golden.gd.

Two variants share this generator (#36):
  * RGB (default, stem `synthetic_cnn`): 3-channel PIXEL_RGB.
  * `--grayscale` (stem `synthetic_cnn_gray`): 1-channel PIXEL_GRAY / FORMAT_L8 deploy path.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_cnn.py [--grayscale]

NOTE (#319): no regenerate-and-diff CI gate exists for this generator (it needs the
torch venv, unlike make_synthetic_attention.py which is byte-pinned in CI). If you edit
it, rerun it and commit the regenerated models/ artifacts in the SAME change, or the
committed fixtures silently drift from the generator.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WIDTH = HEIGHT = 8
N_ACTIONS = 4
SEED = 42


class TinyCNN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * HEIGHT * WIDTH, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def fixed_image_bytes(channels: int) -> bytes:
    # Deterministic ramp; W*H*channels distinct values, all < 256 (192 for RGB, 64 for gray).
    return bytes(range(WIDTH * HEIGHT * channels))


def to_chw_normalized(img_bytes: bytes, channels: int) -> np.ndarray:
    hwc = np.frombuffer(img_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH, channels)
    chw = hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[None, :, :, :]  # [1, channels, 8, 8]


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--grayscale", action="store_true",
                    help="1-channel PIXEL_GRAY fixture (stem synthetic_cnn_gray) for the #36 deploy path")
    args = ap.parse_args()
    channels = 1 if args.grayscale else 3
    stem = "synthetic_cnn_gray" if args.grayscale else "synthetic_cnn"

    torch.manual_seed(SEED)
    model = TinyCNN(channels).eval()
    MODELS.mkdir(exist_ok=True)

    img_bytes = fixed_image_bytes(channels)
    chw = to_chw_normalized(img_bytes, channels)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / (stem + ".onnx")
        dummy = torch.zeros(1, channels, HEIGHT, WIDTH)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        logits_onnx = np.array(sess.run(None, {in_name: chw})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,%d,8,8]" % channels],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / (stem + ".ncnn.param")
    bin_ = MODELS / (stem + ".ncnn.bin")
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    # Best-effort early cross-check of the C++ deploy path's preprocessing via the ncnn
    # python package. The authoritative parity gate is the GDScript golden test, which runs
    # the real C++ run_inference_image; this block only gives an early signal, so API quirks
    # here must not block generation — hence the try/except.
    try:
        import ncnn
        net = ncnn.Net()
        net.load_param(str(param))
        net.load_model(str(bin_))
        ex = net.create_extractor()
        pixel_type = ncnn.Mat.PixelType.PIXEL_GRAY if args.grayscale else ncnn.Mat.PixelType.PIXEL_RGB
        mat = ncnn.Mat.from_pixels(np.frombuffer(img_bytes, dtype=np.uint8), pixel_type, WIDTH, HEIGHT)
        mat.substract_mean_normalize([], [1.0 / 255.0] * channels)
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        logits_ncnn = np.array(out).reshape(-1)
        max_diff = float(np.max(np.abs(logits_onnx - logits_ncnn)))
        print(f"onnx vs ncnn(python) max abs diff: {max_diff:.5f}")
    except Exception as exc:  # noqa: BLE001 — early signal only; GDScript golden is the gate
        print(f"ncnn(python) cross-check skipped ({exc}); GDScript golden remains the gate")

    golden = {
        "width": WIDTH,
        "height": HEIGHT,
        "channels": channels,
        "grayscale": bool(args.grayscale),
        "image_bytes": list(img_bytes),
        "logits": [float(x) for x in logits_onnx],
        "argmax": int(np.argmax(logits_onnx)),
    }
    (MODELS / (stem + "_golden.json")).write_text(json.dumps(golden, indent=2))
    print("golden logits:", golden["logits"], "argmax:", golden["argmax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
