# CameraSensor grayscale deploy + real-render check (#36)

**Status:** approved-by-implementer (autonomous batch)
**Issue:** [#36](https://github.com/minigraphx/godot-native-rl/issues/36) (deferred from #8 + #36-image)
**Date:** 2026-07-07

## Problem

`run_inference_image` force-converts every frame to `FORMAT_RGB8` + `PIXEL_RGB` (3 channels), so a
policy trained on a **grayscale** `CameraSensor` (which captures `FORMAT_L8`, 1 channel) cannot
deploy — the controller feeds it a 3-channel input the 1-channel net rejects. Plus (a) headless can't
render viewports, so the `viewport.get_texture().get_image()` capture path was never exercised with
a real render.

## Design

### (b) C++ grayscale deploy path — the core

- `NcnnRunner::run_inference_image(image, normalize=true, grayscale=false)` gains a `grayscale`
  param. When true: convert to `FORMAT_L8`, `ncnn::Mat::from_pixels(..., PIXEL_GRAY, ...)`, and
  1-channel `substract_mean_normalize`. Default `false` = the unchanged 3-channel RGB path.
- **Controller auto-detects** (deepest fix — no new flag to plumb): `NcnnControllerCore` passes
  `grayscale = (img.get_format() == Image.FORMAT_L8)`. A grayscale CameraSensor produces `L8`, so a
  1-channel policy just works; RGB frames stay the default. Also fixes the debug overlay's channel
  count (`c=1` for gray).
- **Fixture + golden:** `make_synthetic_cnn.py --grayscale` (parametrized — RGB default unchanged)
  writes `models/synthetic_cnn_gray.ncnn.*` + `_golden.json` (1-channel Conv, fixed 8×8 L8 image).
  `test/unit/test_image_inference_gray_golden.gd` loads it, runs
  `run_inference_image(l8_img, true, true)`, asserts parity vs the onnxruntime golden (atol 1e-2,
  argmax exact) — the authoritative test of the C++ PIXEL_GRAY path. Auto-discovered by `run_tests.sh`.

### (a) Real-render check (verification, xvfb)

Headless can't render viewports, but `xvfb-run` gives Godot a virtual display. A small
non-`--headless` scene renders a `SubViewport` with a known pattern and asserts
`CameraSensor.get_image()` (→ `viewport.get_texture().get_image()`) returns a non-empty image of the
expected size/format. Run here via `xvfb-run` to close the deferred "never rendered a viewport"
concern. **Not wired as a permanent CI test** — the headless CI has no display, and viewport-render
pixel determinism across renderers is not worth a flaky gate; the value is the one-time confirmation
+ the grayscale golden (which IS the deploy contract).

### (c) render_size override — deferred

Out of scope for this pass (optional in the issue); the grayscale deploy path is the concrete gap.
**Tracked as #362** so the deferred slice is not lost when #36 closes.

## Testing

- `test_image_inference_gray_golden.gd` — the C++ PIXEL_GRAY parity gate (CI).
- Rebuild the extension (`bin/` gitignored; C++ ABI changed → `template_debug` + `template_release`).
- xvfb real-render verification (manual, documented).
- Full `run_tests.sh` green.

## Files

| File | Change |
|---|---|
| `src/ncnn_runner.{h,cpp}` | `grayscale` param + `PIXEL_GRAY`/`L8` branch |
| `addons/godot_native_rl/controllers/ncnn_controller_core.gd` | auto-detect `L8` → grayscale |
| `scripts/make_synthetic_cnn.py` | `--grayscale` variant |
| `models/synthetic_cnn_gray.ncnn.*`, `_golden.json` | **new** fixtures |
| `test/unit/test_image_inference_gray_golden.gd` | **new** golden test |
| `CLAUDE.md` | note grayscale deploy |

Closes #36 (b) + (a); (c) render_size tracked as #362.
