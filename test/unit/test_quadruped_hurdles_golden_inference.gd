extends SceneTree
# Golden inference regression for the shipped quadruped_hurdles ncnn model (#60 M2 — continuous
# control: 8 hinge-motor action means from a 35-dim obs = M1's 29 + 6 hurdle closeness rays).
# Same contract as the M1 golden: fixed obs -> recorded outputs within tolerance. Retrained
# policy? Flip RECORD=true, rerun, paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/quadruped_walk/models/quadruped_hurdles.ncnn.param"
const BIN := "res://examples/quadruped_walk/models/quadruped_hurdles.ncnn.bin"

const RECORD := false  # true -> print outputs for the OBS cases instead of asserting

# Fixed 35-dim observation cases (joints+vels+up+localvel+dir+contacts+rays). Arbitrary but
# fixed; the last 6 slots are the hurdle rays (0 = none in range .. ~1 = hurdle close).
const OBS: Array = [
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.6,  1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	[0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2,  0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3,  0.05, 0.98, 0.0,  0.4, 0.0, 0.1,  0.0, 0.0, 0.55,  1.0, 0.0, 1.0, 0.0,  0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
	[-0.3, 0.4, -0.2, 0.5, -0.3, 0.4, -0.2, 0.5,  -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6,  -0.1, 0.95, 0.05,  0.6, 0.1, 0.2,  0.0, 0.0, 0.5,  0.0, 1.0, 0.0, 1.0,  0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
]

# Recorded baseline (8 action means per OBS case) from the in-lane 6M-step net that clears 6/6 ramped
# hurdles (tile-offset-fix retrain, #252).
const GOLDEN: Array = [
	[-0.3037109375, -0.324951171875, -0.0391845703125, 1.0205078125, -1.9375, -3.501953125, -3.296875, 1.255859375],
	[-0.626953125, -0.61865234375, 0.87646484375, 2.4296875, -1.734375, -2.423828125, -2.103515625, 0.80419921875],
	[0.2462158203125, -0.6572265625, 0.98583984375, 0.2890625, -0.414794921875, -3.01171875, -3.0859375, -1.4130859375],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "quadruped hurdles model loads")
	if ok:
		for i in range(OBS.size()):
			var out := runner.run_inference(PackedFloat32Array(OBS[i]))
			if RECORD:
				print("GOLDEN_%d = %s" % [i, JSON.stringify(Array(out))])
				continue
			var golden: Array = GOLDEN[i]
			h.assert_eq(out.size(), golden.size(), "case %d output size (8 motor means)" % i)
			var close := true
			for j in range(min(out.size(), golden.size())):
				# rtol 1e-2 + atol floor 1e-2: baseline recorded on macOS arm64, CI is Linux x86
				# (same tolerance rationale as the M1 golden — CI-proven cross-platform).
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
