extends SceneTree
# Golden inference regression for the two multi-policy hide & seek ncnn models
# (scripts/train_hide_seek_multipolicy.py -> export_to_ncnn.py --via torchscript). Loads each model
# via NcnnRunner and asserts run_discrete_action() returns the captured argmax for 5 fixed
# observations. ncnn<->torch.jit parity (50/50 argmax) was verified at conversion time by
# export_to_ncnn.py. If this fails after a retrain/model swap, recapture the goldens from the new
# models and update them here.
#
# obs is 15 floats; index 14 is the role flag (seeker=1.0, hider=0.0). Each policy was trained only
# on its own role's observations, so it is probed with its role flag set accordingly.
#
# The obs are chosen so BOTH policies decide with a comfortable top-2 logit margin (>= ~0.86 here):
# a near-tie argmax can flip between CPU architectures (mac arm64 capture vs Linux x86_64 CI) under
# ncnn's float math, which made an earlier near-tie golden (margin 0.02) fail only on CI (#241).

const Harness = preload("res://test/harness.gd")

# Obs chosen so BOTH nets decide with a comfortable top-2 margin (>= 0.92 here) — re-picked for the
# #274 reduced-walls + dense-reward retrain (the prior obs gave the new nets near-tie 0.16 margins
# that would flip across CPU architectures, the #241 failure mode).
const OBS: Array = [
	[-0.35,0.18,-0.83,-0.97,-0.05,-0.31,0.62,-0.91,-0.39,-0.36,-0.57,-0.06,0.93,-0.73, 1.0],
	[0.05,0.54,-0.69,0.25,-0.21,0.12,0.47,0.0,-0.59,-0.93,-0.99,-0.83,-0.95,0.97, 1.0],
	[-0.69,0.93,0.65,0.59,0.73,0.38,0.83,-0.16,0.28,-0.25,0.15,0.58,-0.95,-0.05, 1.0],
	[0.12,-0.99,-0.52,-0.79,0.75,0.56,0.59,0.66,0.0,0.34,-0.8,0.46,-0.03,-0.69, 1.0],
	[-0.96,0.21,-0.3,-0.61,-0.9,0.95,-0.56,-0.55,0.3,0.78,0.39,0.6,0.96,-0.77, 1.0],
]
const EXPECTED_SEEKER: Array = [2, 3, 2, 2, 2]  # captured from the real ncnn deploy path (role 1.0)
const EXPECTED_HIDER: Array  = [0, 4, 0, 0, 3]  # captured from the real ncnn deploy path (role 0.0)

func _check(h, tag: String, base: String, expected: Array, role_flag: float) -> void:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
		ProjectSettings.globalize_path(base + ".bin"))
	h.assert_true(ok, "%s model loads" % tag)
	if ok:
		for i in range(OBS.size()):
			var o: Array = OBS[i].duplicate()
			o[14] = role_flag
			var got := runner.run_discrete_action(PackedFloat32Array(o))
			h.assert_eq(got, int(expected[i]), "%s golden argmax #%d" % [tag, i])
	runner.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, "seeker", "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn", EXPECTED_SEEKER, 1.0)
	_check(h, "hider", "res://examples/hide_and_seek/models/hide_seek_hider.ncnn", EXPECTED_HIDER, 0.0)
	h.finish(self)
