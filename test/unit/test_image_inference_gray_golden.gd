extends SceneTree
# Golden regression for GRAYSCALE native image inference (#36): loads the committed 1-channel
# synthetic CNN and asserts NcnnRunner.run_inference_image(img, true, grayscale=true) matches the
# onnxruntime golden (within atol=1e-2) for a fixed 8x8 FORMAT_L8 image — exercising the C++
# PIXEL_GRAY / FORMAT_L8 deploy path. Regenerate with:
#   .venv-train/bin/python scripts/make_synthetic_cnn.py --grayscale

const Harness = preload("res://test/harness.gd")
const GOLDEN := "res://models/synthetic_cnn_gray_golden.json"
const PARAM := "res://models/synthetic_cnn_gray.ncnn.param"
const BIN := "res://models/synthetic_cnn_gray.ncnn.bin"

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "gray golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	var w := int(data["width"])
	var ht := int(data["height"])
	var img_bytes := PackedByteArray()
	for v in data["image_bytes"]:
		img_bytes.append(int(v))
	# 1-channel L8 image — the format a grayscale CameraSensor captures.
	var img := Image.create_from_data(w, ht, false, Image.FORMAT_L8, img_bytes)

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "grayscale synthetic CNN loads")
	if ok:
		var logits: PackedFloat32Array = runner.run_inference_image(img, true, true)
		var golden: Array = data["logits"]
		h.assert_eq(logits.size(), golden.size(), "logit count matches golden")
		var within := logits.size() == golden.size()
		for i in range(mini(logits.size(), golden.size())):
			if absf(logits[i] - float(golden[i])) > 1e-2:
				within = false
		h.assert_true(within, "grayscale logits within atol 1e-2 of onnxruntime golden")
		var best := 0
		for i in range(1, logits.size()):
			if logits[i] > logits[best]:
				best = i
		h.assert_eq(best, int(data["argmax"]), "argmax matches golden")

	h.finish(self)
