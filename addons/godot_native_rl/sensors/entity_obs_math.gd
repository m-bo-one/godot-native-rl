class_name EntityObsMath
extends RefCounted

# Pure helpers for the variable-length entity observation block (#46). No scene/physics deps, fully
# headless-unit-testable. An EntitySensor encodes up to N NEAREST entities, each F floats, into a
# FIXED-width flat vector: [N*F entity features (zero-padded)] followed by [N presence flags]
# (1.0 real / 0.0 pad). Variable entity count rides in the flags, NOT in the vector length, so the
# policy input width is stable and the attention encoder derives its mask from the flags.

# Total floats: N entity rows of F features + N presence flags.
static func obs_size(n_max: int, feat: int) -> int:
	return n_max * feat + n_max

# Build the flat block. `entities` is an Array of Dictionaries {"dist": float, "feat": Array}.
# Sorted ascending by dist (stable on ties via original index), capped to the nearest n_max,
# zero-padded to n_max rows, with an appended parallel presence-flag tail. Returns n_max*(feat+1)
# floats. A feat row of the wrong length is defensively pad/truncated so the width is always exact.
static func build_obs(entities: Array, n_max: int, feat: int) -> Array:
	var indexed: Array = []
	for i in range(entities.size()):
		indexed.append({"i": i, "e": entities[i]})
	indexed.sort_custom(func(a, b):
		var da: float = float(a["e"].get("dist", 0.0))
		var db: float = float(b["e"].get("dist", 0.0))
		if da == db:
			return int(a["i"]) < int(b["i"])
		return da < db)
	var rows: Array = []
	var flags: Array = []
	var kept := mini(indexed.size(), n_max)
	for k in range(n_max):
		if k < kept:
			var row: Array = indexed[k]["e"].get("feat", [])
			rows.append_array(_fit(row, feat))
			flags.append(1.0)
		else:
			rows.append_array(_zeros(feat))
			flags.append(0.0)
	rows.append_array(flags)
	return rows

static func _zeros(n: int) -> Array:
	var out: Array = []
	out.resize(n)
	out.fill(0.0)
	return out

# Resize a feature row to exactly `feat` floats (pad with 0.0 / truncate).
static func _fit(row: Array, feat: int) -> Array:
	var out: Array = []
	for i in range(feat):
		out.append(float(row[i]) if i < row.size() else 0.0)
	return out
