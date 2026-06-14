# Pure, headless-unit-tested helpers for the locomotion race (#60 M4). The race runs several lanes
# in one physics space, each creature driven by a DIFFERENT trained net (e.g. early/mid/late
# training "generations"), and ranks them by forward distance — the learning-arc showcase.

# Rank lanes by forward distance, descending. `distances` is a float per lane (index = lane id).
# Returns lane ids ordered first..last. Stable on ties (lower lane id ranks first). Pure.
static func standings(distances: Array) -> Array:
	var idx: Array = []
	for i in range(distances.size()):
		idx.append(i)
	idx.sort_custom(func(a, b):
		if distances[a] == distances[b]:
			return a < b
		return distances[a] > distances[b])
	return idx

# 1-based finishing place for each lane: place[lane] = its rank (1 = leader). Pure.
static func places(distances: Array) -> Array:
	var order := standings(distances)
	var place: Array = []
	place.resize(distances.size())
	for rank in range(order.size()):
		place[order[rank]] = rank + 1
	return place

# True once a lane reaches the finish line (forward distance >= finish_z). Pure.
static func finished(distance: float, finish_z: float) -> bool:
	return distance >= finish_z

# Format a one-line leaderboard row. Pure (display helper, kept testable).
static func format_row(place: int, label: String, distance: float) -> String:
	return "%d. %s  %.1f m" % [place, label, distance]

# Whole seconds until the next generation switch, rounded UP so a partial second still reads >=1
# while frames remain. Clamps to 0 at/after the deadline and guards a non-positive tick rate. Pure.
static func countdown_seconds(frames_left: int, ticks_per_second: int) -> int:
	if frames_left <= 0 or ticks_per_second <= 0:
		return 0
	return int(ceil(float(frames_left) / float(ticks_per_second)))

# "Now playing" banner for the active generation: which net is running, its position in the lineup,
# how far it has reached, and a live countdown to the auto-switch. Pure (display helper). Makes the
# automatic generation cycling unmistakable — the fix for "it shows 500k only / never switches".
static func format_now(index_1based: int, total: int, label: String, reach_m: float, seconds_left: int) -> String:
	return "▶ %s  (%d/%d)  reach %.1f m  ·  next in %ds" % [label, index_1based, total, reach_m, seconds_left]
