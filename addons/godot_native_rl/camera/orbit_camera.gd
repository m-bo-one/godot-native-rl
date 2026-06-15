extends Camera3D
# Reusable orbit camera (#265). FOLLOW mode holds a fixed behind-and-above view (the old
# quadruped_camera behavior); ORBIT mode lets the user drag to rotate + scroll to zoom. The pivot
# tracks the creature each frame in both modes. Cosmetic + inert headless (no input -> stays in
# follow). Reads a duck-typed get_camera_pivot() -> Vector3 from the game node.

# --- Pure helpers (unit-testable; no tree/input needed) ---

# World position `distance` from `pivot` at spherical angles. azimuth=elevation=0 -> straight +Z.
static func orbit_position(pivot: Vector3, azimuth: float, elevation: float, distance: float) -> Vector3:
	return pivot + distance * Vector3(
		cos(elevation) * sin(azimuth),
		sin(elevation),
		cos(elevation) * cos(azimuth))

# Inverse of orbit_position: the spherical angles + distance for a camera-relative offset.
static func offset_to_spherical(offset: Vector3) -> Dictionary:
	var d := offset.length()
	if d < 0.0001:
		return {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0}
	return {
		"azimuth": atan2(offset.x, offset.z),
		"elevation": asin(clampf(offset.y / d, -1.0, 1.0)),
		"distance": d}
