"""Pure-numpy safety rules: robot<->human distance and speed limit (SSM)."""

from __future__ import annotations

import numpy as np


class SafetyStopError(RuntimeError):
    """Raised when a motion is aborted because the robot got too close to the human."""


def capsule_endpoints(center, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the bottom and top points of a vertical capsule axis."""
    c = np.asarray(list(center), dtype=np.float64)  # capsule center (3,)
    offset = np.array([0.0, 0.0, half_height])  # axis is along z (standing human)
    return c - offset, c + offset


def points_to_capsule_distance(points, center, radius: float, half_height: float) -> float:
    """Smallest distance from points (N, 3) to the capsule surface (negative = inside)."""
    a, b = capsule_endpoints(center, half_height)  # capsule axis segment
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))  # (N, 3)
    ab = b - a  # axis direction
    denom = max(float(ab @ ab), 1e-12)  # avoid division by zero for a sphere
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)  # projection of each point on the segment
    closest = a + t[:, None] * ab  # closest point on the axis for each point
    d_axis = np.linalg.norm(p - closest, axis=1)  # distance to the axis
    return float(np.min(d_axis) - radius)  # distance to the surface


def capsule_to_spheres(center, radius: float, half_height: float) -> list[dict]:
    """Approximate the capsule by overlapping spheres (format of the PyRoKi server obstacles)."""
    a, b = capsule_endpoints(center, half_height)
    n = int(np.ceil(2.0 * half_height / radius)) + 1  # enough spheres so they overlap
    return [
        {"type": "sphere", "center": (a + (b - a) * t).tolist(), "radius": float(radius)}
        for t in np.linspace(0.0, 1.0, n)
    ]


def speed_limit(distance: float, cfg) -> float:
    """SSM rule: max joint speed [rad/s] allowed for a given robot<->human clearance."""
    if distance < cfg.min_separation:  # too close -> stop
        return 0.0
    if distance < cfg.slow_zone:  # close -> reduced speed
        return cfg.v_max_reduced
    return cfg.v_max_nominal  # far -> nominal speed
