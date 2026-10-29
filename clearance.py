# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
clearance.py -- The depth range d_t of the braking envelope (preprint_261026.pdf, Sec. 3.6, Eqs. 27-28).

"Let d_t be the smallest range measured in an image region R_t whose back-projection covers the volume swept by
a sphere of radius r_s >= r_v moving along the commanded direction u_t = v_{t,0} / ||v_{t,0}||_2 out to the
sensor range. If u_t lies outside the field of view, or if the fraction of invalid depth pixels in R_t exceeds a
threshold, the speed is limited to a constant v_blind."

``DepthRange`` evaluates this on one depth image with the forward pinhole camera of ``sim/models/aero_vpa_rgbd``
(160 x 128, HFOV 1.274 rad, z-depth along the optical axis = body x):

    * every valid pixel is back-projected to P = z r (r the pixel ray in the body FRD frame, x-component 1);
    * the swept volume is the half-cylinder of radius r_s around the ray along u_t from the camera; d_t is the
      smallest Euclidean range ||P|| of the measured points inside it (along-track >= 0, lateral distance <= r_s),
      or the sensor range when none is: no return inside the region means nothing within range;
    * u_t is v_{t,0}, the translational part of the first knot of the projected chunk (heading frame), rotated
      into the body frame with the attitude of the capture; when it is slower than ``min_speed`` the fastest knot
      gives the direction, and a chunk that barely moves needs no range (d_t = sensor range, not blind);
    * blind: u_t outside the field of view (behind the image plane or projecting outside the image), or more than
      ``max_invalid_fraction`` of the pixels in R_t without a valid return. R_t for this count is the cone of
      directions within asin(r_s / reference_range) of u_t, i.e. the pixels that see the swept volume beyond
      ``reference_range``.

Choices the paper leaves open, as implemented here: the reference range of the invalid-pixel count (2 m) and the
threshold. The default threshold 1.0 disables the invalid-pixel rule, because the Gazebo depth camera reports
everything beyond its range (sky, distant ground) as invalid; on a real stereo sensor, set it (e.g. 0.5).

The camera offset from the vehicle center (0.13 m forward) is neglected, which shortens d_t by at most that
amount: the conservative side.

    python clearance.py      # self-test
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["DepthRange"]


class DepthRange:
    """d_t (and the blind flag) of Sec. 3.6 for one forward depth camera.

    Args:
        width, height, hfov: pinhole camera (pixels, horizontal field of view in rad).
        safety_radius: r_s (m), the radius of the swept sphere (config.safety_radius).
        sensor_range: (near, far) of valid depth readings (m); d_t is at most ``far``.
        max_invalid_fraction: blind above this fraction of invalid pixels in R_t (1.0 disables the rule).
        reference_range: range (m) beyond which R_t is counted for the invalid fraction.
        min_speed: knots slower than this (m/s) give no direction.
    """

    def __init__(self, width: int = 160, height: int = 128, hfov: float = 1.274, safety_radius: float = 0.5,
                 sensor_range: Tuple[float, float] = (0.2, 19.1), max_invalid_fraction: float = 1.0,
                 reference_range: float = 2.0, min_speed: float = 0.1) -> None:
        if not (0.0 < hfov < math.pi and width > 0 and height > 0):
            raise ValueError("invalid camera")
        if not (safety_radius > 0.0 and 0.0 < sensor_range[0] < sensor_range[1] and reference_range > 0.0):
            raise ValueError("safety_radius, sensor_range and reference_range must be positive and ordered")
        if not 0.0 <= max_invalid_fraction <= 1.0:
            raise ValueError("max_invalid_fraction must lie in [0, 1]")
        self.width, self.height = int(width), int(height)
        self.f = 0.5 * width / math.tan(0.5 * hfov)
        u = np.arange(width) + 0.5 - 0.5 * width
        v = np.arange(height) + 0.5 - 0.5 * height
        uu, vv = np.meshgrid(u, v)
        self.rays = np.stack([np.ones_like(uu), uu / self.f, vv / self.f], axis=-1)       # [H, W, 3] FRD, x = 1
        self.unit_rays = self.rays / np.linalg.norm(self.rays, axis=-1, keepdims=True)
        self.radius = float(safety_radius)
        self.near, self.far = float(sensor_range[0]), float(sensor_range[1])
        self.max_invalid_fraction = float(max_invalid_fraction)
        self.cone_cos = math.cos(math.asin(min(1.0, self.radius / float(reference_range))))
        self.min_speed = float(min_speed)

    def direction(self, chunk_heading: np.ndarray) -> Optional[np.ndarray]:
        """u_t in the heading frame ``[3]`` from the chunk ``[H, >=3]``, or None when the chunk barely moves."""
        v = np.asarray(chunk_heading, dtype=np.float64)[:, :3]
        speeds = np.linalg.norm(v, axis=1)
        k = 0 if speeds[0] >= self.min_speed else int(np.argmax(speeds))
        if speeds[k] < self.min_speed:
            return None
        return v[k] / speeds[k]

    def evaluate(self, depth_m: np.ndarray, rotation_body_to_ned: np.ndarray, yaw: float,
                 chunk_heading: np.ndarray) -> Tuple[float, bool]:
        """(d_t in m, blind) for a depth image ``[H, W]`` (m; non-finite or <= 0: no return) and a chunk ``[H, 4]``."""
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != (self.height, self.width):
            raise ValueError(f"expected a {self.height}x{self.width} depth image, got {depth.shape}")
        u_h = self.direction(chunk_heading)
        if u_h is None:
            return self.far, False
        c, s = math.cos(yaw), math.sin(yaw)
        u_ned = np.array([c * u_h[0] - s * u_h[1], s * u_h[0] + c * u_h[1], u_h[2]])
        u = np.asarray(rotation_body_to_ned, dtype=np.float64).T @ u_ned                 # body FRD
        if u[0] <= 1e-6:
            return self.far, True                                                         # behind the image plane
        px, py = self.f * u[1] / u[0] + 0.5 * self.width, self.f * u[2] / u[0] + 0.5 * self.height
        if not (0.0 <= px < self.width and 0.0 <= py < self.height):
            return self.far, True                                                         # outside the field of view
        valid = np.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        if self.max_invalid_fraction < 1.0:
            cone = (self.unit_rays @ u) >= self.cone_cos
            if cone.any() and float((~valid[cone]).mean()) > self.max_invalid_fraction:
                return self.far, True
        points = np.where(valid, depth, 0.0)[..., None] * self.rays                      # [H, W, 3]
        along = points @ u
        range_sq = (points ** 2).sum(axis=-1)
        inside = valid & (along >= 0.0) & (range_sq - along ** 2 <= self.radius ** 2)
        if not inside.any():
            return self.far, False
        return float(min(self.far, math.sqrt(float(range_sq[inside].min())))), False

    def for_step(self, depth_m: np.ndarray, rotation_body_to_ned: np.ndarray, yaw: float
                 ) -> Callable[[Tensor], Tuple[Tensor, Tensor]]:
        """The ``clearance`` callable of ``VPAInferencePipeline.step`` for one frame (batch of one)."""
        def clearance(chunk: Tensor) -> Tuple[Tensor, Tensor]:
            rows = [self.evaluate(depth_m, rotation_body_to_ned, yaw, c.detach().cpu().numpy()) for c in chunk]
            return (torch.tensor([r[0] for r in rows], dtype=chunk.dtype, device=chunk.device),
                    torch.tensor([r[1] for r in rows], dtype=torch.bool, device=chunk.device))
        return clearance


def _self_test() -> None:
    dr = DepthRange()
    level = np.eye(3)                                                       # body = NED, yaw 0
    forward = np.zeros((16, 4))
    forward[:, 0] = 3.0
    depth = np.full((128, 160), np.inf)
    # nothing in range: the sensor range, not blind
    assert dr.evaluate(depth, level, 0.0, forward) == (19.1, False)
    # a wall 5 m ahead: d_t = 5 m (the pixel on the optical axis is at z = 5 m, range ~5 m)
    wall = np.full((128, 160), 5.0)
    d, blind = dr.evaluate(wall, level, 0.0, forward)
    assert not blind and 5.0 <= d < 5.01, d
    # a pole 0.3 m right of the path is inside r_s = 0.5 m; at 1 m right it is outside
    for lateral, inside in ((0.3, True), (1.0, False)):
        img = np.full((128, 160), np.inf)
        col = int(round(dr.f * lateral / 4.0 + 80 - 0.5))
        img[:, col] = 4.0
        d, _ = dr.evaluate(img, level, 0.0, forward)
        assert (d < 19.1) == inside and (not inside or abs(d - math.hypot(4.0, (col + 0.5 - 80) / dr.f * 4.0)) < 0.2), (lateral, d)
    # yaw: flying north-east with yaw 45 degrees is straight ahead for the camera
    ne = np.zeros((16, 4))
    ne[:, 0] = 3.0                                                          # heading frame: forward
    rot45 = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0], [math.sin(math.pi / 4), math.cos(math.pi / 4), 0],
                      [0, 0, 1]])
    assert dr.evaluate(wall, rot45, math.pi / 4, ne)[0] < 5.01
    # sideways or backwards: outside the field of view -> blind
    side = np.zeros((16, 4))
    side[:, 1] = 3.0
    assert dr.evaluate(wall, level, 0.0, side)[1] and dr.evaluate(wall, level, 0.0, -forward)[1]
    # a chunk that barely moves: no direction, no limit
    assert dr.evaluate(wall, level, 0.0, np.zeros((16, 4))) == (19.1, False)
    # slow first knot: the fastest knot gives the direction
    accel = forward.copy()
    accel[0, :3] = 0.0
    assert dr.evaluate(wall, level, 0.0, accel)[0] < 5.01
    # invalid-pixel rule (enabled): no returns in the cone -> blind; disabled by default
    strict = DepthRange(max_invalid_fraction=0.5)
    assert strict.evaluate(depth, level, 0.0, forward)[1] and not dr.evaluate(depth, level, 0.0, forward)[1]
    # ground 2 m below: a 25 degree descent (inside the 31 degree half field of view) meets it at 4.7 m on the
    # path; the 0.5 m sphere touches it earlier, from about 3.4 m
    ground = np.full((128, 160), np.inf)
    rows = np.arange(128) + 0.5 - 64
    with np.errstate(divide="ignore"):
        z = np.where(rows > 0, 2.0 * dr.f / rows, np.inf)                 # flat ground 2 m below the camera
    ground[:] = z[:, None]
    dive = np.zeros((16, 4))
    dive[:, 0], dive[:, 2] = 2.0, 2.0 * math.tan(math.radians(25.0))
    d, blind = dr.evaluate(ground, level, 0.0, dive)
    assert not blind and 3.3 < d < 4.8, d
    steep = np.zeros((16, 4))
    steep[:, 0], steep[:, 2] = 2.0, 2.0                                        # 45 degrees: below the image
    assert dr.evaluate(ground, level, 0.0, steep)[1]
    assert dr.evaluate(ground, level, 0.0, forward)[0] == 19.1               # level flight 2 m up: clear
    # the pipeline callable: a batch of one
    fn = dr.for_step(wall, level, 0.0)
    d_t, b_t = fn(torch.tensor(forward, dtype=torch.float32)[None])
    assert d_t.shape == (1,) and b_t.dtype == torch.bool and float(d_t[0]) < 5.01
    print("clearance.py self-test passed")


if __name__ == "__main__":
    _self_test()
