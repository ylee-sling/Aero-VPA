# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/expert.py -- Privileged expert, labels and HDF5 episode schema for Aero-VPA Phase 1
(preprint_261026.pdf, Secs. 3.1, 3.3, 3.8 and 5.1).

NumPy only (``h5py`` is imported when a file is written), so the ROS 2 collector can import it without
PyTorch. Every quantity that the training code consumes is computed here, once, for the collector, the
offline dry run and the tests:

    * frames:      NED world W, heading frame H_t (W rotated about down by psi_t), body FRD B
    * S_t (Eq. 2): ( R_z(psi)^T v^W,  n^B,  omega^B,  a_bar_t ),  a_bar_t = the setpoint dispatched most
                   recently, re-expressed in the current heading frame (as ``pipeline.proprioception``)
    * g_t (Eq. 3) and the waypoint pointer m_t (Eq. 14), as ``selector.goal_feature`` / ``WaypointTracker``
    * u*_t:        the rule labels of Sec. 3.3, as ``selector.primitive_labels``, on the first expert knot
    * A_t (Eq. 4): the expert's 16 knots, spaced Delta = 20 ms from the capture time tau_t, in H_{tau_t}

The privileged expert (Sec. 5.1) plans on the ground-truth occupancy (``PrivilegedMap``, from the world's
JSON sidecar): A* on a 3-D grid with line-of-sight shortcutting, then a minimum-snap spline [3] through the
via points with the time scaled to the speed and acceleration limits, then refinement by midpoint
insertion until the spline keeps the required clearance. It is tracked by

    v_cmd(t)   = v_ref(t) + k_p (p_ref(t) - p_hat),        a_ff = a_ref(t)
    yaw_cmd(t) = yaw_rate_ref(t) + k_psi wrap(psi_ref(t) - psi_hat)

and the chunk recorded at tau_t is this law evaluated along its own nominal closed loop: the tracking
errors decay as exp(-k s), so knot k is v_ref(tau_t + s) + k_p e_p exp(-k_p s) with s = k Delta. Knot 0
is therefore exactly the command dispatched at tau_t.

HDF5 layout (``SCHEMA_NAME`` / ``SCHEMA_VERSION``; T frames per episode)::

    /                              attrs: schema, schema_version, dimensions, conventions, label rule
    /episode_0000/observations/images    (T, 128, 160, 4) uint16  R, G, B in [0, 255]; D in mm (0 = invalid)
    /episode_0000/observations/state     (T, 13)          float32 S_t (Eq. 2)
    /episode_0000/observations/goal      (T, 4)           float32 g_t (Eq. 3)
    /episode_0000/actions                (T, 16, 4)       float32 A_t (Eq. 4), physical units, frame H_t
    /episode_0000/primitives             (T,)             int64   u*_t in {0, ..., 5}
    /episode_0000/timestamps             (T,)             float64 tau_t (s, simulation time)
    /episode_0000/future_targets         (T,)             int64   frame closest to tau_t + T_nu, -1 if none
    /episode_0000/privileged/...         position, yaw, waypoint index, reference position, odometry age
    /episode_0000                  attrs: complete, success, termination, waypoints_ned, speed_limit, world, ...

An episode is written in blocks while it is flown and marked ``complete`` when it is finalised; readers skip
incomplete groups, and reopening a file for appending deletes them.

``python sim/expert.py`` runs the self-test (and cross-checks against ``selector.py`` when PyTorch is
installed).
"""

from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "PRIMITIVES", "IMAGE_SIZE", "HORIZON", "ACTION_DIM", "PROPRIO_DIM",
    "GOAL_DIM", "KNOT_DT", "PREDICTOR_HORIZON_S", "ACTION_SCALE", "DEPTH_SCALE_M",
    "wrap_angle", "quat_to_rotmat", "yaw_from_quat", "world_to_heading", "heading_to_world",
    "gravity_in_body", "proprioception", "goal_feature", "WaypointTracker", "LabelThresholds",
    "primitive_labels", "Obstacle", "PrivilegedMap", "PlannerConfig", "PlanningError", "MinSnapSpline",
    "ExpertTrajectory", "ExpertPlanner", "load_world", "plan_mission_spline", "CommandLimits", "ExpertController",
    "sample_mission",
    "encode_rgbd", "decode_rgbd", "future_targets", "FrameSample", "DatasetWriter", "EpisodeWriter",
    "PinholeCamera", "KinematicQuadrotor", "DryRunConfig", "dry_run", "record_frame", "synthetic_forest",
]

# ---------------------------------------------------------------------------------------------
# Schema constants (must agree with config.VPAConfig; checked by the self-tests and inspect_hdf5.py)
# ---------------------------------------------------------------------------------------------
SCHEMA_NAME = "aero-vpa/expert-demonstrations"
SCHEMA_VERSION = "1.0.0"
PRIMITIVES: Tuple[str, ...] = (
    "CRUISE", "LATERAL_AVOID", "VERTICAL_AVOID", "YAW_SCAN", "APPROACH", "BRAKE_HOVER",
)
CRUISE, LATERAL_AVOID, VERTICAL_AVOID, YAW_SCAN, APPROACH, BRAKE_HOVER = range(6)
IMAGE_SIZE: Tuple[int, int] = (128, 160)          # (H_img, W_img)
HORIZON = 16                                      # H
ACTION_DIM = 4                                    # d_a = (v_x, v_y, v_z, yaw rate)
PROPRIO_DIM = 13                                  # d_s
GOAL_DIM = 4                                      # d_g
KNOT_DT = 0.02                                    # Delta (s)
PREDICTOR_HORIZON_S = 0.1                         # T_nu (s)
ACTION_SCALE: Tuple[float, float, float, float] = (7.0, 7.0, 2.0, 1.5)   # half-widths of A_0 (config.action_scale)
DEPTH_SCALE_M = 1e-3                              # depth channel: uint16 millimetres
GRAVITY = 9.80665


# ---------------------------------------------------------------------------------------------
# Frames (Sec. 3.1)
# ---------------------------------------------------------------------------------------------
def wrap_angle(a: Any) -> Any:
    """Wrap to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_rotmat(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z), body -> world (PX4 ``VehicleOdometry.q``)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError(f"invalid quaternion {q}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat(q: Sequence[float]) -> float:
    """Yaw psi of the body x-axis about the NED down axis (rad)."""
    r = quat_to_rotmat(q)
    return math.atan2(r[1, 0], r[0, 0])


def world_to_heading(vec_w: np.ndarray, psi: Any) -> np.ndarray:
    """x^H = R_z(psi)^T x^W (``selector.world_to_heading``). ``[..., 3]``, ``[...]`` -> ``[..., 3]``."""
    v = np.asarray(vec_w, dtype=np.float64)
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


def heading_to_world(vec_h: np.ndarray, psi: Any) -> np.ndarray:
    """x^W = R_z(psi) x^H (Eq. 32). ``[..., 3]``, ``[...]`` -> ``[..., 3]``."""
    v = np.asarray(vec_h, dtype=np.float64)
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


def gravity_in_body(q: Sequence[float]) -> np.ndarray:
    """n^B: the unit gravity direction (NED +z) in the body FRD frame, R^T e_z; encodes roll and pitch."""
    return quat_to_rotmat(q)[2, :].copy()


def proprioception(velocity_ned: Sequence[float], q: Sequence[float], body_rates: Sequence[float],
                   last_setpoint: Sequence[float], yaw: float) -> np.ndarray:
    """S_t (Eq. 2) ``[13]`` float32, exactly as ``pipeline.VPAInferencePipeline.proprioception``.

    Args:
        velocity_ned: v^W ``[3]`` (m/s).
        q: attitude (w, x, y, z), body -> NED.
        body_rates: omega^B ``[3]`` (rad/s, body FRD).
        last_setpoint: the setpoint dispatched most recently, ``[4]`` = NED velocity (3) + yaw rate.
        yaw: psi_hat_t (rad), the yaw that defines H_t.
    """
    sp = np.asarray(last_setpoint, dtype=np.float64)
    out = np.concatenate([
        world_to_heading(np.asarray(velocity_ned, dtype=np.float64), yaw),
        gravity_in_body(q),
        np.asarray(body_rates, dtype=np.float64),
        world_to_heading(sp[:3], yaw),
        sp[3:4],
    ])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Goal feature (Eq. 3) and waypoint pointer (Eq. 14)
# ---------------------------------------------------------------------------------------------
def goal_feature(waypoint: Sequence[float], p_hat: Sequence[float], psi_hat: float,
                 saturation_distance: float = 20.0, eps: float = 1e-3) -> np.ndarray:
    """g_t = [ r_t / max(||r_t||, eps_r) ; min(||r_t||, D_max) / D_max ],  r_t = R_z(psi)^T (w - p)."""
    r = world_to_heading(np.asarray(waypoint, dtype=np.float64) - np.asarray(p_hat, dtype=np.float64), psi_hat)
    dist = float(np.linalg.norm(r))
    return np.concatenate([r / max(dist, eps), [min(dist, saturation_distance) / saturation_distance]]).astype(np.float32)


class WaypointTracker:
    """Eq. 14 with a sticky completion flag; ``step`` updates m_t, then returns g_t (Algorithm 1, line 2)."""

    def __init__(self, waypoints: np.ndarray, acceptance_radius: float = 1.0,
                 saturation_distance: float = 20.0, eps: float = 1e-3) -> None:
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 3 or wp.shape[0] < 1:
            raise ValueError(f"waypoints must be [M, 3] with M >= 1, got {wp.shape}")
        self.waypoints = wp
        self.acceptance_radius = float(acceptance_radius)
        self.saturation_distance = float(saturation_distance)
        self.eps = float(eps)
        self.m = 1                       # 1-based, as in the paper
        self.task_complete = False

    def step(self, p_hat: Sequence[float], psi_hat: float) -> np.ndarray:
        p = np.asarray(p_hat, dtype=np.float64)
        reached = float(np.linalg.norm(p - self.waypoints[self.m - 1])) < self.acceptance_radius
        if reached and self.m < len(self.waypoints):
            self.m += 1
        elif reached:
            self.task_complete = True
        return goal_feature(self.waypoints[self.m - 1], p, psi_hat, self.saturation_distance, self.eps)

    def distance_to_final(self, p_hat: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(p_hat, dtype=np.float64) - self.waypoints[-1]))


# ---------------------------------------------------------------------------------------------
# Primitive labels u*_t (Sec. 3.3)
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelThresholds:
    """Thresholds of the Sec. 3.3 rules (the paper gives none). Stored in the file attributes."""

    hover_speed: float = 0.3          # m/s
    approach_distance: float = 4.0    # m to the final waypoint
    vertical_threshold: float = 0.5   # m/s |v_z|
    lateral_threshold: float = 0.5    # m/s |v_y| in H_t
    yaw_rate_threshold: float = 0.5   # rad/s


def primitive_labels(command: np.ndarray, distance_to_final: np.ndarray, th: LabelThresholds) -> np.ndarray:
    """Rules of Sec. 3.3 in priority order (``selector.primitive_labels``). ``[N, 4]``, ``[N]`` -> ``[N]`` int64."""
    cmd = np.atleast_2d(np.asarray(command, dtype=np.float64))
    dist = np.atleast_1d(np.asarray(distance_to_final, dtype=np.float64))
    labels = np.full(cmd.shape[0], CRUISE, dtype=np.int64)
    rules = (  # lowest priority first
        (np.abs(cmd[:, 3]) > th.yaw_rate_threshold, YAW_SCAN),
        (np.abs(cmd[:, 1]) > th.lateral_threshold, LATERAL_AVOID),
        (np.abs(cmd[:, 2]) > th.vertical_threshold, VERTICAL_AVOID),
        (dist < th.approach_distance, APPROACH),
        (np.linalg.norm(cmd[:, :3], axis=1) < th.hover_speed, BRAKE_HOVER),
    )
    for condition, primitive in rules:
        labels[condition] = primitive
    return labels


# ---------------------------------------------------------------------------------------------
# Privileged occupancy (ground-truth obstacles of the world, in the PX4 local NED frame)
# ---------------------------------------------------------------------------------------------
@dataclass
class Obstacle:
    """A primitive obstacle in NED. ``cylinder``: vertical axis, ``center`` at mid-height, ``half_height``.
    ``sphere``: ``radius``. ``box``: axis-aligned, ``half_extents``."""

    kind: str
    center: Tuple[float, float, float]
    radius: float = 0.0
    half_height: float = 0.0
    half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center, dtype=np.float64)
        if self.kind == "cylinder":
            h = np.array([self.radius, self.radius, self.half_height])
        elif self.kind == "sphere":
            h = np.full(3, self.radius)
        elif self.kind == "box":
            h = np.asarray(self.half_extents, dtype=np.float64)
        else:
            raise ValueError(f"unknown obstacle kind {self.kind!r}")
        return c - h, c + h


class PrivilegedMap:
    """Signed distance to the nearest obstacle surface, for planning and collision checks.

    World files store obstacles in the Gazebo ENU world frame; ``from_world_json`` converts them to the
    PX4 local NED frame, whose origin is the spawn point: NED = (y - y0, x - x0, -(z - z0)).
    """

    def __init__(self, obstacles: Iterable[Obstacle] = (),
                 bounds_ned: Optional[Tuple[Sequence[float], Sequence[float]]] = None) -> None:
        self.obstacles: List[Obstacle] = list(obstacles)
        self.bounds = None if bounds_ned is None else (np.asarray(bounds_ned[0], float), np.asarray(bounds_ned[1], float))
        boxes = [o.aabb() for o in self.obstacles]
        self._lo = np.array([b[0] for b in boxes]).reshape(-1, 3)
        self._hi = np.array([b[1] for b in boxes]).reshape(-1, 3)
        self._kind = np.array([o.kind for o in self.obstacles])
        self._c = np.array([o.center for o in self.obstacles], dtype=np.float64).reshape(-1, 3)
        self._r = np.array([o.radius for o in self.obstacles], dtype=np.float64)
        self._hh = np.array([o.half_height for o in self.obstacles], dtype=np.float64)
        self._he = np.array([o.half_extents for o in self.obstacles], dtype=np.float64).reshape(-1, 3)

    @classmethod
    def from_world_json(cls, path: str, spawn_enu: Optional[Sequence[float]] = None) -> "PrivilegedMap":
        with open(path, encoding="utf-8") as f:
            world = json.load(f)
        x0, y0, z0 = (float(v) for v in (spawn_enu if spawn_enu is not None else world.get("spawn_enu", (0, 0, 0))))

        def ned(p: Sequence[float]) -> Tuple[float, float, float]:
            return (float(p[1]) - y0, float(p[0]) - x0, -(float(p[2]) - z0))

        obstacles = []
        for o in world.get("obstacles", []):
            kind = o["type"]
            if kind == "cylinder":
                obstacles.append(Obstacle("cylinder", ned(o["center"]), radius=float(o["radius"]),
                                          half_height=0.5 * float(o["length"])))
            elif kind == "sphere":
                obstacles.append(Obstacle("sphere", ned(o["center"]), radius=float(o["radius"])))
            elif kind == "box":  # axis-aligned in ENU: swap the x/y half-extents
                sx, sy, sz = (0.5 * float(v) for v in o["size"])
                obstacles.append(Obstacle("box", ned(o["center"]), half_extents=(sy, sx, sz)))
            else:
                raise ValueError(f"unsupported obstacle type {kind!r} in {path}")
        bounds = None
        if "bounds_enu" in world:
            (xa, ya), (xb, yb) = world["bounds_enu"]
            bounds = ((min(ya, yb) - y0, min(xa, xb) - x0, -np.inf), (max(ya, yb) - y0, max(xa, xb) - x0, np.inf))
        return cls(obstacles, bounds)

    def __len__(self) -> int:
        return len(self.obstacles)

    cap: float = 25.0   # distances above this are only guaranteed to be > cap (m)

    def distance(self, points: np.ndarray, chunk: int = 1024) -> np.ndarray:
        """Signed distance ``[...]`` from ``points [..., 3]`` to the nearest obstacle surface.

        Exact up to ``cap``; beyond it the value is only guaranteed to exceed ``cap`` (+inf if no obstacle
        is within ``cap``).
        """
        pts = np.asarray(points, dtype=np.float64)
        flat = pts.reshape(-1, 3)
        out = np.full(flat.shape[0], np.inf)
        if len(self.obstacles) == 0 or flat.shape[0] == 0:
            return out.reshape(pts.shape[:-1])
        for start in range(0, flat.shape[0], chunk):
            p = flat[start:start + chunk]
            lo, hi = p.min(axis=0), p.max(axis=0)
            # Obstacles whose bounding box is farther than `cap` from this chunk's bounding box are
            # farther than `cap` from every point in it, so dropping them keeps every distance <= cap exact.
            gap = np.maximum(np.maximum(self._lo - hi, lo - self._hi), 0.0)
            keep = np.nonzero(np.linalg.norm(gap, axis=1) <= self.cap)[0]
            if keep.size:
                out[start:start + chunk] = self._distance_to(p, keep)
        return out.reshape(pts.shape[:-1])

    def _distance_to(self, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
        best = np.full(p.shape[0], np.inf)
        for kind in ("cylinder", "sphere", "box"):
            sel = idx[self._kind[idx] == kind]
            if sel.size == 0:
                continue
            rel = p[:, None, :] - self._c[None, sel, :]                                   # [P, K, 3]
            if kind == "cylinder":
                dr = np.linalg.norm(rel[..., :2], axis=-1) - self._r[None, sel]
                dz = np.abs(rel[..., 2]) - self._hh[None, sel]
                outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
                d = outside + np.minimum(np.maximum(dr, dz), 0.0)
            elif kind == "sphere":
                d = np.linalg.norm(rel, axis=-1) - self._r[None, sel]
            else:
                q = np.abs(rel) - self._he[None, sel, :]
                d = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
            best = np.minimum(best, d.min(axis=1))
        return best

    def segment_clearance(self, a: Sequence[float], b: Sequence[float], step: float = 0.1) -> float:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        return float(self.distance(a[None] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None]).min())

    def inside_bounds(self, p: Sequence[float], margin: float = 0.0) -> bool:
        if self.bounds is None:
            return True
        p = np.asarray(p, float)
        return bool(np.all(p[:2] >= self.bounds[0][:2] + margin) and np.all(p[:2] <= self.bounds[1][:2] - margin))


# ---------------------------------------------------------------------------------------------
# Minimum-snap spline (Mellinger & Kumar [3]) through fixed via points
# ---------------------------------------------------------------------------------------------
_DEG = 7  # degree 7 per segment: p^(8) = 0 is the Euler-Lagrange condition of min int ||p^(4)||^2


def _deriv_row(tau: float, order: int) -> np.ndarray:
    """d^order/dtau^order of [1, tau, ..., tau^7] at ``tau``."""
    row = np.zeros(_DEG + 1)
    for k in range(order, _DEG + 1):
        row[k] = math.factorial(k) / math.factorial(k - order) * tau ** (k - order)
    return row


class MinSnapSpline:
    """Piecewise degree-7 polynomial through ``via [N+1, 3]`` with segment durations ``durations [N]``.

    Start and end at rest (velocity, acceleration and jerk zero); the interior via points are passed
    with C^6 continuity. Degree 7 with C^6 continuity is the square (8N x 8N) linear system whose
    solution is the minimum-snap interpolant, so no quadratic program is needed. Each segment is
    written in normalised time tau = (t - t_i) / T_i and the equations are scaled by T_i^m, which keeps
    the system well conditioned for segment times from 0.1 s to minutes.
    """

    def __init__(self, via: np.ndarray, durations: np.ndarray) -> None:
        via = np.asarray(via, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        n = via.shape[0] - 1
        if via.ndim != 2 or via.shape[1] != 3 or n < 1 or durations.shape != (n,):
            raise ValueError(f"need via [N+1, 3] and durations [N], got {via.shape}, {durations.shape}")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("segment durations must be finite and positive")
        self.via, self.durations = via, durations
        self.knots = np.concatenate([[0.0], np.cumsum(durations)])
        size = 8 * n
        a = np.zeros((size, size))
        b = np.zeros((size, 3))
        row = 0

        def put(seg: int, tau: float, order: int, scale: float = 1.0) -> np.ndarray:
            line = np.zeros(size)
            line[8 * seg:8 * seg + 8] = scale * _deriv_row(tau, order)
            return line

        for m in range(4):                                   # start: p = via_0, v = a = j = 0
            a[row] = put(0, 0.0, m)
            b[row] = via[0] if m == 0 else 0.0
            row += 1
        for m in range(4):                                   # end: p = via_N, v = a = j = 0
            a[row] = put(n - 1, 1.0, m)
            b[row] = via[n] if m == 0 else 0.0
            row += 1
        for j in range(1, n):                                # interior via points
            a[row] = put(j - 1, 1.0, 0)
            b[row] = via[j]
            row += 1
            a[row] = put(j, 0.0, 0)
            b[row] = via[j]
            row += 1
            ratio = durations[j - 1] / durations[j]
            for m in range(1, 7):                            # C^6, scaled by T_{j-1}^m
                a[row] = put(j - 1, 1.0, m) - put(j, 0.0, m, ratio ** m)
                row += 1
        if row != size:
            raise AssertionError("constraint count mismatch")
        self.coeffs = np.linalg.solve(a, b).reshape(n, 8, 3)                            # [N, 8, 3]

    @property
    def duration(self) -> float:
        return float(self.knots[-1])

    def evaluate(self, t: Any, order: int = 0) -> np.ndarray:
        """d^order p / dt^order at times ``t`` (scalar or ``[K]``) -> ``[3]`` or ``[K, 3]``.

        Before 0 and after the end the spline is held at its end points (derivatives zero).
        """
        scalar = np.ndim(t) == 0
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        clamped = np.clip(ts, 0.0, self.duration)
        seg = np.clip(np.searchsorted(self.knots, clamped, side="right") - 1, 0, len(self.durations) - 1)
        dur = self.durations[seg]
        tau = (clamped - self.knots[seg]) / dur
        k = np.arange(_DEG + 1)
        valid = k >= order
        fact = np.where(valid, [math.factorial(i) / math.factorial(i - order) if i >= order else 0.0 for i in k], 0.0)
        powers = np.where(valid[None], tau[:, None] ** np.maximum(k - order, 0)[None], 0.0) * fact[None]   # [K, 8]
        out = np.einsum("kc,kcd->kd", powers, self.coeffs[seg]) / dur[:, None] ** order
        if order > 0:
            out[(ts < 0.0) | (ts > self.duration)] = 0.0
        return out[0] if scalar else out


# ---------------------------------------------------------------------------------------------
# Expert trajectory = spline + rate-limited heading reference
# ---------------------------------------------------------------------------------------------
class ExpertTrajectory:
    """Position spline plus a yaw reference that faces the direction of travel.

    The yaw reference is precomputed on a ``dt`` grid: the desired heading is the direction of the
    horizontal displacement over the next ``lookahead`` seconds (held while it is shorter than
    ``min_displacement``), and the reference follows it with a rate limit of ``yaw_rate_max``. With
    ``fixed_yaw`` the reference turns to that heading instead (used for alignment before a mission).
    """

    def __init__(self, spline: MinSnapSpline, yaw0: float, yaw_rate_max: float = 1.2, lookahead: float = 1.0,
                 min_displacement: float = 0.3, fixed_yaw: Optional[float] = None, dt: float = 0.01,
                 hold_after: float = 5.0) -> None:
        self.spline = spline
        self.dt = float(dt)
        n = int(math.ceil((spline.duration + hold_after) / dt)) + 2
        times = np.arange(n) * dt
        if fixed_yaw is None:
            pos = spline.evaluate(times)
            ahead = spline.evaluate(times + lookahead)
            disp = ahead[:, :2] - pos[:, :2]
            norm = np.linalg.norm(disp, axis=1)
            desired = np.arctan2(disp[:, 1], disp[:, 0])
        else:
            norm = np.full(n, np.inf)
            desired = np.full(n, float(fixed_yaw))
        yaw = np.empty(n)
        yaw[0] = float(yaw0)
        target = float(yaw0)
        step = yaw_rate_max * dt
        for i in range(1, n):
            if norm[i] > min_displacement:
                target = desired[i]
            yaw[i] = yaw[i - 1] + float(np.clip(wrap_angle(target - yaw[i - 1]), -step, step))
        self._times, self._yaw = times, yaw                      # unwrapped (continuous)
        self._yaw_rate = np.gradient(yaw, dt)

    @property
    def duration(self) -> float:
        return self.spline.duration

    def reference(self, t: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(position, velocity, acceleration, yaw, yaw rate) at ``t`` (scalar or ``[K]``)."""
        ts = np.asarray(t, dtype=np.float64)
        yaw = np.interp(ts, self._times, self._yaw)
        rate = np.interp(ts, self._times, self._yaw_rate)
        return (self.spline.evaluate(ts, 0), self.spline.evaluate(ts, 1), self.spline.evaluate(ts, 2),
                wrap_angle(yaw), rate)

    def initial_heading(self, horizon: float = 3.0) -> Optional[float]:
        """Heading of the first horizontal displacement longer than 0.5 m within ``horizon`` s (or None)."""
        times = np.arange(0.0, min(horizon, self.duration) + 1e-9, 0.05)
        disp = self.spline.evaluate(times)[:, :2] - self.spline.evaluate(0.0)[:2]
        idx = np.nonzero(np.linalg.norm(disp, axis=1) > 0.5)[0]
        if idx.size == 0:
            return None
        d = disp[idx[0]]
        return float(math.atan2(d[1], d[0]))


# ---------------------------------------------------------------------------------------------
# Privileged planner: A* + shortcut + min-snap + refinement
# ---------------------------------------------------------------------------------------------
class PlanningError(RuntimeError):
    """No feasible expert trajectory for the requested mission."""


@dataclass
class PlannerConfig:
    resolution: float = 0.5                   # A* grid (m)
    inflation: float = 1.0                    # obstacle inflation for A* and shortcuts (m)
    clearance: float = 0.6                    # required clearance of the final spline: r_v + margin (m)
    altitude_band: Tuple[float, float] = (-6.0, -1.0)   # allowed NED z (m), i.e. 1-6 m above the origin
    box_margin: float = 8.0                   # A* search box around each leg (m)
    max_refinements: int = 10
    accel_limit: float = 4.0                  # reference acceleration bound (m/s^2), below a_max = 5
    climb_limit: float = 1.8                  # reference climb speed (m/s), below the PX4 limit
    descend_limit: float = 1.2                # reference descent speed (m/s), below MPC_Z_VEL_MAX_DN = 1.5
    max_astar_nodes: int = 400_000


class ExpertPlanner:
    def __init__(self, world: PrivilegedMap, cfg: Optional[PlannerConfig] = None) -> None:
        self.world = world
        self.cfg = cfg or PlannerConfig()

    # ---- geometric path ----
    def _free(self, p: np.ndarray) -> np.ndarray:
        return self.world.distance(p) > self.cfg.inflation

    def _segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        if len(self.world) == 0:
            return True
        return self.world.segment_clearance(a, b, step=0.5 * self.cfg.resolution) > self.cfg.inflation

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        res, cfg = self.cfg.resolution, self.cfg
        lo = np.minimum(start, goal) - cfg.box_margin
        hi = np.maximum(start, goal) + cfg.box_margin
        lo[2], hi[2] = min(cfg.altitude_band[0], start[2], goal[2]), max(cfg.altitude_band[1], start[2], goal[2])
        shape = np.maximum(np.ceil((hi - lo) / res).astype(int) + 1, 1)
        if int(np.prod(shape)) > cfg.max_astar_nodes:
            raise PlanningError(f"A* grid {tuple(shape)} too large; split the leg or coarsen the grid")
        axes = [lo[i] + res * np.arange(shape[i]) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)                   # [nx, ny, nz, 3]
        free = self._free(grid.reshape(-1, 3)).reshape(tuple(shape))
        z_ok = (axes[2] >= cfg.altitude_band[0] - 1e-9) & (axes[2] <= cfg.altitude_band[1] + 1e-9)
        free &= z_ok[None, None, :]

        def cell(p: np.ndarray) -> Tuple[int, int, int]:
            c = np.clip(np.rint((p - lo) / res).astype(int), 0, shape - 1)
            return int(c[0]), int(c[1]), int(c[2])

        s, g = cell(start), cell(goal)
        free[s] = True
        free[g] = True
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
        costs = [res * math.sqrt(i * i + j * j + k * k) for i, j, k in offsets]
        goal_xyz = np.asarray(g, dtype=float)

        def h(c: Tuple[int, int, int]) -> float:
            return res * float(np.linalg.norm(np.asarray(c, dtype=float) - goal_xyz))

        open_heap = [(h(s), 0.0, s)]
        best = {s: 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        closed = set()
        while open_heap:
            _, cost, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == g:
                break
            closed.add(cur)
            for (di, dj, dk), step_cost in zip(offsets, costs, strict=True):
                nb = (cur[0] + di, cur[1] + dj, cur[2] + dk)
                if not (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]):
                    continue
                if not free[nb] or nb in closed:
                    continue
                new_cost = cost + step_cost
                if new_cost < best.get(nb, math.inf):
                    best[nb] = new_cost
                    parent[nb] = cur
                    heapq.heappush(open_heap, (new_cost + h(nb), new_cost, nb))
        if g not in parent and g != s:
            raise PlanningError(f"no collision-free path from {np.round(start, 2)} to {np.round(goal, 2)}")
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        path = [lo + res * np.asarray(c, dtype=float) for c in reversed(cells)]
        path[0], path[-1] = start.copy(), goal.copy()
        return path

    def _shortcut(self, path: List[np.ndarray]) -> List[np.ndarray]:
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._segment_free(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def geometric_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        if self._segment_free(start, goal):
            return [start, goal]
        return self._shortcut(self._astar(start, goal))

    # ---- timing ----
    def _allocate(self, via: np.ndarray, speed: float) -> np.ndarray:
        a = self.cfg.accel_limit
        length = np.linalg.norm(np.diff(via, axis=0), axis=1)
        t = np.where(length > speed * speed / a, length / speed + speed / a, 2.0 * np.sqrt(length / a))
        return np.maximum(t, 0.3)

    def _scale_factor(self, spline: MinSnapSpline, speed: float) -> float:
        ts = np.linspace(0.0, spline.duration, max(200, int(spline.duration / 0.02)))
        v, acc = spline.evaluate(ts, 1), spline.evaluate(ts, 2)
        ratios = [
            np.linalg.norm(v[:, :2], axis=1).max() / speed,
            np.maximum(-v[:, 2], 0.0).max() / self.cfg.climb_limit,
            np.maximum(v[:, 2], 0.0).max() / self.cfg.descend_limit,
            math.sqrt(np.linalg.norm(acc, axis=1).max() / self.cfg.accel_limit),
        ]
        return float(max(ratios))

    def fit(self, via: np.ndarray, speed: float) -> MinSnapSpline:
        """Spline through ``via`` with uniform time scaling to the binding limit (exact for this family)."""
        durations = self._allocate(via, speed)
        spline = MinSnapSpline(via, durations)
        for _ in range(3):                    # uniform scaling is exact; repeat only for round-off
            factor = self._scale_factor(spline, speed)
            if abs(factor - 1.0) < 0.01:
                break
            durations = durations * factor
            spline = MinSnapSpline(via, durations)
        return spline

    def _violations(self, spline: MinSnapSpline) -> List[int]:
        ts = np.arange(0.0, spline.duration, 0.02)
        pos = spline.evaluate(ts)
        bad = self.world.distance(pos) < self.cfg.clearance if len(self.world) else np.zeros(len(ts), bool)
        bad |= pos[:, 2] > self.cfg.altitude_band[1] + 0.5                         # too low (NED z down)
        seg = np.clip(np.searchsorted(spline.knots, ts[bad], side="right") - 1, 0, len(spline.durations) - 1)
        return sorted(set(int(s) for s in seg))

    def plan(self, start: Sequence[float], waypoints: np.ndarray, speed: float, yaw0: float) -> ExpertTrajectory:
        """Expert trajectory from ``start`` (at rest) through the mission ``waypoints [M, 3]``, ending at rest."""
        start = np.asarray(start, dtype=np.float64)
        via = [start]
        for wp in np.asarray(waypoints, dtype=np.float64):
            via.extend(self.geometric_path(via[-1], wp)[1:])
        via_arr = np.asarray(via)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(via_arr, axis=0), axis=1) > 1e-3])
        via_arr = via_arr[keep]
        if len(via_arr) < 2:
            raise PlanningError("mission has no displacement")
        for _ in range(self.cfg.max_refinements + 1):
            spline = self.fit(via_arr, speed)
            bad = self._violations(spline)
            if not bad:
                return ExpertTrajectory(spline, yaw0)
            mids = [(0.5 * (via_arr[i] + via_arr[i + 1]), i + 1) for i in bad]
            for mid, pos in sorted(mids, key=lambda m: -m[1]):
                via_arr = np.insert(via_arr, pos, mid, axis=0)
        raise PlanningError("spline keeps violating the clearance after refinement")

    def hold(self, position: Sequence[float], yaw0: float, target: Optional[Sequence[float]] = None,
             yaw_target: Optional[float] = None, speed: float = 1.0) -> ExpertTrajectory:
        """Short trajectory for take-off, alignment and hover (no obstacle check)."""
        p = np.asarray(position, dtype=np.float64)
        q = p if target is None else np.asarray(target, dtype=np.float64)
        if np.linalg.norm(q - p) < 1e-3:
            spline = MinSnapSpline(np.stack([p, p]), np.array([1.0]))
        else:
            spline = self.fit(np.stack([p, q]), speed)
        return ExpertTrajectory(spline, yaw0, fixed_yaw=yaw_target if yaw_target is not None else yaw0)


_WORLD_CACHE: Dict[Optional[str], PrivilegedMap] = {}


def load_world(world_json: Optional[str]) -> PrivilegedMap:
    """``PrivilegedMap`` of a world JSON sidecar (cached per process); an empty map for None."""
    if world_json not in _WORLD_CACHE:
        _WORLD_CACHE[world_json] = PrivilegedMap() if world_json is None else PrivilegedMap.from_world_json(world_json)
    return _WORLD_CACHE[world_json]


def plan_mission_spline(world_json: Optional[str], start: Sequence[float], waypoints: np.ndarray,
                        speed: float) -> MinSnapSpline:
    """Picklable entry point for planning in a worker process: the expert spline of one mission."""
    return ExpertPlanner(load_world(world_json)).plan(start, waypoints, speed, 0.0).spline


# ---------------------------------------------------------------------------------------------
# Tracking law and expert chunk
# ---------------------------------------------------------------------------------------------
@dataclass
class CommandLimits:
    """Saturation of the expert commands (NED): horizontal speed, climb, descent and yaw rate."""

    speed_h: float = 5.0
    climb: float = 2.0
    descend: float = 1.5          # PX4 MPC_Z_VEL_MAX_DN default
    yaw_rate: float = 1.5

    def apply(self, vel_ned: np.ndarray, yaw_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v = np.array(vel_ned, dtype=np.float64, copy=True)
        h = np.linalg.norm(v[..., :2], axis=-1, keepdims=True)
        v[..., :2] *= np.minimum(1.0, self.speed_h / np.maximum(h, 1e-9))
        v[..., 2] = np.clip(v[..., 2], -self.climb, self.descend)
        return v, np.clip(yaw_rate, -self.yaw_rate, self.yaw_rate)


class ExpertController:
    """Tracking law of the privileged expert and its 16-knot chunk (see the module docstring)."""

    def __init__(self, trajectory: ExpertTrajectory, start_time: float, limits: CommandLimits,
                 k_pos: float = 1.5, k_yaw: float = 2.0) -> None:
        self.trajectory = trajectory
        self.t0 = float(start_time)
        self.limits = limits
        self.k_pos, self.k_yaw = float(k_pos), float(k_yaw)

    def elapsed(self, now: float) -> float:
        return float(now) - self.t0

    def finished(self, now: float, settle: float = 0.0) -> bool:
        return self.elapsed(now) >= self.trajectory.duration + settle

    def command(self, now: float, p_hat: Sequence[float], yaw_hat: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """(NED velocity [3], NED acceleration feed-forward [3], yaw rate) to dispatch at ``now``."""
        p, v, a, yaw, rate = self.trajectory.reference(self.elapsed(now))
        vel = v + self.k_pos * (p - np.asarray(p_hat, dtype=np.float64))
        yr = rate + self.k_yaw * wrap_angle(yaw - yaw_hat)
        vel, yr = self.limits.apply(vel, np.asarray(yr))
        return vel, a, float(yr)

    def chunk(self, now: float, p_hat: Sequence[float], yaw_hat: float, horizon: int = HORIZON,
              knot_dt: float = KNOT_DT) -> np.ndarray:
        """A_t ``[H, 4]``: knots at ``now + k Delta`` in the heading frame of ``yaw_hat`` (Eq. 4)."""
        s = np.arange(horizon) * knot_dt
        p0, _, _, yaw0, _ = self.trajectory.reference(self.elapsed(now))
        e_p = p0 - np.asarray(p_hat, dtype=np.float64)
        e_y = float(wrap_angle(yaw0 - yaw_hat))
        _, v, _, _, rate = self.trajectory.reference(self.elapsed(now) + s)
        vel = v + self.k_pos * np.exp(-self.k_pos * s)[:, None] * e_p[None]
        yr = rate + self.k_yaw * np.exp(-self.k_yaw * s) * e_y
        vel, yr = self.limits.apply(vel, yr)
        return np.concatenate([world_to_heading(vel, yaw_hat), yr[:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Mission sampling (Sec. 5.1: 3-6 waypoints, 10-40 m apart)
# ---------------------------------------------------------------------------------------------
def sample_mission(world: PrivilegedMap, rng: np.random.Generator, start: Sequence[float],
                   num_waypoints: Tuple[int, int] = (3, 6), spacing: Tuple[float, float] = (10.0, 40.0),
                   altitude: Tuple[float, float] = (1.5, 4.0), min_clearance: float = 1.5,
                   bounds_margin: float = 5.0, max_tries: int = 200) -> np.ndarray:
    """Waypoints ``[M, 3]`` (NED). Each leg turns by at most 100 degrees from the previous one."""
    start = np.asarray(start, dtype=np.float64)
    for _ in range(max_tries):
        m = int(rng.integers(num_waypoints[0], num_waypoints[1] + 1))
        points, heading, prev = [], float(rng.uniform(-np.pi, np.pi)), start
        for _ in range(m):
            for _ in range(100):
                h = heading + float(rng.uniform(-1.75, 1.75)) if points else float(rng.uniform(-np.pi, np.pi))
                d = float(rng.uniform(*spacing))
                cand = np.array([prev[0] + d * math.cos(h), prev[1] + d * math.sin(h), -float(rng.uniform(*altitude))])
                if not world.inside_bounds(cand, bounds_margin):
                    continue
                if len(world) and float(world.distance(cand)) < min_clearance:
                    continue
                points.append(cand)
                prev, heading = cand, h
                break
            else:
                break
        if len(points) == m:
            return np.asarray(points)
    raise PlanningError("could not sample a mission inside the world bounds")


# ---------------------------------------------------------------------------------------------
# Image encoding and lookahead targets
# ---------------------------------------------------------------------------------------------
def encode_rgbd(rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """``rgb [H, W, 3]`` uint8 + ``depth [H, W]`` (m, non-finite or <= 0 = invalid) -> ``[H, W, 4]`` uint16.

    R, G, B keep their 0-255 values; D is millimetres, 0 where invalid, 65535 at most (65.5 m).
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    if rgb.shape[:2] != depth.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb {rgb.shape} and depth {depth.shape} do not match")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8")
    valid = np.isfinite(depth) & (depth > 0.0)
    mm = np.where(valid, np.clip(np.rint(np.where(valid, depth, 0.0) / DEPTH_SCALE_M), 1, 65535), 0)
    return np.concatenate([rgb.astype(np.uint16), mm.astype(np.uint16)[..., None]], axis=-1)


def decode_rgbd(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``[..., H, W, 4]`` uint16 -> (rgb ``[..., H, W, 3]`` uint8, depth ``[..., H, W]`` float32 m, 0 = invalid)."""
    images = np.asarray(images)
    return images[..., :3].astype(np.uint8), images[..., 3].astype(np.float32) * np.float32(DEPTH_SCALE_M)


def future_targets(timestamps: np.ndarray, lookahead: float = PREDICTOR_HORIZON_S,
                   tolerance: Optional[float] = None) -> np.ndarray:
    """Index of the frame captured closest to tau_t + T_nu (Sec. 3.2), or -1 if none lies within ``tolerance``.

    The default tolerance is 0.6 of the median frame interval, so a dropped frame never silently
    stretches the lookahead.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    out = np.full(ts.shape[0], -1, dtype=np.int64)
    if ts.shape[0] < 2:
        return out
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    tol = 0.6 * float(np.median(np.diff(ts))) if tolerance is None else float(tolerance)
    target = ts + lookahead
    hi = np.clip(np.searchsorted(ts, target), 1, ts.shape[0] - 1)
    lo = hi - 1
    pick = np.where(np.abs(ts[hi] - target) < np.abs(ts[lo] - target), hi, lo)
    ok = (np.abs(ts[pick] - target) <= tol) & (pick > np.arange(ts.shape[0]))
    out[ok] = pick[ok]
    return out


# ---------------------------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------------------------
@dataclass
class FrameSample:
    """One recorded decision step (one camera frame)."""

    timestamp: float                 # tau_t (s)
    image: np.ndarray                # [128, 160, 4] uint16 (encode_rgbd)
    state: np.ndarray                # [13] float32 S_t
    goal: np.ndarray                 # [4] float32 g_t
    action: np.ndarray               # [16, 4] float32 A_t
    primitive: int                   # u*_t
    position: np.ndarray             # [3] p_hat_t (NED)
    yaw: float                       # psi_hat_t
    waypoint_index: int              # m_t (1-based) used for g_t
    task_complete: bool
    reference_position: np.ndarray   # [3] expert p_ref(tau_t)
    odometry_age: float              # tau_t - odometry sample time (s)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def file_attributes(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Root attributes that make a file self-describing (dimensions, units, frames, encodings)."""
    attrs: Dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_height": IMAGE_SIZE[0], "image_width": IMAGE_SIZE[1], "image_channels": 4,
        "image_encoding": "uint16; channels R, G, B in [0, 255] and D in millimetres (0 = invalid)",
        "depth_scale_m": DEPTH_SCALE_M,
        "horizon": HORIZON, "action_dim": ACTION_DIM, "proprio_dim": PROPRIO_DIM, "goal_dim": GOAL_DIM,
        "knot_dt": KNOT_DT, "predictor_horizon_s": PREDICTOR_HORIZON_S,
        "action_scale": np.asarray(ACTION_SCALE, dtype=np.float64),
        "primitives": _json(list(PRIMITIVES)),
        "state_layout": "v^H (3), n^B (3), omega^B (3), a_bar^H (4: v_x, v_y, v_z, yaw rate)",
        "action_layout": "knot k at tau_t + k*knot_dt: v_x, v_y, v_z (m/s, heading frame at tau_t), yaw rate (rad/s)",
        "frames": "W: PX4 local NED (EKF2 origin); H_t: W rotated about down by yaw psi_t; B: body FRD",
    }
    attrs.update(extra or {})
    return attrs


class EpisodeWriter:
    """Appends ``FrameSample``s to one episode group in blocks; ``finalize`` writes the lookahead targets."""

    _SPECS = {
        "observations/images": ((IMAGE_SIZE[0], IMAGE_SIZE[1], 4), np.uint16),
        "observations/state": ((PROPRIO_DIM,), np.float32),
        "observations/goal": ((GOAL_DIM,), np.float32),
        "actions": ((HORIZON, ACTION_DIM), np.float32),
        "primitives": ((), np.int64),
        "timestamps": ((), np.float64),
        "privileged/position_ned": ((3,), np.float32),
        "privileged/yaw": ((), np.float32),
        "privileged/waypoint_index": ((), np.int64),
        "privileged/task_complete": ((), np.bool_),
        "privileged/reference_position_ned": ((3,), np.float32),
        "privileged/odometry_age_s": ((), np.float32),
    }

    def __init__(self, group: Any, block_size: int = 64, compression: Optional[str] = "gzip") -> None:
        self.group = group
        group.attrs["complete"] = False          # set by finalize; readers skip groups an interrupted writer left
        self.block_size = int(block_size)
        self.count = 0
        self._buffer: List[FrameSample] = []
        self._last_time = -math.inf
        for name, (shape, dtype) in self._SPECS.items():
            kwargs: Dict[str, Any] = {}
            if name == "observations/images" and compression:
                kwargs = {"compression": compression, "shuffle": True}
                if compression == "gzip":
                    kwargs["compression_opts"] = 4
            chunks = (1, *shape) if name == "observations/images" else (min(256, self.block_size * 4), *shape)
            group.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=chunks, **kwargs)

    def append(self, sample: FrameSample) -> None:
        if not sample.timestamp > self._last_time:
            raise ValueError(f"timestamps must increase: {sample.timestamp} after {self._last_time}")
        self._last_time = sample.timestamp
        self._buffer.append(sample)
        if len(self._buffer) >= self.block_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        cols = {
            "observations/images": [s.image for s in self._buffer],
            "observations/state": [s.state for s in self._buffer],
            "observations/goal": [s.goal for s in self._buffer],
            "actions": [s.action for s in self._buffer],
            "primitives": [s.primitive for s in self._buffer],
            "timestamps": [s.timestamp for s in self._buffer],
            "privileged/position_ned": [s.position for s in self._buffer],
            "privileged/yaw": [s.yaw for s in self._buffer],
            "privileged/waypoint_index": [s.waypoint_index for s in self._buffer],
            "privileged/task_complete": [s.task_complete for s in self._buffer],
            "privileged/reference_position_ned": [s.reference_position for s in self._buffer],
            "privileged/odometry_age_s": [s.odometry_age for s in self._buffer],
        }
        n0, n = self.count, len(self._buffer)
        for name, values in cols.items():
            shape, dtype = self._SPECS[name]
            arr = np.asarray(values, dtype=dtype)
            if arr.shape != (n, *shape):
                raise ValueError(f"{name}: expected {(n, *shape)}, got {arr.shape}")
            if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: non-finite values")
            ds = self.group[name]
            ds.resize(n0 + n, axis=0)
            ds[n0:n0 + n] = arr
        self.count += n
        self._buffer.clear()

    def finalize(self, attrs: Dict[str, Any]) -> int:
        self.flush()
        ts = self.group["timestamps"][...]
        targets = future_targets(ts) if len(ts) else np.zeros(0, dtype=np.int64)
        ds = self.group.create_dataset("future_targets", data=targets, dtype=np.int64)
        ds.attrs["lookahead_s"] = PREDICTOR_HORIZON_S
        ds.attrs["description"] = "index of the frame captured closest to tau_t + T_nu (Sec. 3.2); -1 = none"
        dts = np.diff(ts) if len(ts) > 1 else np.zeros(0)
        stats = {
            "num_frames": int(len(ts)),
            "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
            "median_frame_dt_s": float(np.median(dts)) if dts.size else 0.0,
            "max_frame_gap_s": float(dts.max()) if dts.size else 0.0,
            "num_future_targets": int((targets >= 0).sum()),
        }
        for key, value in {**stats, **attrs}.items():
            self.group.attrs[key] = _json(value) if isinstance(value, (dict, list, tuple)) else value
        self.group.attrs["complete"] = True
        return int(len(ts))

    def __len__(self) -> int:
        return self.count + len(self._buffer)


class DatasetWriter:
    """One HDF5 file of episodes. ``mode="a"`` appends to an existing file of the same schema version."""

    def __init__(self, path: str, attrs: Optional[Dict[str, Any]] = None, mode: str = "w",
                 compression: Optional[str] = "gzip") -> None:
        import h5py  # local import: the collector imports this module before it needs h5py

        if mode not in ("w", "a"):
            raise ValueError("mode must be 'w' or 'a'")
        self.file = h5py.File(path, mode)
        self.compression = compression
        if mode == "a" and "schema" in self.file.attrs:
            found = (self.file.attrs["schema"], self.file.attrs["schema_version"])
            if found != (SCHEMA_NAME, SCHEMA_VERSION):
                self.file.close()
                raise ValueError(f"{path} has schema {found}, expected {(SCHEMA_NAME, SCHEMA_VERSION)}")
        else:
            for key, value in file_attributes(attrs).items():
                self.file.attrs[key] = value
        for name in [k for k in self.file.keys() if k.startswith("episode_")]:
            if not bool(self.file[name].attrs.get("complete", True)):
                del self.file[name]                # left unfinished by an interrupted writer
        self._next = 1 + max((int(k.split("_")[1]) for k in self.file.keys() if k.startswith("episode_")), default=-1)

    def begin_episode(self) -> Tuple[str, EpisodeWriter]:
        name = f"episode_{self._next:04d}"
        self._next += 1
        return name, EpisodeWriter(self.file.create_group(name), compression=self.compression)

    def discard(self, name: str) -> None:
        if name in self.file:
            del self.file[name]

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


# ---------------------------------------------------------------------------------------------
# Offline dry run: kinematic quadrotor + ray-cast RGB-D camera (no ROS, no Gazebo)
# ---------------------------------------------------------------------------------------------
class PinholeCamera:
    """Forward-looking pinhole camera of ``sim/models/aero_vpa_rgbd`` (160 x 128, HFOV 1.274 rad)."""

    def __init__(self, width: int = IMAGE_SIZE[1], height: int = IMAGE_SIZE[0], hfov: float = 1.274,
                 offset_body: Sequence[float] = (0.13, 0.0, -0.02), near: float = 0.2, far: float = 19.1) -> None:
        self.width, self.height = width, height
        self.f = 0.5 * width / math.tan(0.5 * hfov)
        u = np.arange(width) + 0.5 - 0.5 * width
        v = np.arange(height) + 0.5 - 0.5 * height
        uu, vv = np.meshgrid(u, v)
        self.rays_body = np.stack([np.full_like(uu, self.f), uu, vv], axis=-1) / self.f   # FRD, x-component 1
        self.offset = np.asarray(offset_body, dtype=np.float64)
        self.near, self.far = near, far

    def render(self, world: PrivilegedMap, position: np.ndarray, rotation: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(rgb [H, W, 3] uint8, z-depth [H, W] float32 m; +inf beyond ``far``) of the ground plane and obstacles."""
        origin = position + rotation @ self.offset
        dirs = self.rays_body.reshape(-1, 3) @ rotation.T                              # world; body-x component 1
        with np.errstate(divide="ignore", invalid="ignore"):
            t_best = np.where(dirs[:, 2] > 1e-6, -origin[2] / dirs[:, 2], np.inf)     # ground plane z = 0 (NED)
        t_best[~(t_best > 0)] = np.inf
        gp = np.nan_to_num(origin[None, :2] + t_best[:, None] * dirs[:, :2], posinf=0.0, neginf=0.0)
        shade = 0.35 + 0.1 * ((np.floor(gp[:, 0] / 2) + np.floor(gp[:, 1] / 2)) % 2)
        if len(world):                                   # obstacles inside the horizontal field of view and range
            rel_b = (world._c - origin[None]) @ rotation                                  # body FRD
            extent = world._r + world._hh
            half = 0.5 * self.width / self.f                                              # tan(hfov / 2)
            sel = np.nonzero((rel_b[:, 0] > -extent)
                             & (np.abs(rel_b[:, 1]) - half * rel_b[:, 0] <= extent * math.sqrt(1.0 + half * half))
                             & (np.linalg.norm(rel_b[:, :2], axis=1) < self.far + extent))[0]
            for kind in ("cylinder", "sphere"):
                for idx in np.array_split(sel[world._kind[sel] == kind], max(1, len(sel) // 48)):
                    if idx.size == 0:
                        continue
                    oc = origin[None] - world._c[idx]                                     # [K, 3]
                    dims = slice(0, 2) if kind == "cylinder" else slice(0, 3)
                    a = (dirs[:, dims] ** 2).sum(axis=1)[None]                           # [1, P]
                    b = 2.0 * oc[:, dims] @ dirs[:, dims].T                               # [K, P]
                    c = ((oc[:, dims] ** 2).sum(axis=1) - world._r[idx] ** 2)[:, None]    # [K, 1]
                    disc = b * b - 4.0 * a * c
                    with np.errstate(divide="ignore", invalid="ignore"):
                        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
                    ok = (disc >= 0.0) & (t > 0.0)
                    if kind == "cylinder":
                        ok &= np.abs(oc[:, 2:3] + t * dirs[None, :, 2]) <= world._hh[idx, None]
                    t = np.where(ok, t, np.inf)
                    k = t.argmin(axis=0)
                    t_min = t[k, np.arange(t.shape[1])]
                    hit = t_min < t_best
                    t_best[hit] = t_min[hit]
                    shade[hit] = 0.15 + 0.5 * ((idx[k[hit]] * 0.618) % 1.0)
        depth = t_best.reshape(self.height, self.width)                                 # z-depth along body x
        sky = ~np.isfinite(depth)
        rgb = np.empty((self.height, self.width, 3))
        base = shade.reshape(self.height, self.width)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.9 * base, base, 0.7 * base
        rgb[sky] = (0.55, 0.7, 0.9)
        if rng is not None:
            rgb += rng.normal(0.0, 0.01, rgb.shape)
        depth = np.where((depth >= self.near) & (depth <= self.far), depth, np.inf).astype(np.float32)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth


class KinematicQuadrotor:
    """First-order velocity and yaw-rate tracking with attitude from the commanded acceleration."""

    def __init__(self, position: Sequence[float], yaw: float, tau_v: float = 0.15, tau_r: float = 0.1) -> None:
        self.p = np.asarray(position, dtype=np.float64).copy()
        self.v = np.zeros(3)
        self.acc = np.zeros(3)
        self.yaw, self.yaw_rate = float(yaw), 0.0
        self.tau_v, self.tau_r = tau_v, tau_r

    def step(self, dt: float, vel_cmd: np.ndarray, acc_ff: np.ndarray, yaw_rate_cmd: float) -> None:
        self.acc = (np.asarray(vel_cmd) - self.v) / self.tau_v + np.asarray(acc_ff)
        self.v = self.v + dt * self.acc
        self.p = self.p + dt * self.v
        self.yaw_rate += dt * (yaw_rate_cmd - self.yaw_rate) / self.tau_r
        self.yaw = float(wrap_angle(self.yaw + dt * self.yaw_rate))

    def rotation(self) -> np.ndarray:
        """Body -> NED: thrust axis -z_B along the specific force a - g, heading yaw."""
        f = self.acc - np.array([0.0, 0.0, GRAVITY])
        z_b = -f / np.linalg.norm(f)
        x_c = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        return np.stack([x_b, y_b, z_b], axis=1)

    def quaternion(self) -> np.ndarray:
        r = self.rotation()
        w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
        x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
        y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
        z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
        return np.array([w, x, y, z])


@dataclass
class DryRunConfig:
    num_episodes: int = 2
    seed: int = 0
    camera_rate: float = 50.0
    control_rate: float = 100.0
    physics_rate: float = 250.0
    speed_limits: Tuple[float, ...] = (3.0, 5.0, 7.0)
    num_waypoints: Tuple[int, int] = (3, 6)
    spacing: Tuple[float, float] = (10.0, 40.0)
    altitude: Tuple[float, float] = (1.5, 4.0)
    max_frames: Optional[int] = None
    thresholds: LabelThresholds = field(default_factory=LabelThresholds)


def dry_run(path: str, world: PrivilegedMap, cfg: DryRunConfig, attrs: Optional[Dict[str, Any]] = None) -> List[str]:
    """Collect ``cfg.num_episodes`` expert episodes in the kinematic simulator into ``path``.

    Exercises exactly the recording code of the ROS collector (``record_frame``), so a dry-run file has
    the schema of a Gazebo file; only the dynamics and the images are synthetic.
    """
    rng = np.random.default_rng(cfg.seed)
    planner = ExpertPlanner(world)
    camera = PinholeCamera()
    writer = DatasetWriter(path, {"source": "dry_run (kinematic model, ray-cast RGB-D)",
                                  "label_thresholds": _json(asdict(cfg.thresholds)),
                                  "label_command": "first_knot", **(attrs or {})})
    names = []
    try:
        for _ in range(cfg.num_episodes):
            start = np.array([0.0, 0.0, -2.0])
            waypoints = sample_mission(world, rng, start, cfg.num_waypoints, cfg.spacing, cfg.altitude)
            speed = float(rng.choice(cfg.speed_limits))
            first = planner.plan(start, waypoints, speed, 0.0)
            yaw0 = first.initial_heading() or 0.0
            trajectory = planner.plan(start, waypoints, speed, yaw0)
            vehicle = KinematicQuadrotor(start, yaw0)
            controller = ExpertController(trajectory, 0.0, CommandLimits(speed_h=speed))
            tracker = WaypointTracker(waypoints)
            name, ep = writer.begin_episode()
            last_setpoint = np.zeros(4)
            t, dt = 0.0, 1.0 / cfg.physics_rate
            next_ctrl, next_cam = 0.0, 0.0
            vel_cmd, acc_ff, yr = np.zeros(3), np.zeros(3), 0.0
            collided, end_time = False, trajectory.duration + 1.0
            while t < end_time + 1e-9:
                if t >= next_cam - 1e-9:
                    rgb, depth = camera.render(world, vehicle.p, vehicle.rotation(), rng)
                    q = vehicle.quaternion()
                    sample = record_frame(t, rgb, depth, vehicle.p, vehicle.v, q, np.array([0.0, 0.0, vehicle.yaw_rate]),
                                          last_setpoint, controller, tracker, cfg.thresholds, odometry_age=0.0)
                    ep.append(sample)
                    next_cam += 1.0 / cfg.camera_rate
                    if cfg.max_frames is not None and len(ep) >= cfg.max_frames:
                        break
                if t >= next_ctrl - 1e-9:
                    if len(world) and float(world.distance(vehicle.p)) < 0.3:
                        collided = True
                        break
                    vel_cmd, acc_ff, yr = controller.command(t, vehicle.p, vehicle.yaw)
                    last_setpoint = np.concatenate([vel_cmd, [yr]])
                    next_ctrl += 1.0 / cfg.control_rate
                vehicle.step(dt, vel_cmd, acc_ff, yr)
                t += dt
            success = bool(tracker.task_complete and not collided)
            ep.finalize({"waypoints_ned": waypoints.astype(np.float64), "speed_limit": speed, "success": success,
                         "termination": "collision" if collided else ("complete" if success else "timeout"),
                         "instruction": "", "world": "dry_run", "seed": cfg.seed})
            names.append(name)
    finally:
        writer.close()
    return names


def record_frame(stamp: float, rgb: np.ndarray, depth_m: np.ndarray, position: np.ndarray, velocity_ned: np.ndarray,
                 q: np.ndarray, body_rates: np.ndarray, last_setpoint: np.ndarray, controller: ExpertController,
                 tracker: WaypointTracker, thresholds: LabelThresholds, odometry_age: float) -> FrameSample:
    """Everything recorded for one camera frame captured at ``stamp`` (shared by the collector and the dry run).

    ``position``, ``velocity_ned``, ``q`` and ``body_rates`` are the state estimate at the capture time;
    ``last_setpoint`` is the NED setpoint [4] dispatched most recently before it.
    """
    yaw = yaw_from_quat(q)
    state = proprioception(velocity_ned, q, body_rates, last_setpoint, yaw)
    goal = tracker.step(position, yaw)                                    # Eq. 14, then Eq. 3
    action = controller.chunk(stamp, position, yaw)                       # A_t in H_t
    label = int(primitive_labels(action[:1], np.array([tracker.distance_to_final(position)]), thresholds)[0])
    p_ref = controller.trajectory.reference(controller.elapsed(stamp))[0]
    return FrameSample(
        timestamp=float(stamp), image=encode_rgbd(rgb, depth_m), state=state, goal=goal, action=action,
        primitive=label, position=np.asarray(position, dtype=np.float32), yaw=float(yaw),
        waypoint_index=int(tracker.m), task_complete=bool(tracker.task_complete),
        reference_position=np.asarray(p_ref, dtype=np.float32), odometry_age=float(odometry_age),
    )


def synthetic_forest(seed: int = 0, half_size: float = 50.0, spacing: float = 6.0,
                     clear_radius: float = 4.0) -> PrivilegedMap:
    """A forest of trunk cylinders and canopy spheres in NED (dry runs and tests; Gazebo uses make_world.py)."""
    rng = np.random.default_rng(seed)
    obstacles = []
    for x in np.arange(-half_size, half_size + 1e-9, spacing):
        for y in np.arange(-half_size, half_size + 1e-9, spacing):
            p = np.array([x, y]) + rng.uniform(-0.35, 0.35, 2) * spacing
            if np.linalg.norm(p) < clear_radius:
                continue
            r, h = rng.uniform(0.12, 0.3), rng.uniform(7.0, 11.0)
            obstacles.append(Obstacle("cylinder", (p[0], p[1], -0.5 * h), radius=r, half_height=0.5 * h))
            obstacles.append(Obstacle("sphere", (p[0], p[1], -h), radius=rng.uniform(1.0, 1.8)))
    return PrivilegedMap(obstacles, ((-half_size, -half_size, -np.inf), (half_size, half_size, np.inf)))


# ---------------------------------------------------------------------------------------------
# Self-test (run: python sim/expert.py)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import os
    import sys
    import tempfile

    rng = np.random.default_rng(3)

    # ---- frames: R_z round trip, gravity from a known attitude, S_t layout
    v = rng.normal(size=(5, 3))
    psi = rng.uniform(-3, 3, 5)
    assert np.allclose(heading_to_world(world_to_heading(v, psi), psi), v)
    roll = 0.3
    q_roll = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    assert np.allclose(gravity_in_body(q_roll), [0.0, math.sin(roll), math.cos(roll)])
    q_yaw = np.array([math.cos(0.4), 0.0, 0.0, math.sin(0.4)])
    assert math.isclose(yaw_from_quat(q_yaw), 0.8)
    s = proprioception([1.0, 0.0, 0.0], q_yaw, [0, 0, 0.1], [0.0, 2.0, 0.0, 0.3], 0.8)
    assert s.shape == (PROPRIO_DIM,) and s.dtype == np.float32
    assert np.allclose(s[:3], world_to_heading(np.array([1.0, 0, 0]), 0.8), atol=1e-6)
    assert np.allclose(s[9:13], [*world_to_heading(np.array([0.0, 2.0, 0.0]), 0.8), 0.3], atol=1e-6)

    # ---- goal feature and waypoint pointer
    g = goal_feature([10.0, 0, 0], [0, 0, 0], 0.0)
    assert np.allclose(g, [1, 0, 0, 0.5])
    tr = WaypointTracker(np.array([[1.0, 0, 0], [5.0, 0, 0]]))
    tr.step([0.5, 0, 0], 0.0)
    assert tr.m == 2 and not tr.task_complete
    tr.step([4.5, 0, 0], 0.0)
    assert tr.m == 2 and tr.task_complete

    # ---- labels: priority order of Sec. 3.3
    th = LabelThresholds()
    cmd = np.array([[0.1, 0, 0, 0], [3, 0, 0, 0], [3, 0, 1, 0], [3, 1, 0, 0], [3, 0, 0, 1], [3, 1, 1, 1]])
    assert primitive_labels(cmd, np.full(6, 10.0), th).tolist() == [BRAKE_HOVER, CRUISE, VERTICAL_AVOID,
                                                                       LATERAL_AVOID, YAW_SCAN, VERTICAL_AVOID]
    assert primitive_labels(cmd[1:2], np.array([2.0]), th).tolist() == [APPROACH]

    # ---- min-snap: interpolation, rest at both ends, C^6 at interior knots, p^(8) = 0
    via = np.array([[0, 0, -2], [10, 3, -3], [18, -4, -2.5], [25, 2, -2]], dtype=float)
    spline = MinSnapSpline(via, np.array([3.0, 2.5, 2.0]))
    assert np.allclose(spline.evaluate(spline.knots), via, atol=1e-8)
    for m in (1, 2, 3):
        assert np.allclose(spline.evaluate(np.array([0.0, spline.duration]), m), 0.0, atol=1e-7)
    for knot in spline.knots[1:-1]:
        for m in range(7):
            lo, hi = spline.evaluate(knot - 1e-9, m), spline.evaluate(knot + 1e-9, m)
            assert np.allclose(lo, hi, rtol=1e-4, atol=1e-4 * (1 + np.abs(lo).max())), (knot, m)

    # ---- planner in a forest: clearance, limits, timing
    world = synthetic_forest(seed=1, half_size=30.0)
    planner = ExpertPlanner(world)
    mission = sample_mission(world, rng, [0, 0, -2], (3, 3), (12.0, 20.0))
    traj = planner.plan([0, 0, -2], mission, speed=5.0, yaw0=0.0)
    ts = np.arange(0.0, traj.duration, 0.02)
    pos, vel = traj.spline.evaluate(ts), traj.spline.evaluate(ts, 1)
    assert world.distance(pos).min() >= planner.cfg.clearance - 1e-6
    assert np.linalg.norm(vel[:, :2], axis=1).max() <= 5.0 * 1.02
    assert np.linalg.norm(traj.spline.evaluate(ts, 2), axis=1).max() <= planner.cfg.accel_limit * 1.05
    for wp in mission:                                        # every mission waypoint is on the spline
        assert np.linalg.norm(pos - wp, axis=1).min() < 0.15

    # ---- controller: knot 0 is the dispatched command in H_t; error decay; limits
    ctl = ExpertController(traj, 100.0, CommandLimits(speed_h=5.0))
    p_hat, yaw_hat = pos[50] + np.array([0.3, -0.2, 0.1]), 0.4
    vel_cmd, _, yr = ctl.command(101.0, p_hat, yaw_hat)
    chunk = ctl.chunk(101.0, p_hat, yaw_hat)
    assert chunk.shape == (HORIZON, ACTION_DIM) and chunk.dtype == np.float32
    assert np.allclose(chunk[0, :3], world_to_heading(vel_cmd, yaw_hat), atol=1e-5)
    assert math.isclose(chunk[0, 3], yr, abs_tol=1e-5)
    assert np.all(np.abs(chunk[:, 3]) <= 1.5 + 1e-6) and np.all(np.linalg.norm(chunk[:, :2], axis=1) <= 5.0 + 1e-5)

    # ---- lookahead targets: exactly 5 frames at 50 Hz, -1 at the end and across a gap
    stamps = np.arange(40) * 0.02 + 7.0
    tgt = future_targets(stamps)
    assert np.all(tgt[:35] == np.arange(35) + 5) and np.all(tgt[35:] == -1)
    gappy = np.delete(stamps, 12)
    assert future_targets(gappy)[7] == -1 and future_targets(gappy)[6] == 11 and future_targets(gappy)[0] == 5

    # ---- image encoding round trip
    rgb = rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
    depth = rng.uniform(0.2, 19.0, IMAGE_SIZE).astype(np.float32)
    depth[0, :5] = [np.inf, -np.inf, np.nan, 0.0, -1.0]
    enc = encode_rgbd(rgb, depth)
    rgb2, depth2 = decode_rgbd(enc)
    assert enc.dtype == np.uint16 and enc.shape == (*IMAGE_SIZE, 4) and np.array_equal(rgb2, rgb)
    assert np.all(depth2[0, :5] == 0) and np.abs(depth2[1:] - depth[1:]).max() <= 0.5e-3 + 1e-6

    # ---- dry run: two episodes, schema, a_bar = previous setpoint, labels reproducible
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dry.hdf5")
        names = dry_run(path, world, DryRunConfig(num_episodes=2, seed=4, spacing=(8.0, 12.0), num_waypoints=(3, 3),
                                                  speed_limits=(7.0,)))
        with h5py.File(path, "r") as f:
            assert f.attrs["schema"] == SCHEMA_NAME and names == ["episode_0000", "episode_0001"]
            for name in names:
                ep = f[name]
                n = ep["timestamps"].shape[0]
                assert ep["observations/images"].shape == (n, *IMAGE_SIZE, 4)
                assert ep["observations/state"].shape == (n, PROPRIO_DIM)
                assert ep["observations/goal"].shape == (n, GOAL_DIM)
                assert ep["actions"].shape == (n, HORIZON, ACTION_DIM) and ep["primitives"].shape == (n,)
                assert ep["future_targets"].dtype == np.int64 and bool(ep.attrs["success"])
                ft = ep["future_targets"][...]
                assert np.all(ft[ft >= 0] - np.nonzero(ft >= 0)[0] == 5)
                labels = ep["primitives"][...]
                assert labels.min() >= 0 and labels.max() < len(PRIMITIVES)
                assert ep["privileged/task_complete"][-1] and bool(ep.attrs["complete"])
        with h5py.File(path, "a") as f:                  # an episode left unfinished by an interrupted writer
            f.create_group("episode_0002").attrs["complete"] = False
        writer = DatasetWriter(path, mode="a")           # ... is removed on reopening, numbering continues
        assert "episode_0002" not in writer.file
        name, _ = writer.begin_episode()
        writer.discard(name)
        writer.close()
        with h5py.File(path, "r") as f:
            assert name == "episode_0002" and sorted(f.keys()) == ["episode_0000", "episode_0001"]

    # ---- cross-check against the PyTorch reference implementation (selector.py, config.py)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    try:
        import torch

        import config
        import selector
    except ImportError as exc:  # the ROS collector environment may not have PyTorch
        print(f"  (cross-check with selector.py skipped: {exc})")
    else:
        cfg = config.VPAConfig()
        assert PRIMITIVES == config.PRIMITIVES == selector.PRIMITIVES
        assert (cfg.horizon, cfg.action_dim, cfg.proprio_dim, cfg.goal_dim) == (HORIZON, ACTION_DIM, PROPRIO_DIM, GOAL_DIM)
        assert tuple(cfg.image_size) == IMAGE_SIZE and cfg.action_scale == ACTION_SCALE
        assert math.isclose(cfg.knot_dt, KNOT_DT) and math.isclose(cfg.predictor_horizon_s, PREDICTOR_HORIZON_S)
        wp, p, yaw = rng.normal(size=(8, 3)) * 10, rng.normal(size=(8, 3)) * 10, rng.uniform(-3, 3, 8)
        ref = selector.goal_feature(torch.tensor(wp), torch.tensor(p), torch.tensor(yaw), cfg.goal_saturation_distance,
                                    cfg.goal_eps).numpy()
        ours = np.stack([goal_feature(wp[i], p[i], yaw[i], cfg.goal_saturation_distance, cfg.goal_eps) for i in range(8)])
        assert np.allclose(ours, ref, atol=1e-6)
        assert np.allclose(world_to_heading(wp, yaw), selector.world_to_heading(torch.tensor(wp), torch.tensor(yaw)).numpy())
        cmds = rng.normal(size=(200, 4)) * np.array([2, 1, 1, 1])
        dist = rng.uniform(0, 8, 200)
        ref_labels = selector.primitive_labels(torch.tensor(cmds), torch.tensor(dist), **asdict(th)).numpy()
        assert np.array_equal(primitive_labels(cmds, dist, th), ref_labels)
        trk_ref = selector.WaypointTracker(cfg.waypoint_radius, cfg.goal_saturation_distance, cfg.goal_eps)
        wps = np.array([[2.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0]])
        trk_ref.reset(torch.tensor(wps)[None])
        ours_trk = WaypointTracker(wps, cfg.waypoint_radius, cfg.goal_saturation_distance, cfg.goal_eps)
        for x in np.linspace(0, 6.5, 30):
            out = trk_ref.step(torch.tensor([[x, 0.1, 0]]), torch.tensor([0.2]))
            g_ours = ours_trk.step([x, 0.1, 0], 0.2)
            assert int(out.m_t) == ours_trk.m and bool(out.task_complete) == ours_trk.task_complete
            assert np.allclose(out.goal.numpy()[0], g_ours, atol=1e-6)
    print("sim/expert.py self-test passed")


if __name__ == "__main__":
    _self_test()
