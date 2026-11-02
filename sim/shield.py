# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/shield.py -- An onboard safety shield for a flying policy (sim/fly_policy.py --shield), without the map.

It uses only what the vehicle has: the depth camera, the odometry, the mission's waypoints and the policy's own
outputs (sigma_bar, decision times). It filters every setpoint of the executor and takes over when needed:

    memory      each decision adds the depth returns within ``memory_range`` (the nearest return of every 4 x 4
                pixel block, back-projected with the camera of clearance.DepthRange; ground returns dropped) to a
                local obstacle memory in NED, voxelised; points beside and behind the camera stay remembered
                for ``memory_s`` (static forest);
    barrier     the commanded velocity's approach speed toward each remembered point near the position predicted
                ``lookahead_s`` ahead at the flown velocity is at most min(gain (d - radius), sqrt(2 decel
                (d - radius))), negative inside the radius (it moves away); motion past an obstacle is kept;
    blind       horizontal motion in a direction the camera does not see (outside the HFOV) is capped at
                ``blind_speed`` unless it follows the breadcrumbs (the path flown so far, known free) back;
    uncertainty sigma_bar above sigma_lo caps the speed at ``sigma_speed`` (off by default);
    latency     no decision for ``stale_s``: hold (zero velocity) until the next one;
    altitude    below ``min_altitude`` above home ground: climb at ``climb_speed`` at least;
    backoff     closer than ``hard_radius`` to a remembered point: retrace the breadcrumbs for ``backoff_m`` at
                ``backoff_speed`` (through the barrier; it ends when the barrier blocks it), then nudge toward the active waypoint (``nudge_s`` at ``nudge_speed``, turning
                the camera to it, through the barrier), so that a policy that repeats its last command does not
                stay at rest;
    stall       no progress toward the active waypoint for ``stall_s``: the same nudge.

While the shield only filters, the executor keeps running and the filtered setpoint becomes a_bar (what was
dispatched); while it flies (backoff, nudge, hold), the executor is not ticked and the setpoint is handed to the
pipeline as an external one, as during a DAgger takeover. ``summary()`` goes into the flight record.

    python3 sim/shield.py      # self-test
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from clearance import DepthRange  # noqa: E402

__all__ = ["ShieldConfig", "SafetyShield", "Setpoint", "FLY", "BACKOFF", "NUDGE", "HOLD"]

FLY, BACKOFF, NUDGE, HOLD = "fly", "backoff", "nudge", "hold"


@dataclass
class ShieldConfig:
    radius: float = 0.55            # barrier zero: vehicle centre to a remembered surface point (m; a collision
                                    # is 0.35 m to the surface, and a trunk's side may be nearer than the points seen)
    hard_radius: float = 0.45       # back off when closer than this (m)
    gain: float = 1.5               # approach speed allowed toward a point: gain * (d - radius) (1/s) ...
    decel: float = 3.0              # ... and sqrt(2 decel (d - radius)) (m/s^2)
    push_speed: float = 0.5         # at most this fast away from a point inside the radius (m/s)
    lookahead_s: float = 0.4        # distances from the position this far ahead at the flown velocity (s)
    max_points: int = 24            # nearest points the barrier constrains
    memory_range: float = 8.0       # depth returns kept (m)
    memory_s: float = 15.0          # forget points older than this (s)
    memory_keep: float = 10.0       # and farther than this horizontally (m)
    neighbourhood: float = 3.5      # points the barrier looks at, from the position at the decision (m)
    voxel: float = 0.15             # memory resolution (m)
    block: int = 4                  # nearest return of each block x block pixels
    near: float = 0.2               # depth returns below this are invalid (m)
    ground_altitude: float = 0.25   # returns below this height are ground (m)
    min_altitude: float = 0.8       # climb below this height (m)
    climb_speed: float = 0.5        # (m/s)
    stale_s: float = 0.3            # hold after this long without a decision (s)
    sigma_speed: float = math.inf   # speed cap while sigma_bar > sigma_lo (m/s; off: in SITL it held the vehicle
                                    # at 1 m/s for 18 s of every mission and turned flights into timeouts)
    blind_speed: float = 0.7        # horizontal speed allowed in directions the camera does not see (m/s)
    crumb_spacing: float = 0.25     # breadcrumb every this far (m)
    backoff_m: float = 1.0          # retrace this far (m)
    backoff_speed: float = 0.3      # (m/s)
    stall_s: float = 5.0            # nudge after this long without 0.3 m of progress (s)
    nudge_s: float = 1.0            # (s, after the camera faces the waypoint)
    nudge_speed: float = 0.6        # (m/s)
    nudge_yaw_rate: float = 0.8     # (rad/s)

    @classmethod
    def from_args(cls, args: Any) -> "ShieldConfig":
        """The defaults, with ``args.shield_<field>`` where given (fly_policy.py options)."""
        cfg = cls()
        for f in fields(cls):
            value = getattr(args, "shield_" + f.name, None)
            if value is not None:
                setattr(cfg, f.name, type(getattr(cfg, f.name))(value))
        return cfg


class Setpoint(NamedTuple):
    """What to dispatch: NED velocity, NED acceleration feed-forward, yaw rate; ``takeover``: the shield flies
    (do not tick the executor; hand the setpoint to the pipeline as external); ``changed``: it filtered the
    executor's setpoint (a_bar should be the filtered one)."""

    velocity: np.ndarray
    acceleration: np.ndarray
    yaw_rate: float
    takeover: bool
    changed: bool


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SafetyShield:
    """See the module docstring. One instance per mission (``reset`` at its start)."""

    def __init__(self, cfg: Optional[ShieldConfig] = None, camera: Optional[DepthRange] = None) -> None:
        self.cfg = cfg or ShieldConfig()
        cam = camera or DepthRange()
        self.width, self.height, self.f = cam.width, cam.height, cam.f
        b = self.cfg.block
        hb, wb = self.height // b, self.width // b
        u = (np.arange(wb) + 0.5) * b - 0.5 * self.width
        v = (np.arange(hb) + 0.5) * b - 0.5 * self.height
        uu, vv = np.meshgrid(u, v)
        self.rays = np.stack([np.ones_like(uu), uu / self.f, vv / self.f], axis=-1).reshape(-1, 3)   # body FRD, x = 1
        self.half_fov = math.atan(0.5 * self.width / self.f)
        self.reset(0.0)

    # ------------------------------------------------------------------ state
    def reset(self, t0: float, position: Optional[np.ndarray] = None) -> None:
        self.points = np.zeros((0, 3))
        self.stamps = np.zeros(0)
        self.local = np.zeros((0, 3))
        self.crumbs: List[np.ndarray] = [] if position is None else [np.asarray(position, dtype=np.float64).copy()]
        self.mode, self.mode_since = FLY, float(t0)
        self.last_frame = float(t0)
        self.sigma_high = False
        self.backoff_left, self.nudge_until, self.nudge_end = 0.0, 0.0, 0.0
        self.prev_p: Optional[np.ndarray] = None
        self.target_key: Optional[Tuple[float, ...]] = None
        self.progress_t, self.progress_d = float(t0), math.inf
        self.last_t: Optional[float] = None
        self.reasons: Tuple[str, ...] = ()
        self.frame_ms: List[float] = []
        self.min_memory_clearance = math.inf
        self.counts = {"backoffs": 0, "nudges": 0, "stalls": 0, "holds": 0}
        self.seconds = {k: 0.0 for k in ("barrier", "blind", "sigma", "altitude", "stale", BACKOFF, NUDGE)}

    def summary(self) -> Dict[str, Any]:
        """For the flight record: counts, seconds with each rule active, the smallest clearance to the memory."""
        return {"config": asdict(self.cfg), **self.counts, "seconds": {k: round(v, 2) for k, v in self.seconds.items()},
                "min_memory_clearance_m": None if not math.isfinite(self.min_memory_clearance)
                else round(self.min_memory_clearance, 3),
                "memory_points": int(len(self.points)),
                "frame_ms_p95": round(float(np.percentile(self.frame_ms, 95)), 2) if self.frame_ms else None}

    # ------------------------------------------------------------------ perception
    def on_frame(self, t: float, depth_m: np.ndarray, rotation_body_to_ned: np.ndarray, position: np.ndarray,
                 sigma_bar: Optional[float] = None, sigma_lo: Optional[float] = None) -> None:
        """Add a depth frame (``[H, W]`` z-depth in m, 0 or non-finite = no return) taken at ``position`` with
        attitude ``rotation_body_to_ned``; note the decision time and whether sigma_bar exceeds sigma_lo."""
        started = time.perf_counter()
        c = self.cfg
        self.last_frame = max(self.last_frame, float(t))
        self.sigma_high = (sigma_bar is not None and sigma_lo is not None and math.isfinite(float(sigma_lo))
                           and float(sigma_bar) > float(sigma_lo))
        d = np.asarray(depth_m, dtype=np.float64)
        b = c.block
        hb, wb = self.height // b, self.width // b
        blocks = d[:hb * b, :wb * b].reshape(hb, b, wb, b)
        z = np.where(np.isfinite(blocks) & (blocks >= c.near), blocks, np.inf).min(axis=(1, 3)).reshape(-1)
        keep = np.isfinite(z) & (z <= c.memory_range)
        p = np.asarray(position, dtype=np.float64)
        pts = p[None, :] + (z[keep, None] * self.rays[keep]) @ np.asarray(rotation_body_to_ned, dtype=np.float64).T
        pts = pts[-pts[:, 2] > c.ground_altitude]                                       # NED: height = -z
        allp = np.vstack([self.points, pts])
        allt = np.concatenate([self.stamps, np.full(len(pts), float(t))])
        fresh = (float(t) - allt <= c.memory_s) & (np.linalg.norm(allp[:, :2] - p[None, :2], axis=1) <= c.memory_keep)
        allp, allt = allp[fresh], allt[fresh]
        if len(allp):
            k = np.floor(allp / c.voxel).astype(np.int64)
            key = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
            order = np.argsort(-allt, kind="stable")                                   # newest first
            _, first = np.unique(key[order], return_index=True)
            sel = order[first]
            allp, allt = allp[sel], allt[sel]
        self.points, self.stamps = allp, allt
        self.local = self.points[np.linalg.norm(self.points - p[None, :], axis=1) <= c.neighbourhood]
        self.frame_ms.append(1e3 * (time.perf_counter() - started))

    def clearance(self, position: np.ndarray) -> float:
        """Distance from ``position`` to the nearest remembered point near it (inf without one)."""
        if not len(self.local):
            return math.inf
        return float(np.linalg.norm(self.local - np.asarray(position, dtype=np.float64)[None, :], axis=1).min())

    # ------------------------------------------------------------------ control
    def step(self, t: float, position: np.ndarray, velocity: np.ndarray, yaw: float, target: Optional[np.ndarray],
             executor: Callable[[], Tuple[np.ndarray, np.ndarray, float]]) -> Setpoint:
        """The setpoint at ``t`` from the state estimate; ``executor()`` ticks the policy's executor (called only
        when the shield does not fly itself) and returns its (velocity, acceleration, yaw rate)."""
        c = self.cfg
        t = float(t)
        p, v = np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        moved = 0.0 if self.prev_p is None else float(np.linalg.norm(p - self.prev_p))
        self.prev_p = p.copy()
        if not self.crumbs:
            self.crumbs.append(p.copy())
        if self.mode != BACKOFF:
            self._crumb(p)
        self._progress(t, p, target)
        near = self.clearance(p)
        self.min_memory_clearance = min(self.min_memory_clearance, near)
        was_backing_off = self.mode == BACKOFF
        if self.mode == FLY:
            if near < c.hard_radius and len(self.crumbs) >= 2:
                self._enter(BACKOFF, t)
                self.backoff_left = c.backoff_m
                self.counts["backoffs"] += 1
            elif target is not None and t - self.progress_t > c.stall_s:
                self._start_nudge(t)
                self.counts["stalls"] += 1
        reasons: List[str] = []
        if self.mode == BACKOFF:
            if was_backing_off:
                self.backoff_left -= moved
            cmd = self._backoff(p, near)
            if cmd is not None:                    # the barrier and the altitude floor hold during the backoff too
                cmd, _, _ = self._filters(p, v, yaw, cmd, np.zeros(3), reasons)
                if float(np.linalg.norm(cmd)) < 0.25 * c.backoff_speed:
                    cmd = None                         # blocked behind as well: stop backing off
            if cmd is None:
                self._start_nudge(t) if target is not None else self._enter(FLY, t)
                reasons = []
            else:
                self.seconds[BACKOFF] += dt
                self.reasons = (BACKOFF, *reasons)
                return Setpoint(cmd, np.zeros(3), 0.0, True, True)
        if self.mode == NUDGE:
            if target is None or t >= self.nudge_until:
                self._enter(FLY, t)
            else:
                vel, yaw_rate = self._nudge(t, p, yaw, target)
                vel, acc, _ = self._filters(p, v, yaw, vel, np.zeros(3), reasons)
                self.seconds[NUDGE] += dt
                self.reasons = (NUDGE, *reasons)
                return Setpoint(vel, acc, yaw_rate, True, True)
        if t - self.last_frame > c.stale_s:
            self.seconds["stale"] += dt
            if self.mode != HOLD:
                self.counts["holds"] += 1
                self._enter(HOLD, t)
            self.reasons = ("stale",)
            return Setpoint(np.zeros(3), np.zeros(3), 0.0, True, True)
        if self.mode == HOLD:
            self._enter(FLY, t)
        vel, acc, yaw_rate = executor()
        vel0 = np.asarray(vel, dtype=np.float64)
        vel, acc, changed = self._filters(p, v, yaw, vel0.copy(), np.asarray(acc, dtype=np.float64).copy(), reasons)
        for r in reasons:
            self.seconds[r] += dt
        self.reasons = tuple(reasons)
        return Setpoint(vel, acc, float(yaw_rate), False, changed)

    # ------------------------------------------------------------------ pieces
    def _enter(self, mode: str, t: float) -> None:
        self.mode, self.mode_since = mode, t

    def _start_nudge(self, t: float) -> None:
        self._enter(NUDGE, t)
        self.nudge_end = t + self.cfg.nudge_s
        self.nudge_until = t + self.cfg.nudge_s + 3.0          # turning first may take up to 3 s more
        self.progress_t = t
        self.counts["nudges"] += 1

    def _crumb(self, p: np.ndarray) -> None:
        if np.linalg.norm(p - self.crumbs[-1]) >= self.cfg.crumb_spacing:
            self.crumbs.append(p.copy())
            if len(self.crumbs) > 400:
                del self.crumbs[0]

    def _progress(self, t: float, p: np.ndarray, target: Optional[np.ndarray]) -> None:
        if target is None:
            self.target_key, self.progress_t = None, t
            return
        key = tuple(np.round(np.asarray(target, dtype=np.float64), 2))
        d = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - p))
        if key != self.target_key:
            self.target_key, self.progress_t, self.progress_d = key, t, d
        elif d < self.progress_d - 0.3:
            self.progress_t, self.progress_d = t, d

    def _backoff(self, p: np.ndarray, near: float) -> Optional[np.ndarray]:
        """Velocity toward the previous breadcrumb, or None when the backoff is over."""
        c = self.cfg
        if self.backoff_left <= 0.0 or near >= c.radius + 0.4:
            return None
        while len(self.crumbs) >= 2 and np.linalg.norm(self.crumbs[-1] - p) < 0.15:
            self.crumbs.pop()
        if len(self.crumbs) < 2 and np.linalg.norm(self.crumbs[-1] - p) < 0.15:
            return None
        d = self.crumbs[-1] - p
        n = float(np.linalg.norm(d))
        return c.backoff_speed * d / max(n, 1e-6) * min(1.0, n / 0.15)

    def _nudge(self, t: float, p: np.ndarray, yaw: float, target: np.ndarray) -> Tuple[np.ndarray, float]:
        c = self.cfg
        d = np.asarray(target, dtype=np.float64) - p
        rel = _wrap(math.atan2(d[1], d[0]) - yaw) if np.linalg.norm(d[:2]) > 0.3 else 0.0
        yaw_rate = float(np.clip(1.5 * rel, -c.nudge_yaw_rate, c.nudge_yaw_rate))
        if abs(rel) > self.half_fov:                         # turn the camera to the waypoint first
            self.nudge_end = max(self.nudge_end, t + c.nudge_s)
            return np.zeros(3), yaw_rate
        if t >= self.nudge_end:
            self.nudge_until = t
        n = float(np.linalg.norm(d))
        return c.nudge_speed * d / max(n, 1e-6) * math.cos(rel), yaw_rate

    def _along_crumbs(self, p: np.ndarray, h: np.ndarray) -> bool:
        """Does the horizontal direction ``h`` point back along the breadcrumbs (within 35 deg of one 0.3-2.5 m
        away)?"""
        speed = float(np.linalg.norm(h))
        if speed < 1e-6:
            return True
        for crumb in reversed(self.crumbs[-16:]):
            d = crumb[:2] - p[:2]
            dist = float(np.linalg.norm(d))
            if 0.3 <= dist <= 2.5 and float(d @ h) >= math.cos(math.radians(35.0)) * dist * speed:
                return True
        return False

    def _filters(self, p: np.ndarray, v: np.ndarray, yaw: float, vel: np.ndarray, acc: np.ndarray,
                 reasons: List[str]) -> Tuple[np.ndarray, np.ndarray, bool]:
        c = self.cfg
        changed = False
        if -p[2] < c.min_altitude and vel[2] > -c.climb_speed:                          # NED: up is -z
            vel[2], acc[2] = -c.climb_speed, min(float(acc[2]), 0.0)
            reasons.append("altitude")
            changed = True
        speed = float(np.linalg.norm(vel))
        if self.sigma_high and speed > c.sigma_speed:
            vel *= c.sigma_speed / speed
            acc *= c.sigma_speed / speed
            reasons.append("sigma")
            changed = True
        if len(self.local):
            q = p + v * c.lookahead_s
            dv = self.local - q[None, :]
            dist = np.linalg.norm(dv, axis=1)
            idx = np.argsort(dist)[:c.max_points]
            hit = False
            for _ in range(2):
                for i in idx:
                    n = dv[i] / max(float(dist[i]), 1e-6)
                    gap = float(dist[i]) - c.radius
                    allow = c.gain * gap if gap <= 0.0 else min(c.gain * gap, math.sqrt(2.0 * c.decel * gap))
                    allow = max(allow, -c.push_speed)
                    approach = float(vel @ n)
                    if approach > allow + 1e-6:
                        vel = vel - (approach - allow) * n
                        acc = acc - max(0.0, float(acc @ n)) * n
                        hit = True
            if hit:
                reasons.append("barrier")
                changed = True
        # after the barrier, so that a slide around a trunk that leaves the camera's view is capped as well
        h = vel[:2]
        hs = float(np.linalg.norm(h))
        if hs > c.blind_speed and abs(_wrap(math.atan2(h[1], h[0]) - yaw)) > self.half_fov \
                and not self._along_crumbs(p, h):
            vel[:2] *= c.blind_speed / hs
            acc[:2] *= c.blind_speed / hs
            reasons.append("blind")
            changed = True
        return vel, acc, changed


# ---------------------------------------------------------------------------------------------
# Self-test (python3 sim/shield.py)
# ---------------------------------------------------------------------------------------------
def _depth_of_trunk(cam: DepthRange, rotation: np.ndarray, position: np.ndarray, center: np.ndarray, radius: float
                    ) -> np.ndarray:
    """z-depth image of one vertical cylinder (NED centre, radius) seen from ``position`` (ray casting)."""
    rays_ned = cam.rays.reshape(-1, 3) @ rotation.T                                    # z = 1 along body x
    o = np.asarray(position, dtype=np.float64)
    dx, dy = rays_ned[:, 0], rays_ned[:, 1]
    fx, fy = o[0] - center[0], o[1] - center[1]
    a = dx * dx + dy * dy
    bq = 2.0 * (fx * dx + fy * dy)
    cq = fx * fx + fy * fy - radius * radius
    disc = bq * bq - 4.0 * a * cq
    s = np.where(disc >= 0.0, (-bq - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(2.0 * a, 1e-12), np.inf)
    s = np.where(s > 0.0, s, np.inf)                                                     # s = z-depth (x-component 1)
    return s.reshape(cam.height, cam.width).astype(np.float32)


def _self_test() -> None:
    cam = DepthRange()
    level = np.eye(3)                                    # body = NED at yaw 0: x north, camera looks north
    sh = SafetyShield(camera=cam)
    p0 = np.array([0.0, 0.0, -2.0])
    trunk = np.array([2.0, 0.0])
    sh.reset(0.0, p0)
    depth = _depth_of_trunk(cam, level, p0, trunk, 0.2)
    sh.on_frame(0.0, depth, level, p0, sigma_bar=0.001, sigma_lo=0.01)
    assert len(sh.points) > 10 and abs(float(sh.points[:, 0].min()) - 1.8) < 0.1, sh.points[:, 0].min()
    assert abs(sh.clearance(p0) - 1.8) < 0.15 and not sh.sigma_high
    # ground returns (a camera pitched down sees the ground at height 0) are not obstacles
    sh2 = SafetyShield(camera=cam)
    sh2.reset(0.0, p0)
    pitch = math.radians(-30.0)                          # nose down
    rot = np.array([[math.cos(pitch), 0.0, -math.sin(pitch)], [0.0, 1.0, 0.0], [math.sin(pitch), 0.0, math.cos(pitch)]])
    rays = cam.rays.reshape(-1, 3) @ rot.T
    s = np.where(rays[:, 2] > 1e-6, (0.0 - p0[2]) / np.maximum(rays[:, 2], 1e-6), np.inf)
    sh2.on_frame(0.0, s.reshape(cam.height, cam.width), rot, p0)
    assert len(sh2.points) == 0, len(sh2.points)

    def run(shield, t, p, v, yaw, cmd, target=None):
        return shield.step(t, p, v, yaw, target, lambda: (np.asarray(cmd, dtype=np.float64), np.zeros(3), 0.0))

    # barrier: flying at 2 m/s at the trunk 1.8 m ahead is slowed to the allowed approach speed
    out = run(sh, 0.01, p0, np.array([2.0, 0.0, 0.0]), 0.0, [2.0, 0.0, 0.0])
    c = sh.cfg
    gap = 1.8 - 2.0 * c.lookahead_s - c.radius                                          # from the point 0.8 m on
    assert not out.takeover and out.changed and "barrier" in sh.reasons
    assert out.velocity[0] <= min(c.gain * gap, math.sqrt(2 * c.decel * gap)) + 0.05, out.velocity
    # motion past the trunk (sideways, in view: azimuth 20 deg) is kept
    out = run(sh, 0.02, p0, np.zeros(3), 0.0, [0.94, 0.34, 0.0])
    assert not out.changed and np.allclose(out.velocity, [0.94, 0.34, 0.0])
    # blind: sideways at 1.5 m/s (90 deg from the heading) is capped, unless back along the breadcrumbs
    out = run(sh, 0.03, p0, np.zeros(3), 0.0, [0.0, 1.5, 0.0])
    assert "blind" in sh.reasons and abs(float(np.linalg.norm(out.velocity[:2])) - c.blind_speed) < 1e-6
    sh3 = SafetyShield(camera=cam)
    sh3.reset(0.0, np.array([0.0, -2.0, -2.0]))
    for k in range(9):                                                                   # flown east to y = 0
        run(sh3, 0.01 * k, np.array([0.0, -2.0 + 0.25 * k, -2.0]), np.zeros(3), 0.0, [0.0, 0.0, 0.0])
    sh3.last_frame = 1.0
    out = run(sh3, 0.2, np.array([0.0, 0.0, -2.0]), np.zeros(3), 0.0, [0.0, -1.5, 0.0])  # back west: allowed
    assert not out.changed and abs(out.velocity[1] + 1.5) < 1e-9, out
    # altitude floor (and the uncertainty cap, off by default)
    assert not math.isfinite(sh.cfg.sigma_speed)
    sh.sigma_high = True
    out = run(sh, 0.04, np.array([0.0, 0.0, -0.6]), np.zeros(3), 0.0, [0.0, 0.3, 0.8])
    assert out.velocity[2] <= -0.5 + 1e-9 and "altitude" in sh.reasons
    out = run(sh, 0.05, p0 + np.array([-3.0, 0.0, 0.0]), np.zeros(3), 0.0, [0.0, 0.0, 0.0])
    sh.sigma_high = False
    # latency: no decision for 0.3 s -> hold; the next decision ends it
    out = run(sh, 0.5, p0, np.zeros(3), 0.0, [1.0, 0.0, 0.0])
    assert out.takeover and np.allclose(out.velocity, 0.0) and sh.mode == HOLD and sh.counts["holds"] == 1
    sh.on_frame(0.55, depth, level, p0)
    out = run(sh, 0.56, p0, np.zeros(3), 0.0, [0.5, 0.0, 0.0])
    assert sh.mode == FLY and not out.takeover
    # backoff: closer than the hard radius -> retrace the breadcrumbs, then nudge toward the waypoint
    sh4 = SafetyShield(camera=cam)
    sh4.reset(0.0, np.array([-1.0, 0.0, -2.0]))
    t = 0.0
    for k in range(5):                                                                   # flown north to x = 0
        t += 0.05
        run(sh4, t, np.array([-1.0 + 0.25 * k, 0.0, -2.0]), np.zeros(3), 0.0, [0.0, 0.0, 0.0])
    pc = np.array([1.75, 0.0, -2.0])                                                    # 0.05 m from the trunk surface
    sh4.on_frame(t, _depth_of_trunk(cam, level, pc - np.array([0.5, 0.0, 0.0]), trunk, 0.2), level, pc)
    sh4.local = np.vstack([sh4.local, [[1.8, 0.0, -2.0]]])
    sh4.crumbs.append(np.array([1.5, 0.0, -2.0]))
    p = pc.copy()
    waypoint = np.array([10.0, 0.0, -2.0])
    out = run(sh4, t + 0.01, p, np.zeros(3), 0.0, [1.0, 0.0, 0.0], waypoint)
    assert out.takeover and sh4.mode == BACKOFF and out.velocity[0] < 0.0 and sh4.counts["backoffs"] == 1
    for _ in range(400):                                                                 # follow the setpoints
        t += 0.01
        sh4.last_frame = t
        out = run(sh4, t, p, np.zeros(3), 0.0, [1.0, 0.0, 0.0], waypoint)
        if sh4.mode != BACKOFF:
            break
        p = p + out.velocity * 0.01
    assert sh4.mode == NUDGE and p[0] < 1.4, (sh4.mode, p)
    out = run(sh4, t + 0.01, p, np.zeros(3), 0.0, [1.0, 0.0, 0.0], waypoint)
    assert out.takeover and out.velocity[0] >= 0.0
    # stall: no progress toward the waypoint for 5 s -> nudge, turning to it first
    sh5 = SafetyShield(camera=cam)
    sh5.reset(0.0, p0)
    behind = np.array([-10.0, 0.0, -2.0])
    for k in range(0, 520):
        sh5.last_frame = 0.01 * k
        out = run(sh5, 0.01 * k, p0, np.zeros(3), 0.0, [0.0, 0.0, 0.0], behind)
    assert sh5.counts["stalls"] == 1 and sh5.mode == NUDGE and out.takeover and abs(out.yaw_rate) > 0.5
    assert np.allclose(out.velocity, 0.0)                                                # turning before moving
    # the backoff goes through the barrier: a remembered point on the breadcrumbs behind ends it
    sh7 = SafetyShield(camera=cam)
    sh7.reset(0.0, np.array([-1.0, 0.0, -2.0]))
    for k in range(5):
        run(sh7, 0.05 * k, np.array([-1.0 + 0.25 * k, 0.0, -2.0]), np.zeros(3), 0.0, [0.0, 0.0, 0.0])
    sh7.last_frame = 1.0
    sh7.local = np.array([[0.3, 0.0, -2.0], [-0.57, 0.0, -2.0]])             # one ahead, one on the trail behind
    out = run(sh7, 0.3, np.array([0.0, 0.0, -2.0]), np.zeros(3), 0.0, [1.0, 0.0, 0.0], np.array([10.0, 0.0, -2.0]))
    assert sh7.counts["backoffs"] == 1 and sh7.mode == NUDGE, sh7.mode
    # a slide around a trunk that leaves the camera's view is capped like any unseen motion
    sh8 = SafetyShield(camera=cam)
    sh8.reset(0.0, p0)
    sh8.last_frame = 1.0
    sh8.local = np.array([[0.8, 0.0, -2.0]])                                   # 0.8 m straight ahead
    out = run(sh8, 0.5, p0, np.zeros(3), 0.0, [1.5, 1.5, 0.0])                # 45 deg: in the barrier's way
    assert "barrier" in sh8.reasons and "blind" in sh8.reasons, sh8.reasons
    assert float(np.linalg.norm(out.velocity[:2])) <= sh8.cfg.blind_speed + 1e-6
    # a frame of a forest costs little
    sh6 = SafetyShield(camera=cam)
    sh6.reset(0.0, p0)
    for k in range(30):
        sh6.on_frame(0.06 * k, depth, level, p0)
    assert sh6.summary()["frame_ms_p95"] < 50.0 and sh6.summary()["memory_points"] > 0
    rec = sh4.summary()
    assert rec["backoffs"] == 1 and rec["nudges"] == 1 and rec["seconds"][BACKOFF] > 0
    print("shield.py self-test passed")


if __name__ == "__main__":
    _self_test()
