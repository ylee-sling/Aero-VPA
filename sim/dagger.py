# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/dagger.py -- Stage (iii) of Sec. 3.8: dataset aggregation (DAgger) with the privileged expert as labeller and gate.

Used by ``sim/fly_policy.py --dagger_dir DIR``: the policy flies the mission, and every decision frame is recorded
with the expert's command for the state the vehicle is actually in, in the demonstration schema of sim/expert.py
(one file per world, DIR/<world>.hdf5). cache_latents.py and train_stage2.py read the aggregated data unchanged.

Labels. The expert's reference for a mission is its minimum-snap plan through the waypoints at a speed limit drawn
per mission from --speed_limits (as in the demonstrations), with the yaw reference that faces the direction of
travel. The label of a frame is the expert's tracking law along that reference from the point nearest to the
vehicle (progress tau*, searched from the previous one, 0.5 s back to 2 s ahead):

    knot k = v_ref(tau* + k Delta) + k_p e_p exp(-k_p k Delta),   e_p = p_ref(tau*) - p_hat,

and the yaw rate likewise, saturated as the expert's commands (``ExpertController.chunk`` with its clock at tau*).
On the expert's own flight tau* follows the clock and the label is the recorded one; off it the label steers back
onto the path, turns the camera to the direction of travel and keeps the reference's speed.

Gate. The policy flies unless (``GateConfig``) the vehicle would come close to an obstacle (privileged clearance
along its current velocity over the reaction time plus the braking time), sinks too low, leaves the corridor
around the path, faces away from the path, misses its active waypoint or stops making progress. Then the expert
takes over:
    * off the corridor, facing away or too low, when the straight line back to the path is clear and short: it
      flies the reference from tau* and the frames are still recorded;
    * near an obstacle, after a missed waypoint or a stall, or when the way back is blocked or long: it brakes
      (minimum-jerk stop), backs away from the nearest obstacle to --backoff_clearance if it stopped closer,
      re-plans from there through the remaining waypoints, turns to the new path and flies it. Braking, backing
      off, hovering, planning and turning are not recorded.
The expert's clock on its reference is governed: it runs with time but never more than lead_distance (1 m) ahead
of the path point nearest the vehicle, so after a takeover mid-path (vehicle slow, reference fast) the reference
waits for the vehicle instead of running away and pulling it across corners. While the expert flies, a clearance
below expert_clearance makes it brake and re-plan as well. After --recover_s on the path (within 0.5 m, heading
within 20 degrees, clear of obstacles) the policy gets control back.

Every decision runs the policy as deployed (its chunks are only executed while it has control), so S_t carries the
setpoint that was actually dispatched: the policy's own under its control (where a_bar and the label disagree)
and the expert's under the expert's. Recorded frames form segments, one HDF5 episode per contiguous stretch, with
the mission's waypoints, the reference speed and the shares of policy and expert frames as attributes.

    python sim/dagger.py      # self-test (labels, gate, recording; no ROS)
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert as E  # noqa: E402

DAGGER_VERSION = "1.0.0"
MODES = ("POLICY", "EXPERT", "BRAKE", "BACKOFF", "PLAN", "ALIGN")
TRIGGERS = ("obstacle", "low", "off_path", "heading", "missed_waypoint", "stall")
SOFT = ("low", "off_path", "heading")       # the expert flies back to the path (when the way is clear and short)
Command = Tuple[np.ndarray, np.ndarray, float]                  # NED velocity, NED acceleration, yaw rate
PlanFn = Callable[[np.ndarray, np.ndarray, float], "Future[E.MinSnapSpline]"]


@dataclass
class GateConfig:
    """When the expert takes over, and what is recorded (see the module docstring)."""

    clearance: float = 0.5           # predicted privileged clearance that makes the expert brake (m); the planner keeps 0.6
    lookahead_s: float = 0.5         # shortest prediction horizon (s)
    reaction_s: float = 0.3          # the prediction runs the current velocity for reaction_s + the braking time (s)
    expert_clearance: float = 0.4    # while the expert flies its path: brake and re-plan below this clearance (m)
    backoff_clearance: float = 0.8   # after braking closer than this to an obstacle, back away to it (m)
    lead_distance: float = 1.0       # the expert's reference point runs at most this far ahead of the vehicle's (m)
    altitude: float = 0.8            # height above the home ground (m)
    distance: float = 1.5            # corridor half-width around the expert path (m)
    replan_distance: float = 3.0     # farther from the path: brake and re-plan rather than fly back (m)
    heading_deg: float = 75.0        # heading error to the path's yaw reference
    stall_s: float = 5.0             # no progress along the path or towards the waypoint for this long (s)
    recover_s: float = 2.0           # expert flight on the path before the policy gets control back (s)
    label_distance: float = 2.5      # policy frames farther from the path are not recorded (m)
    correction_clearance: float = 0.4  # the straight line back to the path must keep this clearance (m)
    min_segment: int = 10            # shorter recorded segments are dropped (frames)
    brake_decel: float = 4.0         # a_max of the minimum-jerk stop (m/s^2)
    max_plan_failures: int = 3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f"GateConfig.{name} must be a non-negative number, got {value}")
        if self.replan_distance < self.distance or self.brake_decel <= 0:
            raise ValueError("require replan_distance >= distance and brake_decel > 0")


class Reference:
    """An expert reference trajectory and the vehicle's progress tau* along it."""

    def __init__(self, trajectory: E.ExpertTrajectory, speed: float, waypoints: np.ndarray, first_waypoint: int = 1,
                 dt: float = 0.02) -> None:
        self.trajectory = trajectory
        self.speed = float(speed)
        self.limits = E.CommandLimits(speed_h=self.speed)
        n = int(math.ceil(trajectory.duration / dt)) + 1
        self.ts = np.minimum(np.arange(n) * dt, trajectory.duration)
        self.ps = trajectory.spline.evaluate(self.ts)
        self.tau = 0.0
        # reference time at which the path passes each waypoint it covers (1-based mission indices)
        self.waypoint_times: Dict[int, float] = {
            first_waypoint + i: float(self.ts[int(np.argmin(np.linalg.norm(self.ps - w, axis=1)))])
            for i, w in enumerate(np.asarray(waypoints, dtype=np.float64).reshape(-1, 3))}

    @property
    def duration(self) -> float:
        return self.trajectory.duration

    def nearest(self, position: np.ndarray, back: float = 0.5, ahead: float = 2.0) -> float:
        """tau* of ``position``: the closest sample within [tau - back, tau + ahead] of the current progress."""
        idx = np.nonzero((self.ts >= self.tau - back) & (self.ts <= self.tau + ahead))[0]
        if idx.size == 0:
            idx = np.array([len(self.ts) - 1])
        d = np.linalg.norm(self.ps[idx] - np.asarray(position, dtype=np.float64), axis=1)
        return float(self.ts[idx[int(np.argmin(d))]])

    def advance(self, position: np.ndarray) -> float:
        self.tau = self.nearest(position)
        return self.tau

    def lead_time(self, tau: float, distance: float, horizon: int = 500) -> float:
        """The first reference time after ``tau`` at which the path lies ``distance`` (straight line) from p_ref(tau),
        searched at most ``horizon`` samples ahead (10 s); the end of that window if the path stays closer."""
        i = min(int(np.searchsorted(self.ts, float(tau) - 1e-9)), len(self.ts) - 1)
        stop = min(len(self.ts), i + horizon)
        j = np.nonzero(np.linalg.norm(self.ps[i:stop] - self.ps[i], axis=1) > distance)[0]
        return float(self.ts[i + int(j[0])]) if j.size else float(self.ts[stop - 1])

    def position(self, tau: float) -> np.ndarray:
        return np.asarray(self.trajectory.spline.evaluate(float(tau)), dtype=np.float64)

    def yaw(self, tau: float) -> float:
        return float(self.trajectory.reference(float(tau))[3])

    def controller(self, now: float, tau: float) -> E.ExpertController:
        """The expert's tracking law whose clock reads ``tau`` at time ``now``."""
        return E.ExpertController(self.trajectory, float(now) - float(tau), self.limits)

    def label(self, stamp: float, tau: float, position: np.ndarray, yaw: float) -> np.ndarray:
        """The expert's chunk ``[H, 4]`` (heading frame of ``yaw``) from progress ``tau``."""
        return self.controller(stamp, tau).chunk(stamp, position, yaw)


def braking(v0: np.ndarray, elapsed: float, decel: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Minimum-jerk stop from ``v0`` (Eq. 26): (velocity, acceleration, stop time T_b) at ``elapsed`` seconds."""
    v0 = np.asarray(v0, dtype=np.float64)
    t_b = 1.5 * float(np.linalg.norm(v0)) / decel
    if t_b <= 0.0 or elapsed >= t_b:
        return np.zeros(3), np.zeros(3), t_b
    s = max(0.0, elapsed) / t_b
    return v0 * (1.0 - 3.0 * s * s + 2.0 * s ** 3), v0 * (-6.0 * s + 6.0 * s * s) / t_b, t_b


def completed(value: Any) -> Future:
    """A finished ``Future`` holding ``value`` (or raising it, for an exception): planning without a worker."""
    f: Future = Future()
    if isinstance(value, BaseException):
        f.set_exception(value)
    else:
        f.set_result(value)
    return f


def plan_now(planner: E.ExpertPlanner, start: np.ndarray, waypoints: np.ndarray, speed: float) -> Future:
    """The expert spline from ``start`` (at rest) through ``waypoints``, planned in this process, as a finished
    ``Future`` (dry runs and tests; SITL plans in a worker process)."""
    try:
        return completed(planner.plan(start, waypoints, speed, 0.0).spline)
    except E.PlanningError as exc:
        return completed(exc)


class InlineWriter:
    """``collect_expert_dataset.WriterProcess``'s interface (send / check / close) in this process (dry runs, tests)."""

    def __init__(self, path: str, attrs: Dict[str, Any], append: bool = True, compression: Optional[str] = "gzip") -> None:
        self.writer = E.DatasetWriter(path, attrs, mode="a" if append else "w", compression=compression)
        self.episode: Optional[E.EpisodeWriter] = None
        self.name = ""
        self.finalized: List[Tuple[str, int]] = []

    def send(self, cmd: str, payload: Any = None) -> None:
        if cmd == "begin":
            self.name, self.episode = self.writer.begin_episode()
        elif cmd == "append":
            self.episode.append(payload)
        elif cmd == "finalize":
            self.finalized.append((self.name, self.episode.finalize(payload)))
            self.writer.flush()
            self.episode = None
        elif cmd == "discard":
            self.writer.discard(self.name)
            self.writer.flush()
            self.episode = None
        else:
            raise ValueError(f"unknown writer command {cmd!r}")

    def check(self) -> None:
        pass

    def close(self, timeout: float = 0.0) -> None:
        del timeout
        self.writer.close()


def file_attributes(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Root attributes of a DAgger file (on top of the schema's): provenance of the labels and the gate."""
    return {"source": "dagger", "dagger_version": DAGGER_VERSION,
            "label_thresholds": json.dumps(asdict(E.LabelThresholds())), "label_command": "first_knot",
            **{k: (json.dumps(v, sort_keys=True, default=str) if isinstance(v, (dict, list, tuple)) else v)
               for k, v in extra.items()}}


class DaggerSupervisor:
    """Labels, gate and recording of one DAgger mission.

    Args:
        world: privileged map (gate checks and the clearance of corrections).
        planner: the expert planner (hover and alignment trajectories).
        waypoints: ``[M, 3]`` mission waypoints (NED).
        speed: v_h of the expert references.
        trajectory: the expert's plan of the mission from its start, at rest (tau* = 0 at ``start_time``).
        start_time: mission start (s, simulation clock).
        plan: ``plan(start [3], waypoints [K, 3], speed) -> Future`` of a ``MinSnapSpline`` (re-planning).
        writer: object with ``send(cmd, payload)`` (``WriterProcess`` or ``InlineWriter``), or None to label only.
        episode_attrs: attributes stored with every recorded segment (world, seed, mission, policy, ...).
        gate: ``GateConfig``.
    """

    def __init__(self, world: E.PrivilegedMap, planner: E.ExpertPlanner, waypoints: np.ndarray, speed: float,
                 trajectory: E.ExpertTrajectory, start_time: float, plan: PlanFn, writer: Any = None,
                 episode_attrs: Optional[Dict[str, Any]] = None, gate: Optional[GateConfig] = None) -> None:
        self.world, self.planner = world, planner
        self.waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
        self.speed = float(speed)
        self.plan, self.writer = plan, writer
        self.episode_attrs = dict(episode_attrs or {})
        self.gate = gate or GateConfig()
        self.thresholds = E.LabelThresholds()
        self.reference = Reference(trajectory, self.speed, self.waypoints, 1)
        self.next_reference: Optional[Reference] = None
        self.mode, self.mode_since = "POLICY", float(start_time)
        self.tau_cmd = 0.0                                   # the expert's governed clock on its reference
        self.hold: Optional[E.ExpertController] = None
        self.backoff_z = 0.0
        self.expert_brakes = 0
        self.backoffs = 0
        self.brake_v0, self.brake_t0 = np.zeros(3), float(start_time)
        self.plan_future: Optional[Future] = None
        self.plan_start = np.zeros(3)
        self.plan_failures = 0
        self.failed: Optional[str] = None
        self.m = 1
        self.progress_time, self.progress_tau, self.progress_dist = float(start_time), 0.0, math.inf
        self.last_now: Optional[float] = None
        self.segment: Optional[Dict[str, Any]] = None
        self.takeovers: Counter = Counter()
        self.time_in: Counter = Counter()
        self.frames = {"policy": 0, "expert": 0, "skipped": 0, "discarded": 0}
        self.segments = 0
        self.replans = 0

    # ---------------- helpers ----------------
    def _set_mode(self, mode: str, now: float) -> None:
        self.mode, self.mode_since = mode, float(now)

    def _clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        if not len(self.world):
            return True
        return self.world.segment_clearance(a, b, step=0.1) >= self.gate.correction_clearance

    def _clearance(self, p: np.ndarray) -> float:
        return float(self.world.distance(p)) if len(self.world) else math.inf

    def _track_progress(self, now: float, position: np.ndarray) -> None:
        tau = self.reference.tau
        idx = min(max(self.m, 1), len(self.waypoints)) - 1
        dist = float(np.linalg.norm(position - self.waypoints[idx]))
        if tau >= self.progress_tau + 0.25 or dist <= self.progress_dist - 0.25:
            self.progress_time, self.progress_tau, self.progress_dist = float(now), tau, dist

    def predicted_clearance(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Smallest privileged clearance along the current velocity over the reaction time plus the braking time."""
        if not len(self.world):
            return math.inf
        g = self.gate
        horizon = max(g.lookahead_s, g.reaction_s + 1.5 * float(np.linalg.norm(velocity)) / g.brake_decel)
        ts = np.linspace(0.0, horizon, 6)
        return float(np.min(self.world.distance(position[None] + ts[:, None] * velocity[None])))

    def away(self, position: np.ndarray, h: float = 0.05) -> np.ndarray:
        """Unit horizontal direction away from the nearest obstacle (central differences of the distance field)."""
        if not len(self.world):
            return np.zeros(3)
        steps = np.array([[h, 0.0, 0.0], [-h, 0.0, 0.0], [0.0, h, 0.0], [0.0, -h, 0.0]])
        d = self.world.distance(position[None] + steps)
        grad = np.array([d[0] - d[1], d[2] - d[3], 0.0]) / (2.0 * h)
        n = float(np.linalg.norm(grad))
        return grad / n if n > 1e-6 else np.zeros(3)

    def _start_brake(self, now: float, velocity: np.ndarray, reason: str) -> None:
        self.close_segment(f"takeover: {reason}")
        self.brake_v0, self.brake_t0 = np.asarray(velocity, dtype=np.float64).copy(), float(now)
        self._set_mode("BRAKE", now)

    def _start_plan(self, now: float, position: np.ndarray, yaw: float) -> None:
        self.hold = E.ExpertController(self.planner.hold(position, yaw), now, E.CommandLimits(speed_h=1.0))
        self.plan_start = position.copy()
        remaining = self.waypoints[min(max(self.m, 1), len(self.waypoints)) - 1:]
        self.plan_future = self.plan(position.copy(), remaining, self.speed)
        self.replans += 1
        self._set_mode("PLAN", now)

    def trigger(self, now: float, position: np.ndarray, velocity: np.ndarray, yaw: float) -> Optional[str]:
        """Why the expert must take over now (``TRIGGERS``), or None."""
        g, ref = self.gate, self.reference
        if self.predicted_clearance(position, velocity) < g.clearance:
            return "obstacle"
        if -position[2] < g.altitude:
            return "low"
        if float(np.linalg.norm(ref.position(ref.tau) - position)) > g.distance:
            return "off_path"
        if abs(math.degrees(float(E.wrap_angle(ref.yaw(ref.tau) - yaw)))) > g.heading_deg:
            return "heading"
        t_wp = ref.waypoint_times.get(self.m)
        if t_wp is not None and ref.tau > t_wp + 1.0:
            return "missed_waypoint"
        if float(now) - self.progress_time > g.stall_s:
            return "stall"
        return None

    def _on_track(self, position: np.ndarray, yaw: float) -> bool:
        ref = self.reference
        e = float(np.linalg.norm(ref.position(ref.tau) - position))
        heading = abs(math.degrees(float(E.wrap_angle(ref.yaw(ref.tau) - yaw))))
        return e < 0.5 and heading < 20.0 and self._clearance(position) > self.gate.clearance + 0.3

    # ---------------- executor side (every control tick) ----------------
    def control(self, now: float, position: np.ndarray, velocity: np.ndarray, yaw: float) -> Optional[Command]:
        """The expert's setpoint while it has control, or None while the policy flies."""
        now, p, v = float(now), np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
        dt = 0.0 if self.last_now is None else max(0.0, now - self.last_now)
        self.time_in[self.mode] += dt
        self.last_now = now
        g = self.gate
        if self.mode in ("POLICY", "EXPERT"):
            self.reference.advance(p)
            self._track_progress(now, p)
        if self.mode == "POLICY":
            reason = self.trigger(now, p, v, yaw)
            if reason is None:
                return None
            self.takeovers[reason] += 1
            ref = self.reference
            p_ref = ref.position(ref.tau)
            if reason in SOFT and float(np.linalg.norm(p_ref - p)) <= g.replan_distance and self._clear(p, p_ref):
                self.tau_cmd = ref.tau
                self._set_mode("EXPERT", now)
            else:
                self._start_brake(now, v, reason)
        if self.mode == "EXPERT":
            ref = self.reference
            self.tau_cmd = max(ref.tau, min(self.tau_cmd + dt, ref.lead_time(ref.tau, g.lead_distance)))
            if self._clearance(p) < g.expert_clearance:
                self.expert_brakes += 1
                self._start_brake(now, v, "obstacle while the expert flies")
            elif now - self.mode_since >= g.recover_s and self._on_track(p, yaw):
                self.progress_time = now
                self._set_mode("POLICY", now)
                return None
            else:
                return ref.controller(now, self.tau_cmd).command(now, p, yaw)
        if self.mode == "BRAKE":
            vel, acc, t_b = braking(self.brake_v0, now - self.brake_t0, g.brake_decel)
            if now - self.brake_t0 < t_b or float(np.linalg.norm(v)) > 0.3:
                return vel, acc, 0.0
            if self._clearance(p) < g.backoff_clearance:
                self.backoff_z, self.backoffs = float(p[2]), self.backoffs + 1
                self._set_mode("BACKOFF", now)
            else:
                self._start_plan(now, p, yaw)
        if self.mode == "BACKOFF":
            if self._clearance(p) >= g.backoff_clearance or now - self.mode_since > 4.0:
                self._start_plan(now, p, yaw)
            else:
                return 0.5 * self.away(p) + np.array([0.0, 0.0, self.backoff_z - p[2]]), np.zeros(3), 0.0
        if self.mode == "PLAN":
            if self.plan_future is not None and self.plan_future.done():
                future, self.plan_future = self.plan_future, None
                try:
                    spline = future.result()
                except E.PlanningError:
                    self.plan_failures += 1
                    if self.plan_failures >= self.gate.max_plan_failures:
                        self.failed = "dagger_no_plan"
                    else:
                        remaining = self.waypoints[min(max(self.m, 1), len(self.waypoints)) - 1:]
                        self.plan_future = self.plan(self.plan_start.copy(), remaining, self.speed)
                    return self.hold.command(now, p, yaw)
                trajectory = E.ExpertTrajectory(spline, yaw)
                heading = trajectory.initial_heading()
                first = min(max(self.m, 1), len(self.waypoints))
                self.next_reference = Reference(trajectory, self.speed, self.waypoints[first - 1:], first)
                self.hold = E.ExpertController(self.planner.hold(self.plan_start, yaw,
                                                                 yaw_target=heading if heading is not None else yaw),
                                               now, E.CommandLimits(speed_h=1.0))
                self._set_mode("ALIGN", now)
            return self.hold.command(now, p, yaw)
        if self.mode == "ALIGN":
            target = float(self.hold.trajectory.reference(np.inf)[3])
            if (abs(float(E.wrap_angle(target - yaw))) < 0.08 and now - self.mode_since > 0.5) or now - self.mode_since > 8.0:
                self.reference, self.next_reference = self.next_reference, None
                self.reference.tau, self.tau_cmd = 0.0, 0.0
                self.progress_time, self.progress_tau, self.progress_dist = now, 0.0, math.inf
                self._set_mode("EXPERT", now)
                return self.reference.controller(now, 0.0).command(now, p, yaw)
            return self.hold.command(now, p, yaw)
        return None

    # ---------------- decision side (every camera frame) ----------------
    def on_frame(self, stamp: float, image16: np.ndarray, state: np.ndarray, goal: np.ndarray, waypoint_index: int,
                 task_complete: bool, position: np.ndarray, yaw: float, odometry_age: float = 0.0) -> Optional[Dict[str, Any]]:
        """Label and record the frame of one decision (``state`` and ``goal`` as the policy saw them).

        Returns what was recorded (controller, tau, e_p, first label knot, primitive) or None when the frame is not.
        """
        self.m = int(waypoint_index)
        p = np.asarray(position, dtype=np.float64)
        ref = self.reference
        if self.mode == "EXPERT":
            controller, tau = "expert", self.tau_cmd
            label = ref.label(stamp, tau, p, yaw)
        elif self.mode == "POLICY":
            controller, tau = "policy", ref.nearest(p)
            p_ref = ref.position(tau)
            if float(np.linalg.norm(p_ref - p)) > self.gate.label_distance or not self._clear(p, p_ref):
                self.frames["skipped"] += 1
                self.close_segment("left the corridor")
                return None
            label = ref.label(stamp, tau, p, yaw)
        else:
            self.frames["skipped"] += 1
            return None
        distance_to_final = float(np.linalg.norm(p - self.waypoints[-1]))
        primitive = int(E.primitive_labels(label[:1], np.array([distance_to_final]), self.thresholds)[0])
        p_ref = ref.position(tau)
        sample = E.FrameSample(
            timestamp=float(stamp), image=image16, state=np.asarray(state, dtype=np.float32),
            goal=np.asarray(goal, dtype=np.float32), action=np.asarray(label, dtype=np.float32), primitive=primitive,
            position=p.astype(np.float32), yaw=float(yaw), waypoint_index=int(waypoint_index),
            task_complete=bool(task_complete), reference_position=p_ref.astype(np.float32),
            odometry_age=float(odometry_age))
        if self.writer is not None:
            if self.segment is None:
                self.writer.send("begin")
                self.segment = {"frames": 0, "policy": 0, "expert": 0, "reference_duration": ref.duration}
            self.writer.send("append", sample)
        if self.segment is None:
            self.segment = {"frames": 0, "policy": 0, "expert": 0, "reference_duration": ref.duration}
        self.segment["frames"] += 1
        self.segment[controller] += 1
        return {"controller": controller, "tau": tau, "e_path": float(np.linalg.norm(p_ref - p)), "label0": label[0],
                "primitive": primitive}

    def close_segment(self, reason: str) -> None:
        """End the open recorded segment: an HDF5 episode, or dropped when shorter than ``min_segment``."""
        seg, self.segment = self.segment, None
        if seg is None:
            return
        if seg["frames"] < self.gate.min_segment:
            if self.writer is not None:
                self.writer.send("discard")
            self.frames["discarded"] += seg["frames"]
            return
        self.frames["policy"] += seg["policy"]
        self.frames["expert"] += seg["expert"]
        self.segments += 1
        if self.writer is not None:
            attrs = {**self.episode_attrs, "waypoints_ned": self.waypoints, "speed_limit": self.speed,
                     "success": True, "termination": reason, "instruction": self.episode_attrs.get("instruction", ""),
                     "expert_duration_s": float(seg["reference_duration"]), "dagger_segment": self.segments,
                     "policy_frames": int(seg["policy"]), "expert_frames": int(seg["expert"])}
            self.writer.send("finalize", attrs)

    def finish(self, termination: str) -> Dict[str, Any]:
        """Close the mission (the open segment ends with ``termination``) and return its DAgger statistics."""
        self.close_segment(termination)
        if self.plan_future is not None:
            self.plan_future.cancel()
            self.plan_future = None
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        total = sum(self.time_in.values())
        return {"dagger_frames_policy": self.frames["policy"], "dagger_frames_expert": self.frames["expert"],
                "dagger_frames_skipped": self.frames["skipped"], "dagger_frames_discarded": self.frames["discarded"],
                "dagger_segments": self.segments, "dagger_replans": self.replans,
                "dagger_expert_brakes": self.expert_brakes, "dagger_backoffs": self.backoffs,
                "dagger_takeovers": {k: int(self.takeovers.get(k, 0)) for k in TRIGGERS},
                "dagger_time_s": {m: round(float(self.time_in.get(m, 0.0)), 2) for m in MODES},
                "dagger_policy_share": float(self.time_in.get("POLICY", 0.0) / total) if total > 0 else None,
                "dagger_speed_limit": self.speed, "dagger_failed": self.failed}


# ---------------------------------------------------------------------------------------------
# Self-test: a kinematic vehicle that a scripted "policy" pushes off the expert path
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    import h5py

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)

    # ---- braking profile: from v0 to rest in T_b = 3|v0| / (2 a) with peak deceleration a at T_b / 2
    v, a, t_b = braking(np.array([3.0, 0.0, 0.0]), 0.0, 4.0)
    assert np.allclose(v, [3.0, 0, 0]) and np.allclose(a, 0.0) and math.isclose(t_b, 1.125)
    v, a, _ = braking(np.array([3.0, 0.0, 0.0]), t_b / 2, 4.0)
    assert np.allclose(v, [1.5, 0, 0]) and math.isclose(a[0], -4.0)
    assert np.allclose(braking(np.array([3.0, 0.0, 0.0]), 2.0, 4.0)[0], 0.0)

    # ---- reference: on the expert's own flight the label is the recorded chunk; off it, it steers back
    world = E.PrivilegedMap([E.Obstacle("cylinder", (10.0, 3.0, -3.0), radius=0.3, half_height=3.0)])
    planner = E.ExpertPlanner(world)
    start = np.array([0.0, 0.0, -2.5])
    waypoints = np.array([[20.0, 0.0, -2.5], [20.0, 15.0, -3.0]])
    trajectory = planner.plan(start, waypoints, 3.0, 0.0)
    ref = Reference(trajectory, 3.0, waypoints)
    expert = E.ExpertController(trajectory, 0.0, E.CommandLimits(speed_h=3.0))
    for t in (0.5, 3.0, 6.0):
        p = trajectory.reference(t)[0]
        yaw = float(trajectory.reference(t)[3])
        ref.tau = t - 0.2
        tau = ref.advance(p)
        assert abs(tau - t) < 0.011, (tau, t)
        assert np.allclose(ref.label(100.0 + t, tau, p, yaw), expert.chunk(t, p, yaw), atol=0.05)
    assert set(ref.waypoint_times) == {1, 2} and ref.waypoint_times[1] < ref.waypoint_times[2] <= ref.duration + 1e-9
    ref.tau = 3.0
    p_on, yaw_on = trajectory.reference(3.0)[0], float(trajectory.reference(3.0)[3])
    off = p_on + np.array([-math.sin(yaw_on), math.cos(yaw_on), 0.0])            # 1 m to the right of the path
    lab = ref.label(0.0, ref.nearest(off), off, yaw_on)
    assert lab[0, 1] < -1.0 and lab[0, 0] > 0.5    # the first knot points back left (heading frame) and onwards

    # ---- gate: a scripted policy drifts sideways; the expert takes over, flies back, and hands back
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "w.hdf5")
        writer = InlineWriter(path, file_attributes({"policy_checkpoint": "test"}), append=True, compression=None)

        def plan(s: np.ndarray, wps: np.ndarray, speed: float) -> Future:
            return plan_now(planner, s, wps, speed)

        sup = DaggerSupervisor(world, planner, waypoints, 3.0, E.ExpertTrajectory(trajectory.spline, 0.0), 0.0, plan,
                               writer, {"world": "test", "seed": 1, "mission": 0}, GateConfig(recover_s=1.0))
        vehicle = E.KinematicQuadrotor(start, 0.0)
        tracker = E.WaypointTracker(waypoints)
        image = np.zeros((*E.IMAGE_SIZE, 4), dtype=np.uint16)
        t, dt, modes, recorded = 0.0, 0.01, [], Counter()
        policy_cmd = np.array([2.0, 1.2, 0.0])                       # drifts right of the path
        while t < 60.0 and not tracker.task_complete:
            cmd = sup.control(t, vehicle.p, vehicle.v, vehicle.yaw)
            modes.append(sup.mode)
            if cmd is None:
                vel, acc, yr = E.heading_to_world(policy_cmd, vehicle.yaw), np.zeros(3), 0.0
            else:
                vel, acc, yr = cmd
            if round(t / dt) % 2 == 0:                               # 50 Hz frames
                goal = tracker.step(vehicle.p, vehicle.yaw)
                state = E.proprioception(vehicle.v, vehicle.quaternion(), np.zeros(3), np.array([*vel, yr]), vehicle.yaw)
                info = sup.on_frame(t, image, state, goal, tracker.m, tracker.task_complete, vehicle.p, vehicle.yaw)
                if info is not None:
                    recorded[info["controller"]] += 1
            vehicle.step(dt, vel, acc, yr)
            t = round(t + dt, 2)
        stats = sup.finish("complete" if tracker.task_complete else "timeout")
        writer.close()
        assert tracker.task_complete, "the gated mission did not finish"
        assert "EXPERT" in modes and "POLICY" in modes and stats["dagger_takeovers"]["off_path"] >= 1
        assert recorded["policy"] > 0 and recorded["expert"] > 0
        assert stats["dagger_frames_policy"] + stats["dagger_frames_expert"] + stats["dagger_frames_discarded"] == sum(recorded.values())
        assert 0.0 < stats["dagger_policy_share"] < 1.0
        # the file is a demonstration file: it validates and loads for stage (ii)
        import inspect_hdf5
        from config import VPAConfig
        report = inspect_hdf5.validate_file(path, VPAConfig())
        assert not report["errors"], report["errors"]
        with h5py.File(path, "r") as f:
            assert f.attrs["source"] == "dagger"
            eps = [f[k] for k in f if k.startswith("episode_")]
            assert len(eps) == stats["dagger_segments"] >= 1
            assert sum(int(ep.attrs["policy_frames"]) + int(ep.attrs["expert_frames"]) for ep in eps) == \
                stats["dagger_frames_policy"] + stats["dagger_frames_expert"]
            assert all(bool(ep.attrs["success"]) and bool(ep.attrs["complete"]) for ep in eps)
            labels = np.concatenate([ep["actions"][...] for ep in eps])
            assert np.all(np.isfinite(labels)) and np.abs(labels[:, :, :3]).max() <= 3.0 * math.sqrt(2) + 2.0
        from dataset import AeroVPADataset
        assert len(AeroVPADataset(path, 2)) == stats["dagger_frames_policy"] + stats["dagger_frames_expert"]

    # ---- hard takeover: far off the path -> brake, re-plan from the stop (planning takes 0.5 s here), align, fly
    pending: List[Tuple[Future, np.ndarray, np.ndarray, float]] = []

    def slow_plan(s: np.ndarray, wps: np.ndarray, speed: float) -> Future:
        pending.append((Future(), s, wps, speed))
        return pending[-1][0]

    sup = DaggerSupervisor(world, planner, waypoints, 3.0, E.ExpertTrajectory(trajectory.spline, 0.0), 0.0, slow_plan,
                           None, {}, GateConfig())
    p_far = np.array([5.0, -4.0, -2.5])
    assert sup.trigger(1.0, p_far, np.zeros(3), 0.0) == "off_path"
    seen, t, plan_ticks = [], 1.0, 0
    vehicle = E.KinematicQuadrotor(p_far, 0.0)
    vehicle.v = np.array([1.0, -1.0, 0.0])
    for _ in range(3000):
        if sup.mode == "PLAN":
            plan_ticks += 1
            if plan_ticks == 50:
                future, s0, wps, speed = pending[-1]
                assert np.linalg.norm(s0 - vehicle.p) < 0.2 and np.allclose(wps, waypoints)   # from the stop
                future.set_result(planner.plan(s0, wps, speed, 0.0).spline)
        cmd = sup.control(t, vehicle.p, vehicle.v, vehicle.yaw)
        seen.append(sup.mode)
        assert cmd is not None or sup.mode == "POLICY"
        vel, acc, yr = cmd if cmd is not None else (np.zeros(3), np.zeros(3), 0.0)
        vehicle.step(0.01, vel, acc, yr)
        t += 0.01
        if sup.mode == "POLICY" and "EXPERT" in seen:
            break
    order = [m for i, m in enumerate(seen) if i == 0 or seen[i - 1] != m]
    assert order[:5] == ["BRAKE", "PLAN", "ALIGN", "EXPERT", "POLICY"], order
    assert sup.replans == 1 and sup.reference.waypoint_times and min(sup.reference.waypoint_times) == 1
    assert sup.summary()["dagger_takeovers"]["off_path"] == 1

    # ---- governed clock: an expert takeover with the vehicle held still never runs the reference away
    fast = planner.plan(start, waypoints, 7.0, 0.0)
    sup = DaggerSupervisor(world, planner, waypoints, 7.0, fast, 0.0, plan, None, {}, GateConfig())
    v_mid = fast.reference(2.0)[1]
    side = np.array([-v_mid[1], v_mid[0], 0.0]) / float(np.linalg.norm(v_mid[:2]))
    p_mid = fast.reference(2.0)[0] + 2.2 * side                           # 2.2 m beside the path: off the corridor
    sup.reference.tau = 2.0
    cmds = [sup.control(t, p_mid, np.zeros(3), float(fast.reference(2.0)[3])) for t in np.arange(0.0, 5.0, 0.01)]
    assert sup.mode == "EXPERT" and sup.takeovers["off_path"] == 1
    tau_star = sup.reference.nearest(p_mid)
    assert tau_star <= sup.tau_cmd <= sup.reference.lead_time(tau_star, 1.0) + 1e-9           # at most 1 m ahead
    lead = float(np.linalg.norm(sup.reference.position(sup.tau_cmd) - sup.reference.position(tau_star)))
    assert lead <= 1.0 + 0.2, lead
    assert max(float(np.linalg.norm(c[0])) for c in cmds) <= 7.0 * math.sqrt(2) + 2.0           # saturated, not runaway

    # ---- head-on at a trunk: the prediction brakes in time, backs off if close, re-plans; no collision
    trunk = E.PrivilegedMap([E.Obstacle("cylinder", (6.0, 0.0, -3.0), radius=0.3, half_height=3.0)])
    tplanner = E.ExpertPlanner(trunk)
    goal = np.array([[12.0, 0.0, -2.5]])
    sup = DaggerSupervisor(trunk, tplanner, goal, 3.0, tplanner.plan(start, goal, 3.0, 0.0), 0.0,
                           lambda s, w, v: plan_now(tplanner, s, w, v), None, {}, GateConfig(distance=10.0, replan_distance=10.0))
    vehicle = E.KinematicQuadrotor(start, 0.0)
    seen, closest, t = [], math.inf, 0.0
    for _ in range(4000):
        cmd = sup.control(t, vehicle.p, vehicle.v, vehicle.yaw)
        seen.append(sup.mode)
        vel, acc, yr = cmd if cmd is not None else (np.array([2.0, 0.0, 0.0]), np.zeros(3), 0.0)   # straight at it
        vehicle.step(0.01, vel, acc, yr)
        closest = min(closest, float(trunk.distance(vehicle.p)))
        t += 0.01
        if sup.mode == "EXPERT" and "BRAKE" in seen:
            break
    order = [m for i, m in enumerate(seen) if (i == 0 or seen[i - 1] != m) and m != "PLAN"]   # planning is instant
    assert sup.takeovers["obstacle"] == 1 and order == ["POLICY", "BRAKE", "ALIGN", "EXPERT"], order
    assert sup.replans == 1 and closest > 1.0, closest                    # braked well before the trunk
    # stopped close to the trunk: back away before planning (the planner needs 0.6 m at the start)
    sup = DaggerSupervisor(trunk, tplanner, goal, 3.0, tplanner.plan(start, goal, 3.0, 0.0), 0.0,
                           lambda s, w, v: plan_now(tplanner, s, w, v), None, {}, GateConfig())
    p_close = np.array([5.3, 0.0, -2.5])                                  # 0.4 m from the trunk surface
    sup._start_brake(0.0, np.zeros(3), "test")
    t, vehicle = 0.0, E.KinematicQuadrotor(p_close, 0.0)
    for _ in range(600):
        cmd = sup.control(t, vehicle.p, vehicle.v, vehicle.yaw)
        if sup.mode == "BACKOFF":
            assert cmd is not None and float(cmd[0] @ np.array([-1.0, 0.0, 0.0])) > 0.4      # away from the trunk
        vehicle.step(0.01, *(cmd if cmd is not None else (np.zeros(3), np.zeros(3), 0.0)))
        t += 0.01
        if sup.mode in ("PLAN", "ALIGN", "EXPERT"):
            break
    assert sup.backoffs == 1 and float(trunk.distance(sup.plan_start)) >= 0.8 - 1e-6 and sup.replans == 1

    # ---- the expert itself too close (tracking error near a trunk): brake and re-plan
    sup = DaggerSupervisor(trunk, tplanner, goal, 3.0, tplanner.plan(start, goal, 3.0, 0.0), 0.0,
                           lambda s, w, v: plan_now(tplanner, s, w, v), None, {}, GateConfig())
    sup._set_mode("EXPERT", 0.0)
    assert sup.control(0.01, np.array([5.35, 0.0, -2.5]), np.array([1.0, 0.0, 0.0]), 0.0) is not None
    assert sup.mode == "BRAKE" and sup.expert_brakes == 1 and sup.summary()["dagger_expert_brakes"] == 1
    print("dagger.py self-test passed")


if __name__ == "__main__":
    _self_test()
