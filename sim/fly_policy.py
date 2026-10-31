# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/fly_policy.py -- Fly a calibrated Aero-VPA policy closed-loop in PX4 SITL and score every mission.

The policy runs Algorithm 1 as deployed (``pipeline.VPAInferencePipeline``): one decision per synchronised RGB-D
frame (Eq. 6 -> selector -> P_omega and the uncertainty gate -> CFM chunk -> braking envelope with the depth
range d_t of Sec. 3.6, ``clearance.DepthRange``) and the 100 Hz executor that turns the newest chunk into PX4
velocity setpoints with acceleration feed-forward. The vehicle is the one of the demonstrations (``x500_vpa``),
started by ``run_headless_sitl.sh``:

    sim/run_headless_sitl.sh --px4-dir ~/PX4-Autopilot --ros-ws ~/px4_ros_ws --world sim/worlds/val/W.sdf -- \\
        python3 sim/fly_policy.py checkpoints/policy_cal_ep.pt --episodes 10 --seed 1000 --results flights.jsonl
    sim/evaluate_worlds.sh checkpoints/policy_cal_ep.pt sim/worlds/val/*.sdf          # many worlds, resumable
    python3 sim/summarize_flights.py flights.jsonl                                        # success rates and more

Protocol (the same for every policy, so two checkpoints fly identical missions):
    take off to --takeoff_altitude above the spawn point (home); then for episode k = start .. start + N - 1:
    sample the mission (3-6 waypoints, 10-40 m apart, as in the demonstrations) from the generator seeded with
    (--seed, k) at the nominal home (0, 0, -takeoff_altitude); plan the expert reference at --expert_speed on
    the privileged map (only its initial heading and duration are used: the vehicle turns to that heading
    before the policy takes over, and the mission times out after timeout_factor x duration + timeout_extra);
    fly the mission with the policy; fly back home with the privileged expert; next episode.
    A mission ends with success (the final waypoint reached, Eq. 14), collision (privileged distance below
    --collision_distance), ground (below 0.3 m), out_of_bounds, timeout, camera_timeout or px4_left_offboard.
    After a collision or ground contact the vehicle may be damaged, so the run ends with exit status 3; the
    record is written first and --episode_start continues with the next mission (evaluate_worlds.sh does).

Every mission appends one JSON record to --results: outcome, duration, path length, speeds, the smallest
privileged clearance, waypoints reached, fallback decisions and onsets, time braking or holding, blind
decisions, sigma_bar, the speed scale s_t, primitive counts, decision latency (compute) and decision age
(capture to commit, the T_e2e that the envelope assumes below --latency_bound), setpoint gaps and dropped frames.

--dagger_dir DIR turns the flights into stage-(iii) data collection (sim/dagger.py): the privileged expert labels
every frame with its own command for the state the vehicle is in and takes over before trouble; the frames go to
DIR/<world>.hdf5 in the demonstration schema (sim/dagger_round.sh runs a whole round).

--trace DIR also writes every decision of every mission to DIR/<tag>_<world>_e<k>.jsonl (the first line is the
mission record): the state estimate, the active waypoint, the selected and final primitive, sigma and the gate,
the raw and committed first knot, d_t with the blind reason and the direction in the camera, v_safe and s_t,
the executor mode and the privileged clearance. sim/show_trace.py summarises a trace and draws it (SVG).

--dry_run flies the same protocol without ROS: the kinematic model and the ray-cast camera of sim/expert.py in a
synthetic forest (or --world_json). Its images are not Gazebo's, so it checks the plumbing, not the policy.

Before the first mission the policy decides --warmup times on a synthetic frame (``warm_up``): the first decision
of a process pays for lazy GPU initialisation, and a decision blocks the single-threaded ROS executor, so without
it the first mission of a run can start with no setpoint going out for as long as that decision takes.

Exit status: 0 all missions flown, 1 errors, 3 after a collision (restart the simulator, continue), 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import expert as E  # noqa: E402
from checkpoint import config_from_dict, load_checkpoint  # noqa: E402
from clearance import REASONS, DepthRange  # noqa: E402
from collect_expert_dataset import OdometryBuffer, WriterProcess, _ignore_sigint, _read_environment, decode_image  # noqa: E402
from dagger import TRIGGERS, DaggerSupervisor, GateConfig, InlineWriter, plan_now  # noqa: E402
from dagger import file_attributes as dagger_file_attributes  # noqa: E402
from pipeline import MODE_TRACK, VPAInferencePipeline  # noqa: E402

RESULT_VERSION = 1
CRASHES = ("collision", "ground")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", nargs="?", help="calibrated policy checkpoint (calibrate_sigma.py)")
    p.add_argument("--episodes", type=int, default=10, help="missions to fly")
    p.add_argument("--episode_start", type=int, default=0, help="index of the first mission")
    p.add_argument("--seed", type=int, default=0, help="mission k is sampled from the generator seeded with (seed, k)")
    p.add_argument("--results", default="flights.jsonl", help="JSON lines file; one record per mission is appended")
    p.add_argument("--tag", default=None, help="name of the policy in the results (default: checkpoint file name)")
    p.add_argument("--world_json", default=os.environ.get("AERO_VPA_WORLD_JSON"),
                   help="privileged map (world JSON sidecar); default: $AERO_VPA_WORLD_JSON")
    p.add_argument("--num_waypoints", type=int, nargs=2, default=[3, 6])
    p.add_argument("--spacing", type=float, nargs=2, default=[10.0, 40.0], help="waypoint spacing (m)")
    p.add_argument("--altitude", type=float, nargs=2, default=[1.5, 4.0], help="waypoint altitude (m)")
    p.add_argument("--takeoff_altitude", type=float, default=2.5)
    p.add_argument("--expert_speed", type=float, default=3.0, help="v_h of the reference plan (timeout, time ratio)")
    p.add_argument("--timeout_factor", type=float, default=1.5)
    p.add_argument("--timeout_extra_s", type=float, default=15.0)
    p.add_argument("--camera_timeout", type=float, default=0.5,
                   help="end the mission (camera_timeout) after this long without a decision, simulation time (s): no "
                        "frame arrived, or none had odometry bracketing its capture (the record's max_decision_gap_s)")
    p.add_argument("--return_speed", type=float, default=2.0, help="speed of the expert flight back home (m/s)")
    p.add_argument("--latency_bound", type=float, default=0.06, help="T_e2e^max of the braking envelope (s)")
    p.add_argument("--tracking_lag", type=float, default=0.15, help="T_trk of the flight controller (s)")
    p.add_argument("--execution_mode", choices=["async", "sync"], default="async")
    p.add_argument("--max_invalid_fraction", type=float, default=1.0,
                   help="blind above this fraction of pixels without depth in R_t (1: off; Gazebo has no returns "
                        "beyond its range)")
    p.add_argument("--collision_distance", type=float, default=0.35, help="privileged distance of a collision (m)")
    p.add_argument("--trace", default=None, metavar="DIR", help="write a per-decision trace of every mission to DIR")
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup", type=int, default=3,
                   help="decisions on a synthetic frame before the first mission, so that its first decision does not "
                        "freeze the control loop with lazy GPU initialisation (0: off)")
    p.add_argument("--dry_run", action="store_true", help="kinematic model and ray-cast camera, no ROS (plumbing only)")
    dag = p.add_argument_group("DAgger (stage iii, sim/dagger.py)")
    dag.add_argument("--dagger_dir", default=None, metavar="DIR",
                     help="record every frame with the expert's label to DIR/<world>.hdf5; the expert takes over when needed")
    dag.add_argument("--speed_limits", type=float, nargs="+", default=[3.0, 5.0, 7.0],
                     help="v_h of the expert reference, drawn per mission from (seed, k) as in the demonstrations")
    dag.add_argument("--gate_clearance", type=float, default=0.5, help="privileged clearance now or 0.5 s ahead (m)")
    dag.add_argument("--gate_altitude", type=float, default=0.8, help="height above the home ground (m)")
    dag.add_argument("--gate_distance", type=float, default=1.5, help="corridor half-width around the expert path (m)")
    dag.add_argument("--gate_heading", type=float, default=75.0, help="heading error to the path (deg)")
    dag.add_argument("--gate_stall_s", type=float, default=5.0, help="time without progress (s)")
    dag.add_argument("--recover_s", type=float, default=2.0, help="expert flight on the path before handing back (s)")
    dag.add_argument("--dagger_compression", choices=["gzip", "lzf", "none"], default="gzip")
    ros = p.add_argument_group("ROS 2")
    ros.add_argument("--rgb_topic", default=os.environ.get("AERO_VPA_RGB_TOPIC", "/aero_vpa/rgbd/image"))
    ros.add_argument("--depth_topic", default=os.environ.get("AERO_VPA_DEPTH_TOPIC", "/aero_vpa/rgbd/depth_image"))
    ros.add_argument("--odometry_topic", default="/fmu/out/vehicle_odometry")
    ros.add_argument("--status_topic", default="/fmu/out/vehicle_status_v1")
    ros.add_argument("--sync_slop", type=float, default=0.005)
    ros.add_argument("--control_rate", type=float, default=100.0, help="executor rate f_c (Hz)")
    ros.add_argument("--max_odometry_gap", type=float, default=0.05)
    ros.add_argument("--startup_timeout", type=float, default=60.0)
    ros.add_argument("--rearm_timeout", type=float, default=30.0)
    p.add_argument("--self-test", dest="self_test", action="store_true", help="run the self-test (dry runs) and exit")
    args = p.parse_args(argv)
    if args.self_test:
        return args
    if not args.checkpoint:
        p.error("give the policy checkpoint (or --self-test)")
    if args.episodes < 1:
        p.error("--episodes must be >= 1")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.dagger_dir and (not args.speed_limits or min(args.speed_limits) <= 0.0):
        p.error("--speed_limits must be positive")
    args.tag = args.tag or os.path.basename(args.checkpoint)
    return args


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# ---------------------------------------------------------------------------------------------
# Policy and one mission
# ---------------------------------------------------------------------------------------------
class Policy:
    """The calibrated closed loop of a checkpoint, its image normaliser and the depth range of Sec. 3.6."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device = pick_device(args.device)
        ckpt = load_checkpoint(args.checkpoint, kind="policy")
        self.cfg = config_from_dict(ckpt["config"])
        if self.cfg.sigma_lo is None or self.cfg.sigma_crit is None:
            raise SystemExit(f"{args.checkpoint} is not calibrated: run calibrate_sigma.py first")
        self.checkpoint_id = ckpt["checkpoint_id"]
        self.pipe = VPAInferencePipeline.from_checkpoint(
            args.checkpoint, latency_bound=args.latency_bound, tracking_lag=args.tracking_lag, text_encoder="auto",
            execution_mode=args.execution_mode, map_location=self.device)
        self.pipe.eval()
        self.normalizer = self.cfg.build_rgbd_normalizer().to(self.device)
        self.depth_range = DepthRange(safety_radius=self.cfg.safety_radius, max_invalid_fraction=args.max_invalid_fraction)
        self.meta = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_id": self.checkpoint_id,
                     "sigma_lo": self.cfg.sigma_lo, "sigma_crit": self.cfg.sigma_crit,
                     "predictor_residual": bool(self.cfg.predictor_residual), "latency_bound": args.latency_bound,
                     "tracking_lag": args.tracking_lag, "execution_mode": args.execution_mode, "device": str(self.device),
                     "blind_speed": self.cfg.blind_speed, "safety_radius": self.cfg.safety_radius,
                     "waypoint_radius": self.cfg.waypoint_radius,
                     "policy_state_inputs": list(self.cfg.policy_state_inputs)}


def _stats_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "p95": math.nan, "max": math.nan}
    a = 1e3 * np.asarray(values)
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


class Mission:
    """One mission flown by the policy: decisions, executor ticks, termination checks and the record."""

    def __init__(self, policy: Policy, world: E.PrivilegedMap, waypoints: np.ndarray, start_time: float,
                 expert_duration: float, args: argparse.Namespace, dagger: Optional[DaggerSupervisor] = None) -> None:
        self.policy, self.world, self.args = policy, world, args
        self.dagger = dagger
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.t0 = float(start_time)
        self.expert_duration = float(expert_duration)
        self.deadline = self.t0 + args.timeout_factor * self.expert_duration + args.timeout_extra_s
        policy.pipe.reset(None, torch.as_tensor(self.waypoints, dtype=torch.float32)[None], mission_start=self.t0)
        self.decisions = self.fallback_decisions = self.fallback_onsets = self.blind_decisions = 0
        self.dropped_frames = 0
        self.max_decision_gap = 0.0                         # longest time without a decision (from the start)
        self.prev_fallback = False
        self.latency: List[float] = []
        self.age: List[float] = []
        self.sigma_bar: List[float] = []
        self.scale: List[float] = []
        self.primitives = np.zeros(policy.cfg.num_primitives, dtype=np.int64)
        self.path_length, self.max_speed, self.speed_sum, self.speed_n = 0.0, 0.0, 0.0, 0
        self.min_clearance = math.inf
        self.not_tracking_s, self.has_chunk = 0.0, False
        self.prev_pos: Optional[np.ndarray] = None
        self.prev_tick: Optional[float] = self.t0           # the wait for the first setpoint is a gap too
        self.max_setpoint_gap = 0.0
        self.last_frame = self.t0
        self.end_time = self.t0
        self.last_mode, self.last_setpoint = -1, np.zeros(4)
        self.last_acc = np.zeros(3)                         # acceleration feed-forward of the last setpoint (traces)
        self.gap_since_row = 0.0                            # longest gap between setpoints since the last trace row
        self.trace: Optional[List[Dict[str, Any]]] = [] if getattr(args, "trace", None) else None

    @torch.no_grad()
    def decide(self, stamp: float, rgb: np.ndarray, depth: np.ndarray, position: np.ndarray, velocity: np.ndarray,
               q: np.ndarray, rates: np.ndarray, clock: Optional[Any] = None, odometry_age: float = 0.0) -> None:
        """One decision on the frame captured at ``stamp`` with the state estimate at that time; ``clock()`` gives
        the simulation time after the decision (decision age = capture to commit). With DAgger, the frame is then
        recorded with the expert's label (``dagger.DaggerSupervisor.on_frame``)."""
        p, dev = self.policy, self.policy.device
        self.max_decision_gap = max(self.max_decision_gap, float(stamp) - self.last_frame)
        self.last_frame = max(self.last_frame, float(stamp))
        image16 = E.encode_rgbd(rgb, depth)                                   # exactly as recorded for training
        rgb8 = image16[..., :3].astype(np.uint8)
        depth_m = image16[..., 3].astype(np.float32) * np.float32(E.DEPTH_SCALE_M)   # 0 = no return
        image = p.normalizer(torch.from_numpy(rgb8).permute(2, 0, 1)[None].to(dev),
                             torch.from_numpy(depth_m)[None].to(dev)).image
        yaw = float(E.yaw_from_quat(q))
        rotation = E.quat_to_rotmat(q)
        f32 = dict(dtype=torch.float32, device=dev)
        state = p.pipe.proprioception(torch.as_tensor(velocity, **f32)[None], torch.as_tensor(E.gravity_in_body(q), **f32)[None],
                                      torch.as_tensor(rates, **f32)[None], torch.tensor([yaw], **f32))
        readings = []

        def clearance(chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """d_t and the blind flag of Sec. 3.6 for a batch of one (as DepthRange.for_step), keeping the reading."""
            r = p.depth_range.evaluate_detail(depth_m, rotation, yaw, chunk[0].detach().cpu().numpy())
            readings.append(r)
            return (torch.tensor([r.range], dtype=chunk.dtype, device=chunk.device),
                    torch.tensor([r.blind], dtype=torch.bool, device=chunk.device))

        started = time.perf_counter()
        out = p.pipe.step(image, state, torch.as_tensor(position, **f32)[None], torch.tensor([yaw], **f32), clearance,
                          float(stamp))
        self.latency.append(time.perf_counter() - started)
        if clock is not None:
            self.age.append(float(clock()) - float(stamp))
        self.decisions += 1
        fallback = bool(out.fallback[0])
        self.fallback_decisions += int(fallback)
        self.fallback_onsets += int(fallback and not self.prev_fallback)
        self.prev_fallback = fallback
        self.blind_decisions += int(bool(readings and readings[0].blind))
        self.sigma_bar.append(float(out.sigma_bar[0]))
        if out.scale is not None and bool(out.committed[0]):
            self.scale.append(float(out.scale[0]))
        self.primitives[int(out.primitive[0])] += 1
        labelled = None
        if self.dagger is not None:
            labelled = self.dagger.on_frame(stamp, image16, state[0].detach().cpu().numpy(),
                                            out.goal[0].detach().cpu().numpy(), int(out.waypoint_index[0]),
                                            bool(out.task_complete[0]), position, yaw, odometry_age)
        if self.trace is not None:
            row = self._trace_row(stamp, position, velocity, yaw, q, rates, state, out, readings[0] if readings else None)
            if self.dagger is not None:
                row["dagger_mode"] = self.dagger.mode
                if math.isfinite(self.dagger.cap):
                    row["dagger_cap"] = round(float(self.dagger.cap), 2)     # the expert's speed governor
                if labelled is not None:
                    row.update(controller=labelled["controller"], tau=_rounded(labelled["tau"]),
                               e_path=_rounded(labelled["e_path"]), label0=_rounded(labelled["label0"]),
                               label_primitive=labelled["primitive"])
                choice, event = self.dagger.last_choice, self.dagger.last_event
                if choice is not None:                                       # the expert's last prediction
                    row["dagger_choice"] = {k: v for k, v in choice.items() if k != "t"}
                    if choice["t"] is not None:
                        row["dagger_choice_t"] = _rounded(choice["t"] - self.t0, 2)
                if event is not None:                                        # its last takeover or stop
                    row["dagger_event"] = [_rounded(event[0] - self.t0, 2), event[1], event[2]]
            self.trace.append(row)
        self.gap_since_row = 0.0

    def _trace_row(self, stamp: float, position: np.ndarray, velocity: np.ndarray, yaw: float, q: np.ndarray,
                   rates: np.ndarray, state: torch.Tensor, out: Any, reading: Any) -> Dict[str, Any]:
        """One decision of the --trace file (rounded; angles in degrees, frames as named)."""
        r = _rounded
        m = int(out.waypoint_index[0])
        wp = self.waypoints[min(max(m, 1), len(self.waypoints)) - 1]
        gravity = E.gravity_in_body(q)
        row: Dict[str, Any] = {
            "t": r(float(stamp) - self.t0), "compute_ms": r(1e3 * self.latency[-1], 1),
            "age_ms": r(1e3 * self.age[-1], 1) if self.age else None,
            "p": r(position), "v": r(velocity), "yaw": r(math.degrees(yaw), 1), "rates": r(rates),
            "tilt": r(math.degrees(math.acos(float(np.clip(gravity[2], -1.0, 1.0)))), 1),
            "m": m, "wp_dist": r(np.linalg.norm(np.asarray(position, dtype=np.float64) - wp)), "goal": r(out.goal[0]),
            "u_sel": int(out.selected_primitive[0]), "u": int(out.primitive[0]), "fallback": bool(out.fallback[0]),
            "sigma": r(out.sigma[0], 4), "sigma_bar": r(out.sigma_bar[0], 4), "alpha": r(out.alpha[0]),
            "H": int(out.validity_horizon[0]), "state": r(state[0]),
            "clearance": r(self.world.distance(np.asarray(position, dtype=np.float64))) if len(self.world) else None,
            "mode": self.last_mode, "setpoint": r(self.last_setpoint), "setpoint_acc": r(self.last_acc),
            "tick_gap_ms": r(1e3 * self.gap_since_row, 1),
        }
        if out.chunk_raw is not None and bool(out.committed[0]):
            raw = out.chunk_raw[0].cpu().numpy()
            row.update(raw0=r(raw[0]), raw_peak=r(np.linalg.norm(raw[:, :3], axis=1).max()),
                       cmd0=r(out.chunk[0, 0]), v_safe=r(out.v_safe[0]), scale=r(out.scale[0]))
        if reading is not None:
            row.update(d=r(reading.range), blind=bool(reading.blind), view=reading.reason, az=r(reading.azimuth, 1),
                       el=r(reading.elevation, 1))
        return row

    def save_trace(self, directory: str, record: Dict[str, Any]) -> str:
        """Write DIR/<tag>_<world>_e<k>.jsonl: the mission record, then one line per decision."""
        os.makedirs(directory, exist_ok=True)
        name = f"{record['tag']}_{record['world']}_e{int(record['episode']):03d}.jsonl"
        path = os.path.join(directory, "".join(c if c.isalnum() or c in "._-" else "_" for c in name))
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "mission", **record}, sort_keys=True) + "\n")
            for row in self.trace or []:
                f.write(json.dumps({"kind": "decision", **row}) + "\n")
        os.replace(path + ".tmp", path)
        return path

    def tick(self, now: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Executor tick: (NED velocity, NED acceleration feed-forward, yaw rate) to dispatch."""
        out = self.policy.pipe.tick(float(now))
        if self.prev_tick is not None:
            dt = float(now) - self.prev_tick
            self.max_setpoint_gap = max(self.max_setpoint_gap, dt)
            self.gap_since_row = max(self.gap_since_row, dt)
            self.has_chunk = self.has_chunk or bool(torch.isfinite(out.chunk_age[0]))
            if self.has_chunk and int(out.mode[0]) != MODE_TRACK:
                self.not_tracking_s += dt
        self.prev_tick = float(now)
        vel, acc = out.velocity[0].cpu().numpy().astype(np.float64), out.acceleration[0].cpu().numpy().astype(np.float64)
        yaw_rate = float(out.yaw_rate[0])
        self.last_mode, self.last_setpoint, self.last_acc = int(out.mode[0]), np.array([*vel, yaw_rate]), acc
        return vel, acc, yaw_rate

    def command(self, now: float, position: np.ndarray, velocity: np.ndarray, yaw: float
                ) -> Tuple[np.ndarray, np.ndarray, float]:
        """The setpoint to dispatch at ``now``: the expert's while it has control (DAgger), else the executor's.

        Under the expert the executor is not ticked; the dispatched setpoint is handed to the pipeline instead
        (a_bar of the next decision, and the point it cross-fades from when the policy flies again).
        """
        if self.dagger is not None:
            cmd = self.dagger.control(now, position, velocity, yaw)
            if cmd is not None:
                vel, acc, yaw_rate = (np.asarray(cmd[0], dtype=np.float64), np.asarray(cmd[1], dtype=np.float64),
                                      float(cmd[2]))
                self.policy.pipe.external_setpoint(torch.as_tensor(vel, dtype=torch.float32)[None],
                                                   torch.tensor([yaw_rate], dtype=torch.float32))
                if self.prev_tick is not None:
                    self.max_setpoint_gap = max(self.max_setpoint_gap, float(now) - self.prev_tick)
                    self.gap_since_row = max(self.gap_since_row, float(now) - self.prev_tick)
                self.prev_tick = float(now)
                self.last_mode, self.last_setpoint, self.last_acc = -2, np.array([*vel, yaw_rate]), acc   # -2: expert
                return vel, acc, yaw_rate
        vel, acc, yaw_rate = self.tick(now)
        if self.dagger is not None:
            self.dagger.dispatched(now, vel, acc)              # the policy's setpoint, for the expert's predictions
        return vel, acc, yaw_rate

    def observe(self, now: float, position: np.ndarray, velocity: np.ndarray) -> Optional[str]:
        """Motion statistics and the termination check at ``now``; the reason when the mission ends."""
        position = np.asarray(position, dtype=np.float64)
        if self.prev_pos is not None:
            self.path_length += float(np.linalg.norm(position - self.prev_pos))
        self.prev_pos = position.copy()
        speed = float(np.linalg.norm(velocity))
        self.max_speed = max(self.max_speed, speed)
        self.speed_sum, self.speed_n = self.speed_sum + speed, self.speed_n + 1
        clearance = float(self.world.distance(position)) if len(self.world) else math.inf
        self.min_clearance = min(self.min_clearance, clearance)
        self.end_time = float(now)
        if clearance < self.args.collision_distance:
            return "collision"
        if self.dagger is not None and self.dagger.failed:
            return self.dagger.failed
        if position[2] > -0.3:
            return "ground"
        if not self.world.inside_bounds(position):
            return "out_of_bounds"
        if bool(self.policy.pipe.task_complete[0]):
            return "success"
        if now > self.deadline:
            return "timeout"
        if now - self.last_frame > getattr(self.args, "camera_timeout", 0.5) and now - self.t0 > 1.0:
            return "camera_timeout"
        return None

    def record(self, termination: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        success = termination == "success"
        m = int(self.policy.pipe.waypoint_index[0])
        reached = len(self.waypoints) if success else max(0, m - 1)
        legs = np.diff(np.vstack([np.asarray(extra.get("home", self.waypoints[0])), self.waypoints]), axis=0)
        duration = self.end_time - self.t0
        return {
            "version": RESULT_VERSION, **extra, "success": success, "termination": termination,
            "duration_s": duration, "expert_duration_s": self.expert_duration,
            "time_ratio": duration / self.expert_duration if success and self.expert_duration > 0 else None,
            "waypoints": len(self.waypoints), "waypoints_reached": int(reached),
            "mission_length_m": float(np.linalg.norm(legs, axis=1).sum()), "waypoints_ned": self.waypoints.tolist(),
            "path_length_m": self.path_length, "mean_speed": self.speed_sum / max(1, self.speed_n),
            "final_position_ned": self.prev_pos.tolist() if self.prev_pos is not None else None,
            "max_speed": self.max_speed, "min_clearance_m": self.min_clearance if math.isfinite(self.min_clearance) else None,
            "decisions": self.decisions, "fallback_decisions": self.fallback_decisions,
            "fallback_onsets": self.fallback_onsets, "not_tracking_s": self.not_tracking_s,
            "blind_decisions": self.blind_decisions,
            "sigma_bar_max": max(self.sigma_bar) if self.sigma_bar else None,
            "sigma_bar_mean": float(np.mean(self.sigma_bar)) if self.sigma_bar else None,
            "scale_mean": float(np.mean(self.scale)) if self.scale else None,
            "primitive_counts": self.primitives.tolist(), "decision_latency_ms": _stats_ms(self.latency),
            "decision_age_ms": _stats_ms(self.age), "max_setpoint_gap_s": self.max_setpoint_gap,
            "dropped_frames": self.dropped_frames, "max_decision_gap_s": self.max_decision_gap,
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(self.dagger.summary() if self.dagger is not None else {}),
        }


def _rounded(x: Any, n: int = 3) -> Any:
    """A number or an array (NumPy, or a torch tensor on any device) as rounded Python floats for JSON."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().to("cpu", torch.float64).numpy()
    a = np.asarray(x, dtype=np.float64)
    return round(float(a), n) if a.ndim == 0 else [round(float(v), n) for v in a.ravel()]


def append_record(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def warm_up(policy: Policy, args: argparse.Namespace, decisions: int) -> List[float]:
    """Decide ``decisions`` times (with executor ticks) in a throwaway mission on a synthetic frame; returns the
    wall time of each decision (ms).

    The first decision of a process pays for lazy GPU initialisation (library handles, kernel loading), and a
    decision blocks the single-threaded ROS executor, so meanwhile no setpoint goes out and PX4 keeps executing
    the last one. Pilot 4's only crash began like that: in the first mission of its run the vehicle stayed put for
    0.76 s while turning 39 deg at the alignment's yaw rate, then caught up at 5.5 m/s and left its path. Missions
    are unchanged: the next ``Mission`` resets the pipeline, and the random-number states are restored here.
    """
    if decisions <= 0:
        return []
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    quiet = argparse.Namespace(**{**vars(args), "trace": None})
    home = np.array([0.0, 0.0, -float(args.takeoff_altitude)])
    times: List[float] = []
    try:
        mission = Mission(policy, E.PrivilegedMap(), home[None] + np.array([[10.0, 0.0, 0.0]]), 0.0, 10.0, quiet)
        rgb = np.full((*E.IMAGE_SIZE, 3), 96, dtype=np.uint8)
        depth = np.tile(np.linspace(2.0, 8.0, E.IMAGE_SIZE[1], dtype=np.float32), (E.IMAGE_SIZE[0], 1))
        level = np.array([1.0, 0.0, 0.0, 0.0])                                 # (w, x, y, z): level, facing north
        for i in range(decisions):
            stamp = 0.03 * i
            started = time.perf_counter()
            mission.decide(stamp, rgb, depth, home, np.zeros(3), level, np.zeros(3))
            for k in range(3):
                mission.tick(stamp + 0.01 * k)
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            times.append(1e3 * (time.perf_counter() - started))
        policy.pipe.external_setpoint(torch.zeros(1, 3), torch.zeros(1))      # the DAgger takeover path
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return times


def mission_for(world: E.PrivilegedMap, args: argparse.Namespace, k: int, home: np.ndarray, attempt: int = 0) -> np.ndarray:
    """The waypoints of mission ``k``: identical for every policy and every restart. ``attempt`` > 0 draws a
    replacement, in the same order for everyone, when the expert cannot plan the mission."""
    rng = np.random.default_rng([int(args.seed), int(k)] + ([int(attempt)] if attempt else []))
    return E.sample_mission(world, rng, home, tuple(args.num_waypoints), tuple(args.spacing), tuple(args.altitude))


def mission_speed(args: argparse.Namespace, k: int) -> float:
    """v_h of the expert reference of mission ``k``: --expert_speed, or with DAgger a draw from --speed_limits
    seeded with (seed, k), so that the labels have the speeds of the demonstrations."""
    if not args.dagger_dir:
        return float(args.expert_speed)
    return float(np.random.default_rng([int(args.seed), int(k), 7]).choice(np.asarray(args.speed_limits, dtype=float)))


def plan_mission(world: E.PrivilegedMap, planner: E.ExpertPlanner, args: argparse.Namespace, k: int,
                 home: np.ndarray) -> Tuple[np.ndarray, E.ExpertTrajectory]:
    """(waypoints, expert reference at ``mission_speed``) of mission ``k`` (dry run: planned in this process)."""
    for attempt in range(10):
        waypoints = mission_for(world, args, k, home, attempt)
        try:
            return waypoints, planner.plan(home, waypoints, mission_speed(args, k), 0.0)
        except E.PlanningError:
            continue
    raise E.PlanningError(f"no plannable mission {k} in 10 draws")


def world_name(args: argparse.Namespace) -> str:
    return os.environ.get("AERO_VPA_WORLD_NAME") or (os.path.splitext(os.path.basename(args.world_json))[0]
                                                     if args.world_json else "synthetic")


def dagger_gate(args: argparse.Namespace) -> GateConfig:
    return GateConfig(clearance=args.gate_clearance, altitude=args.gate_altitude, distance=args.gate_distance,
                      replan_distance=max(3.0, args.gate_distance), heading_deg=args.gate_heading,
                      stall_s=args.gate_stall_s, recover_s=args.recover_s)


def dagger_file(args: argparse.Namespace, policy: Policy) -> Tuple[str, Dict[str, Any]]:
    """(DIR/<world>.hdf5, its root attributes) of the DAgger data of this run."""
    attrs = dagger_file_attributes({"policy_checkpoint": policy.meta["checkpoint"],
                                    "policy_checkpoint_id": policy.checkpoint_id, "tag": args.tag,
                                    "gate": vars(dagger_gate(args)), "speed_limits": list(args.speed_limits),
                                    "sim_environment": _read_environment(),
                                    "fly_policy_args": {k: v for k, v in vars(args).items()}})
    return os.path.join(args.dagger_dir, world_name(args) + ".hdf5"), attrs


def make_supervisor(args: argparse.Namespace, policy: Policy, world: E.PrivilegedMap, planner: E.ExpertPlanner,
                    waypoints: np.ndarray, k: int, spline: E.MinSnapSpline, yaw: float, start_time: float, plan: Any,
                    writer: Any) -> DaggerSupervisor:
    """The DAgger supervisor of mission ``k``, whose reference is the expert's plan ``spline`` from the start."""
    attrs = {"world": world_name(args), "world_json": args.world_json or "", "seed": int(args.seed), "mission": int(k),
             "tag": args.tag, "policy_checkpoint_id": policy.checkpoint_id, "instruction": ""}
    return DaggerSupervisor(world, planner, waypoints, mission_speed(args, k), E.ExpertTrajectory(spline, yaw),
                            start_time, plan, writer, attrs, dagger_gate(args))


def _summary(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "no missions"
    ok = sum(r["success"] for r in records)
    ends: Dict[str, int] = {}
    for r in records:
        ends[r["termination"]] = ends.get(r["termination"], 0) + 1
    return f"{ok}/{len(records)} successful; " + ", ".join(f"{k} {v}" for k, v in sorted(ends.items()))


def _record_extra(args: argparse.Namespace, policy: Policy, k: int, mode: str, home: np.ndarray) -> Dict[str, Any]:
    return {"tag": args.tag, **policy.meta, "world": world_name(args), "world_json": args.world_json or "",
            "seed": int(args.seed), "episode": int(k), "mode": mode, "home": [float(v) for v in home],
            "expert_speed": mission_speed(args, k), "dagger": bool(args.dagger_dir),
            "sim_environment": _read_environment() if mode == "sitl" else {}}


# ---------------------------------------------------------------------------------------------
# Dry run: kinematic model and ray-cast camera
# ---------------------------------------------------------------------------------------------
def run_dry(args: argparse.Namespace, physics_rate: float = 250.0, camera_rate: float = 50.0) -> int:
    world = E.load_world(args.world_json) if args.world_json else E.synthetic_forest(seed=args.seed)
    policy = Policy(args)
    policy.meta["warmup_ms"] = [round(x, 1) for x in warm_up(policy, args, args.warmup)]
    planner, camera = E.ExpertPlanner(world), E.PinholeCamera()
    home = np.array([0.0, 0.0, -args.takeoff_altitude])
    render_rng = np.random.default_rng(args.seed)
    records = []
    writer = None
    if args.dagger_dir:
        os.makedirs(args.dagger_dir, exist_ok=True)
        path, attrs = dagger_file(args, policy)
        writer = InlineWriter(path, attrs, append=True,
                              compression=None if args.dagger_compression == "none" else args.dagger_compression)
    for k in range(args.episode_start, args.episode_start + args.episodes):
        waypoints, reference = plan_mission(world, planner, args, k, home)
        vehicle = E.KinematicQuadrotor(home, reference.initial_heading() or 0.0)
        supervisor = None
        if writer is not None:
            supervisor = make_supervisor(args, policy, world, planner, waypoints, k, reference.spline, vehicle.yaw, 0.0,
                                         lambda s, w, v: plan_now(planner, s, w, v), writer)
        mission = Mission(policy, world, waypoints, 0.0, reference.duration, args, supervisor)
        t, dt = 0.0, 1.0 / physics_rate
        next_cam, next_ctrl = 0.0, 0.0
        vel, acc, yr = np.zeros(3), np.zeros(3), 0.0
        termination = None
        while termination is None:
            if t >= next_cam - 1e-9:
                rgb, depth = camera.render(world, vehicle.p, vehicle.rotation(), render_rng)
                mission.decide(t, rgb, depth, vehicle.p.copy(), vehicle.v.copy(), vehicle.quaternion(),
                               np.array([0.0, 0.0, vehicle.yaw_rate]), clock=lambda t=t: t)
                next_cam += 1.0 / camera_rate
            if t >= next_ctrl - 1e-9:
                vel, acc, yr = mission.command(t, vehicle.p.copy(), vehicle.v.copy(), vehicle.yaw)
                termination = mission.observe(t, vehicle.p, vehicle.v)
                next_ctrl += 1.0 / args.control_rate
            vehicle.step(dt, vel, acc, yr)
            t += dt
        if supervisor is not None:
            supervisor.finish(termination)
        record = mission.record(termination, _record_extra(args, policy, k, "dry_run", home))
        if args.trace:
            record["trace"] = mission.save_trace(args.trace, record)
        append_record(args.results, record)
        records.append(record)
        print(f"[fly] mission {k}: {termination} after {record['duration_s']:.1f} s, {record['waypoints_reached']}/"
              f"{record['waypoints']} waypoints, {record['decisions']} decisions", flush=True)
    if writer is not None:
        writer.close()
    print(f"[fly] {args.tag}: {_summary(records)}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# ROS 2 node (PX4 SITL)
# ---------------------------------------------------------------------------------------------
def run_ros(args: argparse.Namespace) -> int:
    import message_filters
    import rclpy
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image

    px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    odom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                          history=HistoryPolicy.KEEP_LAST, depth=20)

    class PolicyFlight(Node):
        def __init__(self) -> None:
            super().__init__("aero_vpa_policy_flight",
                             parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.args = args
            self.world = E.load_world(args.world_json)
            if args.world_json is None:
                self.get_logger().warning("no --world_json / $AERO_VPA_WORLD_JSON: no collision check, no expert map")
            self.policy = Policy(args)
            try:                                       # before spinning: nothing waits on the executor yet
                times = warm_up(self.policy, args, args.warmup)
                self.policy.meta["warmup_ms"] = [round(x, 1) for x in times]
                if times:
                    self.get_logger().info("policy warm-up: decisions took " + ", ".join(f"{x:.0f}" for x in times)
                                           + " ms")
            except Exception as exc:                   # never block a flight: the first decision is just slow then
                self.get_logger().warning(f"policy warm-up failed ({type(exc).__name__}: {exc})")
            self.planner = E.ExpertPlanner(self.world)
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=_ignore_sigint)
            self.writer: Optional[WriterProcess] = None
            if args.dagger_dir:
                os.makedirs(args.dagger_dir, exist_ok=True)
                path, attrs = dagger_file(args, self.policy)
                self.writer = WriterProcess(path, attrs, append=True,
                                            compression=None if args.dagger_compression == "none" else args.dagger_compression)
                self.get_logger().info(f"DAgger: expert-labelled frames go to {path}")
            self.plan_spline: Optional[E.MinSnapSpline] = None
            self.home_nominal = np.array([0.0, 0.0, -args.takeoff_altitude])
            self.home: Optional[np.ndarray] = None
            self.odom = OdometryBuffer()
            self.status: Optional[Any] = None
            self.status_wall = -math.inf
            self.frames_seen = 0
            self.last_frame_sim = -math.inf
            self.last_frame_wall = -math.inf
            self.frame: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
            self.phase, self.phase_since = "WAIT", 0.0
            self.controller: Optional[E.ExpertController] = None
            self.plan_future: Optional[Future] = None
            self.mission: Optional[Mission] = None
            self.waypoints: Optional[np.ndarray] = None
            self.reference_duration = 0.0
            self.heading = 0.0
            self.k = args.episode_start
            self.plan_attempt = 0
            self.records: List[Dict[str, Any]] = []
            self.last_command_wall = -math.inf
            self.exit_code: Optional[int] = None
            self.start_wall = time.monotonic()
            self.ready_logged = False
            self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
            self.pub_traj = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)
            self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", px4_qos)
            self.create_subscription(VehicleOdometry, args.odometry_topic, self._on_odometry, odom_qos)
            self.create_subscription(VehicleStatus, args.status_topic, self._on_status, px4_qos)
            rgb_sub = message_filters.Subscriber(self, Image, args.rgb_topic, qos_profile=qos_profile_sensor_data)
            depth_sub = message_filters.Subscriber(self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data)
            self.sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=args.sync_slop)
            self.sync.registerCallback(self._on_rgbd)
            self.create_timer(1.0 / args.control_rate, self._on_control)
            self.create_timer(1.0, self._on_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.msgs = (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus)
            self.get_logger().info(f"policy {args.tag} ({self.policy.checkpoint_id[:8]}) on {self.policy.device}; "
                                   f"missions {args.episode_start}..{args.episode_start + args.episodes - 1}, seed {args.seed}")

        # ---------------- helpers (as in collect_expert_dataset.py) ----------------
        def sim_now(self) -> float:
            t = self.odom.now()
            return t if t is not None else self.get_clock().now().nanoseconds * 1e-9

        @staticmethod
        def px4_us(sim_time: float) -> int:
            return max(0, int(round(sim_time * 1e6)))

        def set_phase(self, phase: str) -> None:
            self.get_logger().info(f"phase {self.phase} -> {phase}")
            self.phase, self.phase_since = phase, self.sim_now()

        def command(self, cmd: int, param1: float = 0.0, param2: float = 0.0) -> None:
            _, _, VehicleCommand, _ = self.msgs
            msg = VehicleCommand()
            msg.timestamp = self.px4_us(self.sim_now())
            msg.command = cmd
            msg.param1, msg.param2 = float(param1), float(param2)
            sysid = int(self.status.system_id) if self.status is not None and self.status.system_id else 1
            msg.target_system, msg.target_component = sysid, 1
            msg.source_system, msg.source_component = 1, 1
            msg.from_external = True
            self.pub_cmd.publish(msg)
            self.last_command_wall = time.monotonic()

        def publish_setpoint(self, now: float, vel: np.ndarray, acc: np.ndarray, yaw_rate: float) -> None:
            OffboardControlMode, TrajectorySetpoint, _, _ = self.msgs
            stamp = self.px4_us(now)
            mode = OffboardControlMode()
            mode.timestamp = stamp
            mode.position, mode.velocity, mode.acceleration = False, True, False
            mode.attitude, mode.body_rate = False, False
            sp = TrajectorySetpoint()
            sp.timestamp = stamp
            nan3 = [math.nan] * 3
            sp.position, sp.jerk = nan3, nan3
            sp.velocity = [float(v) for v in vel]
            sp.acceleration = [float(a) for a in acc]
            sp.yaw, sp.yawspeed = math.nan, float(yaw_rate)
            self.pub_mode.publish(mode)
            self.pub_traj.publish(sp)

        def hover(self, now: float, pos: np.ndarray, yaw: float, yaw_target: Optional[float] = None) -> None:
            self.controller = E.ExpertController(self.planner.hold(pos, yaw, yaw_target=yaw_target), now,
                                                 E.CommandLimits(speed_h=1.0))

        # ---------------- callbacks ----------------
        def _on_status(self, msg: Any) -> None:
            self.status, self.status_wall = msg, time.monotonic()

        def _on_odometry(self, msg: Any) -> None:
            pos, vel, q = np.array(msg.position, float), np.array(msg.velocity, float), np.array(msg.q, float)
            rates = np.array(msg.angular_velocity, float)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(q))):
                return
            if msg.pose_frame != msg.POSE_FRAME_NED:
                return
            if msg.velocity_frame == msg.VELOCITY_FRAME_BODY_FRD:
                vel = E.quat_to_rotmat(q) @ vel
            elif msg.velocity_frame != msg.VELOCITY_FRAME_NED:
                return
            self.odom.add(msg.timestamp_sample * 1e-6, pos, vel, q, rates)
            self._try_decide()

        def _on_rgbd(self, rgb_msg: Any, depth_msg: Any) -> None:
            stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
            self.frames_seen += 1
            self.last_frame_sim, self.last_frame_wall = stamp, time.monotonic()
            if rgb_msg.encoding not in ("rgb8", "bgr8") or depth_msg.encoding != "32FC1":
                self.fail_run(f"unexpected encodings {rgb_msg.encoding} / {depth_msg.encoding} (want rgb8 / 32FC1)")
                return
            if (rgb_msg.height, rgb_msg.width) != E.IMAGE_SIZE or (depth_msg.height, depth_msg.width) != E.IMAGE_SIZE:
                self.fail_run(f"camera gives {rgb_msg.width}x{rgb_msg.height}; the policy needs {E.IMAGE_SIZE[1]}x"
                              f"{E.IMAGE_SIZE[0]} RGB-D (spawn the x500_vpa model)")
                return
            if self.phase != "MISSION" or self.mission is None or stamp < self.mission.t0:
                return
            rgb = decode_image(rgb_msg, 3, np.uint8)
            if rgb_msg.encoding == "bgr8":
                rgb = rgb[..., ::-1]
            if self.frame is not None:
                self.mission.dropped_frames += 1                       # superseded before a decision could use it
            self.frame = (stamp, np.ascontiguousarray(rgb), decode_image(depth_msg, 1, np.float32).copy())
            self._try_decide()

        def _try_decide(self) -> None:
            """Decide on the newest frame once odometry brackets its capture time (Algorithm 1, DECIDE)."""
            if self.frame is None or self.mission is None or self.phase != "MISSION" or self.exit_code is not None:
                return
            stamp, rgb, depth = self.frame
            latest = self.odom.latest()
            if latest is None or stamp > latest[0]:
                return
            self.frame = None
            est = self.odom.at(stamp, self.args.max_odometry_gap)
            if est is None:
                self.mission.dropped_frames += 1
                return
            pos, vel, q, rates, age = est
            try:
                self.mission.decide(stamp, rgb, depth, pos, vel, q, rates, clock=self.sim_now, odometry_age=age)
            except Exception as exc:
                self.fail_run(f"decision failed: {type(exc).__name__}: {exc}")

        def _on_watchdog(self) -> None:
            wall = time.monotonic()
            waited = wall - self.start_wall
            if self.exit_code is not None:
                return
            if self.ready_logged:
                stale = [name for name, last, limit in (("vehicle status", self.status_wall, 5.0),
                                                        ("odometry", self.odom.arrival_wall, 2.0),
                                                        ("RGB-D frames", self.last_frame_wall, 5.0)) if wall - last > limit]
                if stale:
                    self.fail_run(f"{', '.join(stale)} stopped arriving (did PX4, the DDS agent or ros_gz_bridge exit?)")
                return
            missing = [name for name, ok in (("/clock", self.get_clock().now().nanoseconds > 0),
                                             ("odometry", self.odom.latest() is not None),
                                             ("vehicle status", self.status is not None),
                                             ("RGB-D frames", self.frames_seen > 0)) if not ok]
            if not missing:
                skew = self.odom.latest()[0] - self.last_frame_sim
                if abs(skew) > 1.0:
                    self.fail_run(f"PX4 odometry stamps are {skew:+.1f} s from the image stamps: start PX4 with "
                                  "UXRCE_DDS_SYNCT=0 (run_headless_sitl.sh does)")
                    return
                self.ready_logged = True
                self.phase_since = self.sim_now()
                self.get_logger().info(f"inputs ready after {waited:.1f} s")
            elif waited > self.args.startup_timeout:
                self.fail_run("no data after %.0f s on: %s" % (waited, ", ".join(missing)))

        # ---------------- bookkeeping ----------------
        def fail_run(self, reason: str, code: int = 1) -> None:
            if self.exit_code is None:
                self.get_logger().error(reason)
                self.exit_code = code

        def finish_mission(self, termination: str, now: float, pos: np.ndarray, yaw: float) -> None:
            mission, self.mission, self.frame = self.mission, None, None
            if mission.dagger is not None:
                mission.dagger.finish(termination)
            record = mission.record(termination, _record_extra(self.args, self.policy, self.k, "sitl",
                                                               self.home if self.home is not None else self.home_nominal))
            if self.args.trace:
                try:
                    record["trace"] = mission.save_trace(self.args.trace, record)
                except OSError as exc:
                    self.get_logger().warning(f"could not write the trace: {exc}")
            append_record(self.args.results, record)
            self.records.append(record)
            age = record["decision_age_ms"]
            self.get_logger().info(
                f"mission {self.k}: {termination} after {record['duration_s']:.1f} s; {record['waypoints_reached']}/"
                f"{record['waypoints']} waypoints; {record['fallback_onsets']} fallback onsets; decision age p95 "
                f"{age['p95']:.0f} ms, compute p95 {record['decision_latency_ms']['p95']:.0f} ms")
            if age["p95"] > 1e3 * self.args.latency_bound:
                self.get_logger().warning(f"decision age p95 {age['p95']:.0f} ms exceeds --latency_bound "
                                          f"{1e3 * self.args.latency_bound:.0f} ms: the braking envelope assumes less")
            self.k += 1
            if termination in CRASHES:
                self.get_logger().error(f"{termination}: ending the run; restart the simulator and continue with "
                                        f"--episode_start {self.k}")
                self.set_phase("DONE")
                self.exit_code = 3
                return
            self.hover(now, pos, yaw)
            self.plan_future = self.pool.submit(E.plan_mission_spline, self.args.world_json, pos,
                                                (self.home if self.home is not None else self.home_nominal)[None],
                                                self.args.return_speed)
            self.set_phase("RETURN_PLAN")

        # ---------------- flight logic (simulation-time timer) ----------------
        def _on_control(self) -> None:
            if self.exit_code is not None:
                return
            try:
                if self.writer is not None:
                    self.writer.check()
                self._step()
            except Exception as exc:
                self.fail_run(f"{type(exc).__name__}: {exc}")

        def _step(self) -> None:
            now = self.sim_now()
            last = self.odom.latest()
            st = self.status
            _, _, VehicleCommand, VehicleStatus = self.msgs
            armed = st is not None and st.arming_state == VehicleStatus.ARMING_STATE_ARMED
            offboard = st is not None and st.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            if self.phase in ("WAIT", "ARM") or last is None:
                self.publish_setpoint(now, np.zeros(3), np.zeros(3), 0.0)
            if last is None or not self.ready_logged:
                return
            pos, vel, yaw = last[1], last[2], E.yaw_from_quat(last[3])
            elapsed = now - self.phase_since

            if self.phase == "WAIT":
                if not armed and st.pre_flight_checks_pass and elapsed > 1.5:
                    self.set_phase("ARM")
                elif elapsed > self.args.rearm_timeout:
                    raise RuntimeError(f"PX4 pre-flight checks did not pass within {self.args.rearm_timeout:.0f} s")
                return
            if self.phase == "ARM":
                if armed and offboard:
                    self.home = pos + np.array([0.0, 0.0, -self.args.takeoff_altitude])
                    self.controller = E.ExpertController(self.planner.hold(pos, yaw, target=self.home, speed=1.0), now,
                                                         E.CommandLimits(speed_h=1.0))
                    self.set_phase("TAKEOFF")
                elif time.monotonic() - self.last_command_wall > 0.5:
                    if elapsed > 20.0:
                        raise RuntimeError("PX4 did not accept offboard mode and arming within 20 s")
                    if not offboard:
                        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    else:
                        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                return
            if self.phase == "LAND":
                if offboard and self.controller is not None:
                    v_cmd, a_ff, yr = self.controller.command(now, pos, yaw)
                    self.publish_setpoint(now, v_cmd, a_ff, yr)
                if not armed:
                    self.set_phase("DONE")
                    self.exit_code = 0
                elif time.monotonic() - self.last_command_wall > 2.0:
                    if elapsed > 90.0:
                        raise RuntimeError("landing did not complete within 90 s")
                    self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                return
            if not (armed and offboard):
                if self.mission is not None:
                    self.finish_mission("px4_left_offboard", now, pos, yaw)
                raise RuntimeError(f"PX4 left offboard (arming_state {st.arming_state}, nav_state {st.nav_state})")

            if self.phase == "MISSION":
                self._try_decide()
                v_cmd, a_ff, yr = self.mission.command(now, pos, vel, yaw)
                self.publish_setpoint(now, v_cmd, a_ff, yr)
                reason = self.mission.observe(now, pos, vel)
                if reason is None and now - self.last_frame_sim > self.args.camera_timeout and now - self.mission.t0 > 1.0:
                    reason = "camera_timeout"
                if reason is not None:
                    self.finish_mission(reason, now, pos, yaw)
                return
            if self.phase == "TAKEOFF":
                done = self.controller.finished(now) and np.linalg.norm(pos - self.home) < 0.3
                if done and np.linalg.norm(vel) < 0.3:
                    self.hover(now, pos, yaw)
                    self.set_phase("PLAN")
                elif elapsed > 30.0:
                    raise RuntimeError("take-off did not reach the target altitude within 30 s")
            elif self.phase == "PLAN":
                if self.plan_future is None:
                    self.waypoints = mission_for(self.world, self.args, self.k, self.home_nominal, self.plan_attempt)
                    self.plan_future = self.pool.submit(E.plan_mission_spline, self.args.world_json, self.home_nominal,
                                                        self.waypoints, mission_speed(self.args, self.k))
                elif self.plan_future.done():
                    future, self.plan_future = self.plan_future, None
                    try:
                        spline = future.result()
                    except E.PlanningError as exc:
                        self.plan_attempt += 1
                        if self.plan_attempt >= 10:
                            raise RuntimeError(f"no plannable mission {self.k} in 10 draws") from exc
                        self.get_logger().warning(f"expert cannot plan mission {self.k} ({exc}); drawing replacement "
                                                  f"{self.plan_attempt}")
                        return
                    self.plan_attempt = 0
                    self.plan_spline = spline
                    reference = E.ExpertTrajectory(spline, yaw)
                    self.reference_duration = reference.duration
                    self.heading = reference.initial_heading() or yaw
                    self.hover(now, pos, yaw, yaw_target=self.heading)
                    self.set_phase("ALIGN")
                elif elapsed > 60.0:
                    raise RuntimeError("expert planning took more than 60 s")
            elif self.phase == "ALIGN":
                if (abs(float(E.wrap_angle(self.heading - yaw))) < 0.08 and elapsed > 0.5) or elapsed > 8.0:
                    supervisor = None
                    if self.writer is not None:
                        supervisor = make_supervisor(
                            self.args, self.policy, self.world, self.planner, self.waypoints, self.k, self.plan_spline,
                            yaw, now, lambda s, w, v: self.pool.submit(E.plan_mission_spline, self.args.world_json, s, w, v),
                            self.writer)
                    self.mission = Mission(self.policy, self.world, self.waypoints, now, self.reference_duration, self.args,
                                           supervisor)
                    self.frame = None
                    self.get_logger().info(f"mission {self.k}: {len(self.waypoints)} waypoints, reference "
                                           f"{self.reference_duration:.1f} s at {mission_speed(self.args, self.k):g} m/s"
                                           + (" (DAgger)" if supervisor is not None else ""))
                    self.set_phase("MISSION")
                    return
            elif self.phase == "RETURN_PLAN":
                if self.plan_future.done():
                    future, self.plan_future = self.plan_future, None
                    try:
                        spline = future.result()
                    except E.PlanningError as exc:
                        self.get_logger().error(f"no path home ({exc}): ending the run; restart and continue with "
                                                f"--episode_start {self.k}")
                        self.set_phase("DONE")
                        self.exit_code = 3
                        return
                    trajectory = E.ExpertTrajectory(spline, yaw)
                    self.controller = E.ExpertController(trajectory, now, E.CommandLimits(speed_h=self.args.return_speed))
                    self.set_phase("RETURN")
                elif elapsed > 60.0:
                    raise RuntimeError("planning the flight home took more than 60 s")
            elif self.phase == "RETURN":
                ctl = self.controller
                if ctl.finished(now, settle=1.0) and np.linalg.norm(pos - self.home) < 0.5:
                    self.hover(now, pos, yaw)
                    if self.k >= self.args.episode_start + self.args.episodes:
                        self.set_phase("LAND")
                    else:
                        self.set_phase("PLAN")
                elif elapsed > 2.0 * ctl.trajectory.duration + 20.0:
                    self.get_logger().error("the flight home did not arrive: ending the run")
                    self.set_phase("DONE")
                    self.exit_code = 3
                    return
            if self.controller is not None:
                v_cmd, a_ff, yr = self.controller.command(now, pos, yaw)
                self.publish_setpoint(now, v_cmd, a_ff, yr)

        def shutdown(self) -> None:
            if self.mission is not None:
                self.get_logger().info(f"mission {self.k} interrupted after {self.mission.end_time - self.mission.t0:.1f} s; "
                                       "not recorded (it is flown again on the next run)")
                if self.mission.dagger is not None:
                    try:
                        self.mission.dagger.finish("interrupted")
                    except Exception as exc:  # the writer may already have failed; the run is ending anyway
                        self.get_logger().warning(f"could not close the DAgger segment: {exc}")
                if self.args.trace and self.mission.trace:
                    try:
                        partial = self.mission.record("interrupted", _record_extra(
                            self.args, self.policy, self.k, "sitl", self.home if self.home is not None else self.home_nominal))
                        self.mission.save_trace(self.args.trace, partial)
                    except (OSError, ValueError):
                        pass
            self.pool.shutdown(wait=False, cancel_futures=True)
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception as exc:
                    self.get_logger().error(f"DAgger writer: {exc}")

    sys.setswitchinterval(0.001)
    rclpy.init()
    node = PolicyFlight()
    code = 1
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = node.exit_code if node.exit_code is not None else 130
    except KeyboardInterrupt:
        code = 130
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"[fly] {args.tag}: {_summary(node.records)} (exit {code})", flush=True)
    return code


# ---------------------------------------------------------------------------------------------
# Self-test (dry runs with an untrained, calibrated policy)
# ---------------------------------------------------------------------------------------------
def _demo_policy(path: str) -> None:
    """A policy checkpoint with random weights and fixed calibration: plumbing only."""
    from checkpoint import environment_info, make_checkpoint, null_text_encoder, save_checkpoint
    from config import VPAConfig

    cfg = VPAConfig(sigma_lo=0.2, sigma_crit=0.7)
    torch.manual_seed(0)
    pipe = VPAInferencePipeline.from_config(cfg, null_text_encoder(cfg.text_dim), latency_bound=0.05, tracking_lag=0.03)
    mu_z, gamma = torch.zeros(cfg.latent_dim), 1.0
    mu_s, sd_s = torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim)
    pipe.selector.set_input_standardization(mu_z, gamma, mu_s, sd_s)
    pipe.calibrate(gamma, mu_z, mu_s, sd_s)
    states = {"vision_encoder": pipe.vision_encoder, "predictor": pipe.predictor, "selector": pipe.selector,
              "solver": pipe.solver, "text_null_embedding": pipe.text_encoder.null_embedding.detach()}
    calibration = {"gamma_bar_star": gamma, "latent_mean": mu_z, "proprio_mean": mu_s, "proprio_std": sd_s}
    save_checkpoint(path, make_checkpoint("policy", cfg, states, calibration, {"note": "fly_policy self-test",
                                                                                **environment_info()}))


def _self_test() -> None:
    import tempfile

    from summarize_flights import load, report

    # trace values: tensors NumPy cannot read directly (requires_grad here, CUDA on a GPU) are copied first
    assert _rounded(torch.tensor([1.23456, 2.0], requires_grad=True), 2) == [1.23, 2.0]
    assert _rounded(torch.tensor(0.5), 1) == 0.5 and _rounded(np.float32(2.25), 1) == 2.2 and _rounded(None) is None
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, results = os.path.join(tmp, "policy.pt"), os.path.join(tmp, "flights.jsonl")
        _demo_policy(ckpt)
        base = [ckpt, "--dry_run", "--seed", "3", "--spacing", "4", "6", "--num_waypoints", "1", "1", "--timeout_factor",
                "0.5", "--timeout_extra_s", "1", "--results", results, "--device", "cpu"]
        assert run_dry(parse_args(base + ["--episodes", "2"])) == 0
        trace_dir = os.path.join(tmp, "trace")
        assert run_dry(parse_args(base + ["--episodes", "1", "--episode_start", "2", "--tag", "again",
                                          "--trace", trace_dir])) == 0
        records = [json.loads(line) for line in open(results, encoding="utf-8")]
        assert [(r["tag"], r["episode"]) for r in records] == [("policy.pt", 0), ("policy.pt", 1), ("again", 2)]
        assert all(0.0 < r["max_decision_gap_s"] < 0.5 for r in records), [r["max_decision_gap_s"] for r in records]
        # --camera_timeout: frames 0.8 s apart end a mission after 0.5 s without a decision, not after 2 s
        slow = os.path.join(tmp, "slow.jsonl")
        for limit in ("0.5", "2"):
            assert run_dry(parse_args(base[:-4] + ["--results", slow, "--device", "cpu", "--episodes", "1",
                                                   "--episode_start", "3", "--camera_timeout", limit]),
                           camera_rate=1.25) == 0
        fast, patient = [json.loads(line) for line in open(slow, encoding="utf-8")]
        assert fast["termination"] == "camera_timeout" and patient["termination"] != "camera_timeout", (fast, patient)
        assert 0.7 < patient["max_decision_gap_s"] < 0.9, patient["max_decision_gap_s"]
        # --trace: the mission record, then one row per decision
        assert "trace" not in records[0] and os.path.isfile(records[2]["trace"])
        assert os.path.basename(records[2]["trace"]) == "again_synthetic_e002.jsonl"
        lines = [json.loads(line) for line in open(records[2]["trace"], encoding="utf-8")]
        assert lines[0]["kind"] == "mission" and lines[0]["termination"] == records[2]["termination"]
        rows = lines[1:]
        assert len(rows) == records[2]["decisions"] and all(row["kind"] == "decision" for row in rows)
        assert {"t", "p", "v", "yaw", "m", "wp_dist", "goal", "u_sel", "u", "sigma_bar", "state", "mode",
                "clearance"} <= set(rows[0])
        assert sum(bool(row.get("blind")) for row in rows) == records[2]["blind_decisions"]
        assert all(row["view"] in REASONS for row in rows if "view" in row)
        assert any("raw0" in row and "scale" in row for row in rows) and len(records[2]["final_position_ned"]) == 3
        need = {"success", "termination", "duration_s", "expert_duration_s", "waypoints_ned", "path_length_m",
                "min_clearance_m", "decisions", "fallback_onsets", "decision_latency_ms", "decision_age_ms",
                "checkpoint_id", "sigma_crit", "world", "seed", "episode", "primitive_counts"}
        for r in records:
            assert need <= set(r) and r["decisions"] > 0 and r["mode"] == "dry_run" and r["world"] == "synthetic"
            assert r["termination"] in ("success", "timeout", "collision", "ground", "out_of_bounds", "camera_timeout")
            assert sum(r["primitive_counts"]) == r["decisions"] and r["path_length_m"] >= 0.0
        # missions depend only on (seed, index): identical for every policy and after a restart
        args = parse_args(base)
        world, home = E.synthetic_forest(seed=3), np.array([0.0, 0.0, -args.takeoff_altitude])
        assert np.allclose(mission_for(world, args, 0, home), records[0]["waypoints_ned"])
        assert np.allclose(plan_mission(world, E.ExpertPlanner(world), args, 2, home)[0], records[2]["waypoints_ned"])
        assert not np.allclose(mission_for(world, args, 1, home), records[0]["waypoints_ned"])
        assert not np.allclose(mission_for(world, args, 0, home, attempt=1), records[0]["waypoints_ned"])
        out = report(load([results]))
        assert out["stats"]["policy.pt"]["missions"] == 2 and out["stats"]["again"]["missions"] == 1
        assert records[0]["warmup_ms"] and len(records[0]["warmup_ms"]) == args.warmup == 3     # on by default

        # the warm-up runs decisions of a throwaway mission and changes nothing in the real ones
        flown = []
        for warm in ("0", "2"):
            path = os.path.join(tmp, f"warm{warm}.jsonl")
            torch.manual_seed(11)
            assert run_dry(parse_args(base + ["--episodes", "1", "--episode_start", "5", "--warmup", warm,
                                              "--results", path])) == 0
            flown.append(json.loads(open(path, encoding="utf-8").readline()))
        cold, warm_rec = flown
        assert cold["warmup_ms"] == [] and len(warm_rec["warmup_ms"]) == 2 and min(warm_rec["warmup_ms"]) > 0.0
        assert 0.0 <= cold["max_setpoint_gap_s"] < 0.05          # dry run: a setpoint every 10 ms from the start
        # a first setpoint that comes late (the control loop blocked at the mission start) is a setpoint gap
        late = Mission(Policy(args), E.PrivilegedMap(), np.array([[10.0, 0.0, -2.5]]), 100.0, 10.0, args)
        late.command(100.7, np.array([0.0, 0.0, -2.5]), np.zeros(3), 0.0)
        assert math.isclose(late.max_setpoint_gap, 0.7, abs_tol=1e-9), late.max_setpoint_gap
        same = ("termination", "decisions", "waypoints_reached", "fallback_onsets", "blind_decisions", "primitive_counts")
        assert all(cold[k] == warm_rec[k] for k in same), {k: (cold[k], warm_rec[k]) for k in same}
        for k in ("path_length_m", "sigma_bar_max", "min_clearance_m"):
            assert math.isclose(cold[k], warm_rec[k], rel_tol=1e-9, abs_tol=1e-9), (k, cold[k], warm_rec[k])

        # DAgger (sim/dagger.py): the untrained policy flies, the expert labels every frame and takes over
        import h5py

        import inspect_hdf5
        from config import VPAConfig
        from dataset import AeroVPADataset

        dagger_dir, dagger_results = os.path.join(tmp, "dagger"), os.path.join(tmp, "dagger.jsonl")
        dag = [ckpt, "--dry_run", "--seed", "4", "--spacing", "6", "8", "--num_waypoints", "2", "2", "--timeout_factor",
               "3", "--timeout_extra_s", "20", "--results", dagger_results, "--device", "cpu", "--dagger_dir", dagger_dir,
               "--speed_limits", "3", "--episodes", "1", "--trace", trace_dir, "--tag", "dag"]
        assert run_dry(parse_args(dag), camera_rate=10.0) == 0                     # 10 Hz frames: a faster test
        rec = [json.loads(line) for line in open(dagger_results, encoding="utf-8")][-1]
        assert rec["dagger"] and rec["dagger_speed_limit"] == 3.0 and rec["expert_speed"] == 3.0
        frames = rec["dagger_frames_policy"] + rec["dagger_frames_expert"]
        assert frames > 0 and sum(rec["dagger_takeovers"].values()) >= 1 and rec["dagger_segments"] >= 1
        path = os.path.join(dagger_dir, "synthetic.hdf5")
        report_h5 = inspect_hdf5.validate_file(path, VPAConfig())
        assert not report_h5["errors"], report_h5["errors"]
        assert len(AeroVPADataset(path, 2)) == frames
        with h5py.File(path, "r") as f:
            assert f.attrs["source"] == "dagger" and f.attrs["policy_checkpoint_id"] == rec["checkpoint_id"]
            eps = [f[k] for k in sorted(f) if k.startswith("episode_")]
            assert all(int(ep.attrs["mission"]) == 0 and float(ep.attrs["speed_limit"]) == 3.0 for ep in eps)
        rows = [json.loads(line) for line in open(rec["trace"], encoding="utf-8")][1:]
        assert {"dagger_mode", "setpoint_acc", "tick_gap_ms"} <= set(rows[0])
        assert any(r.get("controller") == "expert" for r in rows)
        assert all(0.0 <= r["tick_gap_ms"] < 50.0 for r in rows[1:])            # dry run: a setpoint every 10 ms
        choices = [r["dagger_choice"] for r in rows if "dagger_choice" in r]   # the expert's predictions
        assert choices and all(c["way"] in ("fly", "slow", "stop") and "fly" in c for c in choices)
        assert any(r.get("dagger_event", [None, None])[1] in TRIGGERS for r in rows)          # a takeover and its way
        labelled = [r for r in rows if "controller" in r]
        assert len(labelled) >= frames and all(len(r["label0"]) == 4 for r in labelled)
        # a second run appends to the same file
        assert run_dry(parse_args(dag[:-6] + ["--episodes", "1", "--episode_start", "1", "--tag", "dag"]),
                       camera_rate=10.0) == 0
        with h5py.File(path, "r") as f:
            missions = sorted({int(f[k].attrs["mission"]) for k in f if k.startswith("episode_")})
        assert missions == [0, 1], missions
    print("fly_policy.py self-test passed")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    return run_dry(args) if args.dry_run else run_ros(args)


if __name__ == "__main__":
    sys.exit(main())
