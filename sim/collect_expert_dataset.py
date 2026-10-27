# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
sim/collect_expert_dataset.py -- Privileged-expert demonstrations from Gazebo Harmonic + PX4 SITL
(preprint_261026.pdf, Secs. 3.8 and 5.1), written to HDF5 for stages (i) and (ii).

Run inside the stack of ``run_headless_sitl.sh`` (which exports the world, topics and ROS domain):

    sim/run_headless_sitl.sh --px4-dir ~/PX4-Autopilot --ros-ws ~/px4_ros_ws -- \\
        python3 sim/collect_expert_dataset.py --num_episodes 2 --output pilot_sample.hdf5

or offline, without ROS or Gazebo (kinematic quadrotor, ray-cast RGB-D; same recording code and schema):

    python3 sim/collect_expert_dataset.py --dry_run --num_episodes 2 --output pilot_sample.hdf5

ROS 2 interface (simulation time, ``use_sim_time``):
    in   /aero_vpa/rgbd/image, /aero_vpa/rgbd/depth_image  (sensor_msgs/Image, synchronised with
         message_filters.ApproximateTimeSynchronizer), /fmu/out/vehicle_odometry, /fmu/out/vehicle_status_v1
    out  /fmu/in/offboard_control_mode and /fmu/in/trajectory_setpoint at 100 Hz (velocity + acceleration
         feed-forward + yaw rate, NED, Sec. 4.3), /fmu/in/vehicle_command (offboard mode, arm, land)

One sample per synchronised RGB-D frame (50 Hz with x500_vpa). The odometry is a separate, faster stream
without a ROS header, so it is not put through the synchroniser (which would throttle the frames to the
5 Hz status rate and match on arrival time); instead each frame is paired with the odometry interpolated
at its capture time, using the PX4 sample timestamps mapped to simulation time. S_t uses the setpoint
dispatched most recently before the capture time (Eq. 2), and the expert chunk is evaluated at the
capture time, so every sample is consistent at tau_t.

Flight sequence: wait for PX4 and the sensors, stream setpoints, switch to offboard and arm, take off, and
then for each episode: sample a mission (3-6 waypoints, 10-40 m apart), plan it on the privileged map in a
worker process (hovering meanwhile), turn to the initial heading, fly it while recording, hover. Episodes
are chained in the air; after the last one, or after a failed one, the vehicle lands and re-arms. A
collision ends the run instead (exit status 3): the vehicle may be tipped over or caught in a tree, and PX4
then never passes its pre-flight checks again, so the simulator has to be restarted. Episodes recorded
before it stay in the file; ``--append`` continues it (sim/collect_worlds.sh does this). HDF5 is written by
a separate process so that compression never delays the 100 Hz setpoint stream.

Exit status: 0 when every requested episode succeeded, 1 on errors, 2 when too many episodes failed in a
row, 3 after a collision (restart the simulator and append).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert as E  # noqa: E402

COLLECTOR_VERSION = "1.0.0"


# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num_episodes", type=int, default=2, help="successful episodes to record (pilot: 2; scale-up: 2000)")
    p.add_argument("--output", default="pilot_sample.hdf5", help="HDF5 file")
    p.add_argument("--append", action="store_true", help="append episodes to an existing file of the same schema")
    p.add_argument("--world_json", default=os.environ.get("AERO_VPA_WORLD_JSON"),
                   help="privileged map (world JSON sidecar); default: $AERO_VPA_WORLD_JSON")
    p.add_argument("--seed", type=int, default=0, help="seed of the mission sampler")
    p.add_argument("--speed_limits", type=float, nargs="+", default=[3.0, 5.0, 7.0], help="v_h per episode (m/s)")
    p.add_argument("--num_waypoints", type=int, nargs=2, default=[3, 6])
    p.add_argument("--spacing", type=float, nargs=2, default=[10.0, 40.0], help="waypoint spacing (m)")
    p.add_argument("--altitude", type=float, nargs=2, default=[1.5, 4.0], help="waypoint altitude above the origin (m)")
    p.add_argument("--takeoff_altitude", type=float, default=2.5)
    p.add_argument("--instruction", default="", help="optional instruction x stored with every episode")
    p.add_argument("--max_failures", type=int, default=5, help="consecutive failed episodes before giving up")
    p.add_argument("--rearm_timeout", type=float, default=30.0,
                   help="simulation seconds to wait for PX4's pre-flight checks before arming")
    p.add_argument("--keep_failed", action="store_true", help="keep failed episodes (success=False) in the file")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    p.add_argument("--dry_run", action="store_true", help="no ROS/Gazebo: kinematic model and ray-cast camera")
    ros = p.add_argument_group("ROS 2")
    ros.add_argument("--rgb_topic", default=os.environ.get("AERO_VPA_RGB_TOPIC", "/aero_vpa/rgbd/image"))
    ros.add_argument("--depth_topic", default=os.environ.get("AERO_VPA_DEPTH_TOPIC", "/aero_vpa/rgbd/depth_image"))
    ros.add_argument("--odometry_topic", default="/fmu/out/vehicle_odometry")
    ros.add_argument("--status_topic", default="/fmu/out/vehicle_status_v1", help="/fmu/out/vehicle_status before PX4 1.16")
    ros.add_argument("--sync_slop", type=float, default=0.005, help="ApproximateTimeSynchronizer slop (s)")
    ros.add_argument("--control_rate", type=float, default=100.0, help="setpoint rate f_c (Hz)")
    ros.add_argument("--max_odometry_gap", type=float, default=0.05, help="largest odometry gap to interpolate across (s)")
    ros.add_argument("--startup_timeout", type=float, default=60.0, help="wall-clock seconds to wait for every input")
    args = p.parse_args(argv)
    if args.num_episodes < 1:
        p.error("--num_episodes must be >= 1")
    return args


def _read_environment() -> Dict[str, str]:
    """Provenance written by run_headless_sitl.sh (world, model, PX4 and Gazebo versions)."""
    run_dir = os.environ.get("AERO_VPA_RUN_DIR")
    out: Dict[str, str] = {}
    if run_dir and os.path.isfile(os.path.join(run_dir, "environment.txt")):
        with open(os.path.join(run_dir, "environment.txt"), encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.rstrip("\n").split("=", 1)
                    out[key] = value
    return out


def _file_attrs(args: argparse.Namespace, source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "collector_version": COLLECTOR_VERSION,
        "label_thresholds": json.dumps(asdict(E.LabelThresholds())),
        "label_command": "first_knot",
        "sim_environment": json.dumps(_read_environment(), sort_keys=True),
        "collector_args": json.dumps({k: v for k, v in vars(args).items()}, sort_keys=True, default=str),
    }


# ---------------------------------------------------------------------------------------------
# HDF5 writer process (keeps compression off the control loop)
# ---------------------------------------------------------------------------------------------
def _ignore_sigint() -> None:
    """Worker processes leave Ctrl-C to the parent, which closes the episode and then stops them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _writer_main(path: str, attrs: Dict[str, Any], mode: str, compression: Optional[str],
                 commands: "mp.Queue[Any]", results: "mp.Queue[Any]") -> None:
    _ignore_sigint()
    writer = None
    episode: Optional[E.EpisodeWriter] = None
    name = ""
    try:
        writer = E.DatasetWriter(path, attrs, mode=mode, compression=compression)
        results.put(("ready", None))
        while True:
            cmd, payload = commands.get()
            if cmd == "begin":
                name, episode = writer.begin_episode()
                results.put(("begun", name))
            elif cmd == "append":
                episode.append(payload)
            elif cmd == "finalize":
                frames = episode.finalize(payload)
                writer.flush()
                results.put(("finalized", (name, frames)))
                episode = None
            elif cmd == "discard":
                writer.discard(name)
                writer.flush()
                results.put(("discarded", name))
                episode = None
            elif cmd == "close":
                break
    except Exception as exc:  # reported to the parent, which aborts the run
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if writer is not None:
            writer.close()
        results.put(("closed", None))


class WriterProcess:
    def __init__(self, path: str, attrs: Dict[str, Any], append: bool, compression: Optional[str]) -> None:
        ctx = mp.get_context("spawn")
        self.commands: "mp.Queue[Any]" = ctx.Queue(maxsize=2000)
        self.results: "mp.Queue[Any]" = ctx.Queue()
        self.process = ctx.Process(target=_writer_main, name="hdf5-writer", daemon=True,
                                   args=(path, attrs, "a" if append else "w", compression, self.commands, self.results))
        self.process.start()
        kind, payload = self.results.get(timeout=60)
        if kind != "ready":
            raise RuntimeError(f"HDF5 writer failed to start: {payload}")

    def send(self, cmd: str, payload: Any = None) -> None:
        self.check()
        self.commands.put((cmd, payload), timeout=10)

    def check(self) -> None:
        while True:
            try:
                kind, payload = self.results.get_nowait()
            except queue.Empty:
                return
            if kind == "error":
                raise RuntimeError(f"HDF5 writer: {payload}")
            if kind in ("finalized", "discarded"):
                print(f"[collector] {kind} {payload}", flush=True)

    def close(self, timeout: float = 600.0) -> None:
        if self.process.is_alive():
            self.commands.put(("close", None))
            self.process.join(timeout)
        self.check()


# ---------------------------------------------------------------------------------------------
# Odometry buffer (PX4 timestamps mapped to simulation time)
# ---------------------------------------------------------------------------------------------
class OdometryBuffer:
    """Recent odometry samples; ``at(t)`` interpolates position, velocity, rates and attitude at time ``t``.

    Sample times are the PX4 ``timestamp_sample`` values. With UXRCE_DDS_SYNCT=0 (set by
    run_headless_sitl.sh) PX4 stamps in its lockstep clock, which gz_bridge sets from the Gazebo clock, so
    they are simulation time, the clock of the image stamps. ``now()`` extrapolates the newest stamp with
    the wall clock: a fresher estimate of simulation time than the ROS /clock, which is delivered through
    the same executor as the images and can lag by tens of milliseconds.
    """

    def __init__(self, horizon: float = 2.0) -> None:
        self.samples: Deque[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self.horizon = horizon
        self.arrival_wall = -math.inf

    def add(self, t: float, position: np.ndarray, velocity: np.ndarray, q: np.ndarray, rates: np.ndarray) -> None:
        if self.samples and t <= self.samples[-1][0]:
            return
        self.samples.append((t, position, velocity, q, rates))
        self.arrival_wall = time.monotonic()
        while self.samples and self.samples[0][0] < t - self.horizon:
            self.samples.popleft()

    def now(self, max_extrapolation: float = 0.2) -> Optional[float]:
        """Simulation time now, from the newest sample (None if odometry is older than ``max_extrapolation``)."""
        if not self.samples:
            return None
        since = time.monotonic() - self.arrival_wall
        return self.samples[-1][0] + since if since <= max_extrapolation else None

    def latest(self) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        return self.samples[-1] if self.samples else None

    def at(self, t: float, max_gap: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """(position, velocity, q, rates, age) at ``t``; None if ``t`` is not bracketed within ``max_gap``."""
        s = self.samples
        if len(s) < 2 or t < s[0][0] or t > s[-1][0]:
            return None
        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = s[lo], s[hi]
        if b[0] - a[0] > max_gap:
            return None
        w = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
        qa, qb = a[3], b[3] if float(np.dot(a[3], b[3])) >= 0.0 else -b[3]
        q = (1.0 - w) * qa + w * qb
        return ((1 - w) * a[1] + w * b[1], (1 - w) * a[2] + w * b[2], q / np.linalg.norm(q),
                (1 - w) * a[4] + w * b[4], t - a[0])


def decode_image(msg: Any, channels: int, dtype: Any) -> np.ndarray:
    """sensor_msgs/Image -> ``[H, W]`` or ``[H, W, C]`` array, honouring the row stride."""
    if msg.is_bigendian:
        raise ValueError("big-endian images are not supported")
    itemsize = np.dtype(dtype).itemsize
    row = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * channels * itemsize]
    arr = np.ascontiguousarray(row).view(dtype).reshape(msg.height, msg.width, channels)
    return arr[..., 0] if channels == 1 else arr


# ---------------------------------------------------------------------------------------------
# ROS 2 node
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

    class ExpertCollector(Node):
        def __init__(self) -> None:
            super().__init__("aero_vpa_expert_collector",
                             parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.args = args
            self.rng = np.random.default_rng(args.seed)
            self.world = E.load_world(args.world_json)
            if args.world_json is None:
                self.get_logger().warning("no --world_json / $AERO_VPA_WORLD_JSON: planning without obstacles")
            self.thresholds = E.LabelThresholds()
            self.planner = E.ExpertPlanner(self.world)
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=_ignore_sigint)
            self.writer = WriterProcess(args.output, _file_attrs(args, "gazebo + px4_sitl"), args.append,
                                        None if args.compression == "none" else args.compression)
            self.odom = OdometryBuffer()
            self.status: Optional[Any] = None
            self.status_wall = -math.inf
            self.frames_seen = 0
            self.last_frame_sim = -math.inf
            self.last_frame_wall = -math.inf
            self.pending: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=50)
            self.setpoints: Deque[Tuple[float, np.ndarray]] = deque(maxlen=400)
            self.setpoint_gap_max = 0.0
            self.phase = "WAIT"
            self.phase_since = 0.0
            self.controller: Optional[E.ExpertController] = None
            self.plan_future: Optional[Future] = None
            self.plan_spline: Optional[E.MinSnapSpline] = None
            self.mission: Optional[np.ndarray] = None
            self.speed = 0.0
            self.tracker: Optional[E.WaypointTracker] = None
            self.episode_open = False
            self.episode_stats: Dict[str, Any] = {}
            self.successes = 0
            self.consecutive_failures = 0
            self.attempts = 0
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
            self.create_timer(1.0 / args.control_rate, self._on_control)        # simulation time
            self.create_timer(1.0, self._on_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.msgs = (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus)

        # ---------------- helpers ----------------
        def ros_now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def sim_now(self) -> float:
            """Simulation time: extrapolated odometry stamp when fresh, else the ROS clock (/clock)."""
            t = self.odom.now()
            return t if t is not None else self.ros_now()

        @staticmethod
        def px4_us(sim_time: float) -> int:
            return max(0, int(round(sim_time * 1e6)))

        def set_phase(self, phase: str) -> None:
            self.get_logger().info(f"phase {self.phase} -> {phase}")
            self.phase, self.phase_since = phase, self.sim_now()

        def state_estimate(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
            last = self.odom.latest()
            if last is None:
                return None
            return last[1], last[2], E.yaw_from_quat(last[3])

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
            if self.setpoints:
                self.setpoint_gap_max = max(self.setpoint_gap_max, now - self.setpoints[-1][0])
            self.setpoints.append((now, np.array([*vel, yaw_rate], dtype=np.float64)))

        def setpoint_before(self, t: float) -> np.ndarray:
            """The setpoint dispatched most recently at or before ``t`` (a_bar_t, NED), zero if none."""
            for when, sp in reversed(self.setpoints):
                if when <= t:
                    return sp
            return np.zeros(4)

        # ---------------- callbacks ----------------
        def _on_status(self, msg: Any) -> None:
            self.status, self.status_wall = msg, time.monotonic()

        def _on_odometry(self, msg: Any) -> None:
            pos, vel, q = np.array(msg.position, float), np.array(msg.velocity, float), np.array(msg.q, float)
            rates = np.array(msg.angular_velocity, float)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(q))):
                return
            if msg.pose_frame != msg.POSE_FRAME_NED:
                self.get_logger().error(f"odometry pose_frame {msg.pose_frame} is not NED", throttle_duration_sec=5.0)
                return
            if msg.velocity_frame == msg.VELOCITY_FRAME_BODY_FRD:
                vel = E.quat_to_rotmat(q) @ vel
            elif msg.velocity_frame != msg.VELOCITY_FRAME_NED:
                self.get_logger().error(f"odometry velocity_frame {msg.velocity_frame} unsupported", throttle_duration_sec=5.0)
                return
            self.odom.add(msg.timestamp_sample * 1e-6, pos, vel, q, rates)
            self._drain()

        def _on_rgbd(self, rgb_msg: Any, depth_msg: Any) -> None:
            stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
            self.frames_seen += 1
            self.last_frame_sim, self.last_frame_wall = stamp, time.monotonic()
            if rgb_msg.encoding not in ("rgb8", "bgr8") or depth_msg.encoding != "32FC1":
                self.fail_run(f"unexpected encodings {rgb_msg.encoding} / {depth_msg.encoding} (want rgb8 / 32FC1)")
                return
            if (rgb_msg.height, rgb_msg.width) != E.IMAGE_SIZE or (depth_msg.height, depth_msg.width) != E.IMAGE_SIZE:
                self.fail_run(f"camera gives {rgb_msg.width}x{rgb_msg.height} RGB and {depth_msg.width}x{depth_msg.height} "
                              f"depth; Aero-VPA needs one aligned {E.IMAGE_SIZE[1]}x{E.IMAGE_SIZE[0]} RGB-D sensor "
                              "(spawn the x500_vpa model)")
                return
            if self.phase != "MISSION" or not self.episode_open or stamp < self.controller.t0:
                return
            rgb = decode_image(rgb_msg, 3, np.uint8)
            if rgb_msg.encoding == "bgr8":
                rgb = rgb[..., ::-1]
            depth = decode_image(depth_msg, 1, np.float32)
            self.pending.append((stamp, np.ascontiguousarray(rgb), depth.copy()))
            self._drain()

        def _drain(self) -> None:
            """Record pending frames whose capture time is now bracketed by odometry."""
            while self.pending and self.episode_open:
                stamp, rgb, depth = self.pending[0]
                latest = self.odom.latest()
                if latest is not None and stamp > latest[0]:
                    return                                     # wait for odometry beyond the capture time
                self.pending.popleft()
                est = self.odom.at(stamp, self.args.max_odometry_gap)
                if est is None:
                    self.episode_stats["dropped_frames"] += 1
                    continue
                pos, vel, q, rates, age = est
                sample = E.record_frame(stamp, rgb, depth, pos, vel, q, rates, self.setpoint_before(stamp),
                                        self.controller, self.tracker, self.thresholds, age)
                self.writer.send("append", sample)
                self.episode_stats["frames"] += 1

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
            missing = []
            if self.ros_now() <= 0.0:
                missing.append("/clock (ros_gz_bridge)")
            if self.odom.latest() is None:
                missing.append(self.args.odometry_topic)
            if self.status is None:
                missing.append(self.args.status_topic)
            if self.frames_seen == 0:
                missing.append(f"synchronised {self.args.rgb_topic} + {self.args.depth_topic}")
            if not missing:
                skew = self.odom.latest()[0] - self.last_frame_sim
                if abs(skew) > 1.0:
                    self.fail_run(f"PX4 odometry stamps are {skew:+.1f} s from the image stamps: PX4 is not stamping in "
                                  "simulation time; start PX4 with PX4_PARAM_UXRCE_DDS_SYNCT=0 (run_headless_sitl.sh does)")
                    return
                self.ready_logged = True
                self.phase_since = self.sim_now()
                self.get_logger().info(f"inputs ready after {waited:.1f} s (odometry - image stamp {1e3 * skew:+.0f} ms)")
            elif waited > self.args.startup_timeout:
                self.fail_run("no data after %.0f s on: %s" % (waited, ", ".join(missing)))
            elif int(waited) % 5 == 0:
                self.get_logger().info("waiting for: " + ", ".join(missing))

        # ---------------- episode bookkeeping ----------------
        def fail_run(self, reason: str, code: int = 1) -> None:
            if self.exit_code is None:
                self.get_logger().error(reason)
                if self.episode_open:
                    self.end_episode(False, "aborted: " + reason)
                self.exit_code = code

        def begin_episode(self, now: float, yaw: float) -> None:
            trajectory = E.ExpertTrajectory(self.plan_spline, yaw)
            self.controller = E.ExpertController(trajectory, now, E.CommandLimits(speed_h=self.speed))
            self.tracker = E.WaypointTracker(self.mission)
            self.writer.send("begin")
            self.episode_open = True
            self.pending.clear()
            self.episode_stats = {"frames": 0, "dropped_frames": 0}
            self.setpoint_gap_max = 0.0
            self.attempts += 1
            self.get_logger().info(f"episode {self.attempts}: {len(self.mission)} waypoints, v_h {self.speed} m/s, "
                                   f"expert duration {trajectory.duration:.1f} s")

        def end_episode(self, success: bool, termination: str) -> None:
            self.episode_open = False
            attrs = {
                "waypoints_ned": self.mission.astype(np.float64), "speed_limit": float(self.speed),
                "success": bool(success), "termination": termination, "instruction": self.args.instruction,
                "world": os.environ.get("AERO_VPA_WORLD_NAME", ""), "world_json": self.args.world_json or "",
                "seed": int(self.args.seed), "expert_duration_s": float(self.controller.trajectory.duration),
                "dropped_frames": int(self.episode_stats.get("dropped_frames", 0)),
                "max_setpoint_gap_s": float(self.setpoint_gap_max),
            }
            if success or self.args.keep_failed:
                self.writer.send("finalize", attrs)
            else:
                self.writer.send("discard")
            if success:
                self.successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            self.get_logger().info(f"episode {self.attempts} {'succeeded' if success else 'failed'} ({termination}); "
                                   f"{self.episode_stats.get('frames', 0)} frames, "
                                   f"{self.successes}/{self.args.num_episodes} recorded")

        # ---------------- flight logic (simulation-time timer) ----------------
        def _on_control(self) -> None:
            if self.exit_code is not None:
                return
            try:
                self.writer.check()
                self._step()
            except Exception as exc:  # any error ends the run cleanly (the episode is closed first)
                self.fail_run(f"{type(exc).__name__}: {exc}")

        def _step(self) -> None:
            now = self.sim_now()
            est = self.state_estimate()
            st = self.status
            _, _, VehicleCommand, VehicleStatus = self.msgs
            armed = st is not None and st.arming_state == VehicleStatus.ARMING_STATE_ARMED
            offboard = st is not None and st.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

            if self.phase in ("WAIT", "ARM") or est is None:
                self.publish_setpoint(now, np.zeros(3), np.zeros(3), 0.0)
            if est is None or not self.ready_logged:
                return
            pos, vel, yaw = est
            elapsed = now - self.phase_since

            if self.phase == "WAIT":
                if not armed and st.pre_flight_checks_pass and elapsed > 1.5:
                    self.set_phase("ARM")
                elif elapsed > self.args.rearm_timeout:
                    raise RuntimeError(f"PX4 pre-flight checks did not pass within {self.args.rearm_timeout:.0f} s "
                                       f"({'armed' if armed else 'disarmed'}); restart the simulator")
            elif self.phase == "ARM":
                if armed and offboard:
                    target = pos + np.array([0.0, 0.0, -self.args.takeoff_altitude])
                    self.controller = E.ExpertController(self.planner.hold(pos, yaw, target=target, speed=1.0), now,
                                                         E.CommandLimits(speed_h=1.0))
                    self.set_phase("TAKEOFF")
                elif time.monotonic() - self.last_command_wall > 0.5:      # one command at a time, then re-check
                    if elapsed > 20.0:
                        raise RuntimeError("PX4 did not accept offboard mode and arming within 20 s")
                    if not offboard:
                        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # custom main mode 6: offboard
                    else:
                        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            elif self.phase in ("TAKEOFF", "HOVER", "PLAN", "ALIGN", "MISSION"):
                if not (armed and offboard):
                    if self.episode_open:
                        self.end_episode(False, "px4_left_offboard")
                    raise RuntimeError(f"PX4 left offboard (arming_state {st.arming_state}, nav_state {st.nav_state})")
                self._fly(now, pos, vel, yaw, elapsed)
            elif self.phase == "LAND":
                if offboard and self.controller is not None:          # hold position until PX4 takes over
                    v_cmd, a_ff, yr = self.controller.command(now, pos, yaw)
                    self.publish_setpoint(now, v_cmd, a_ff, yr)
                if not armed:
                    if self.successes >= self.args.num_episodes:
                        self.set_phase("DONE")
                        self.exit_code = 0
                    elif self.consecutive_failures >= self.args.max_failures:
                        self.get_logger().error(f"{self.consecutive_failures} consecutive failures; giving up")
                        self.set_phase("DONE")
                        self.exit_code = 2
                    else:
                        self.set_phase("WAIT")
                elif time.monotonic() - self.last_command_wall > 2.0:
                    if elapsed > 90.0:
                        raise RuntimeError("landing did not complete within 90 s")
                    self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        def _hover(self, now: float, pos: np.ndarray, yaw: float, yaw_target: Optional[float] = None) -> None:
            self.controller = E.ExpertController(self.planner.hold(pos, yaw, yaw_target=yaw_target), now,
                                                 E.CommandLimits(speed_h=1.0))

        def _fly(self, now: float, pos: np.ndarray, vel: np.ndarray, yaw: float, elapsed: float) -> None:
            if self.phase == "TAKEOFF":
                done = self.controller.finished(now) and np.linalg.norm(
                    pos - self.controller.trajectory.reference(self.controller.elapsed(now))[0]) < 0.3
                if done and np.linalg.norm(vel) < 0.3:
                    self._hover(now, pos, yaw)
                    self.set_phase("PLAN")
                elif elapsed > 30.0:
                    raise RuntimeError("take-off did not reach the target altitude within 30 s")
            elif self.phase == "HOVER":
                if elapsed > 1.0:
                    self.set_phase("PLAN")
            elif self.phase == "PLAN":
                if self.plan_future is None:
                    self.mission = E.sample_mission(self.world, self.rng, pos, tuple(self.args.num_waypoints),
                                                    tuple(self.args.spacing), tuple(self.args.altitude))
                    self.speed = float(self.rng.choice(self.args.speed_limits))
                    self.plan_future = self.pool.submit(E.plan_mission_spline, self.args.world_json, pos, self.mission,
                                                        self.speed)
                elif self.plan_future.done():
                    future, self.plan_future = self.plan_future, None
                    try:
                        self.plan_spline = future.result()
                    except E.PlanningError as exc:
                        self.get_logger().warning(f"planning failed ({exc}); resampling the mission")
                        return
                    heading = E.ExpertTrajectory(self.plan_spline, yaw).initial_heading()
                    self._hover(now, pos, yaw, yaw_target=heading if heading is not None else yaw)
                    self.set_phase("ALIGN")
                elif elapsed > 60.0:
                    raise RuntimeError("expert planning took more than 60 s")
            elif self.phase == "ALIGN":
                target = float(self.controller.trajectory.reference(np.inf)[3])
                if (abs(float(E.wrap_angle(target - yaw))) < 0.08 and elapsed > 0.5) or elapsed > 8.0:
                    start = np.asarray(self.plan_spline.evaluate(0.0))
                    if np.linalg.norm(pos - start) > 1.0:              # drifted while planning: replan from here
                        self.set_phase("PLAN")
                        return
                    self.begin_episode(now, yaw)
                    self.set_phase("MISSION")
            elif self.phase == "MISSION":
                self._check_mission(now, pos)
                if not self.episode_open:
                    return
            if self.phase == "MISSION" or self.controller is not None:
                v_cmd, a_ff, yr = self.controller.command(now, pos, yaw)
                self.publish_setpoint(now, v_cmd, a_ff, yr)

        def _check_mission(self, now: float, pos: np.ndarray) -> None:
            ctl, reason = self.controller, None
            if len(self.world) and float(self.world.distance(pos)) < 0.35:
                reason = "collision"
            elif np.linalg.norm(pos - ctl.trajectory.reference(ctl.elapsed(now))[0]) > 2.5:
                reason = "tracking_error"
            elif now - self.last_frame_sim > 0.5:
                reason = "camera_timeout"
            elif ctl.elapsed(now) > 1.5 * ctl.trajectory.duration + 15.0:
                reason = "timeout"
            yaw = E.yaw_from_quat(self.odom.latest()[3])
            if reason is not None:
                self.end_episode(False, reason)
                if reason == "collision":     # the vehicle may be damaged in the simulation: end the run here
                    self.get_logger().error("collision: ending the run; restart the simulator and use --append")
                    self.set_phase("DONE")
                    self.exit_code = 3
                    return
                self._hover(now, pos, yaw)
                self.set_phase("LAND")
                return
            if self.tracker.task_complete and ctl.finished(now, settle=1.0):
                self.end_episode(True, "complete")
                self._hover(now, pos, yaw)
                self.set_phase("LAND" if self.successes >= self.args.num_episodes else "HOVER")

        def shutdown(self) -> None:
            if self.episode_open:
                self.end_episode(False, "interrupted")
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.writer.close()

    sys.setswitchinterval(0.001)  # keep the 100 Hz timer responsive while other threads hold the GIL
    rclpy.init()
    node = ExpertCollector()
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
    print(f"[collector] {node.successes} successful episodes in {args.output} (exit {code})", flush=True)
    return code


# ---------------------------------------------------------------------------------------------
# Offline dry run
# ---------------------------------------------------------------------------------------------
def run_dry(args: argparse.Namespace) -> int:
    world = E.load_world(args.world_json) if args.world_json else E.synthetic_forest(seed=args.seed)
    if args.append:
        raise SystemExit("--append is not supported with --dry_run")
    cfg = E.DryRunConfig(num_episodes=args.num_episodes, seed=args.seed, speed_limits=tuple(args.speed_limits),
                         num_waypoints=tuple(args.num_waypoints), spacing=tuple(args.spacing),
                         altitude=tuple(args.altitude))
    names = E.dry_run(args.output, world, cfg, _file_attrs(args, "dry_run (kinematic model, ray-cast RGB-D)"))
    print(f"[collector] dry run wrote {len(names)} episodes to {args.output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_dry(args) if args.dry_run else run_ros(args)


if __name__ == "__main__":
    sys.exit(main())
